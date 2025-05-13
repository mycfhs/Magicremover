import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from IPython.display import display
from tqdm.notebook import tqdm
from torch.autograd import grad
import torch.nn as nn
import torchvision
from diffusers import DDPMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput, logger
import random
import torch.nn.functional as F
from torch.optim.adam import Adam

class Save_Attention(nn.Module):
    def __init__(self, object_id_d, object_id_s):
        super(Save_Attention, self).__init__() 
        self.reset()
        self.object_id_d = object_id_d
        self.object_id_s = object_id_s
        

    def save_attn(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
        #if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[self.state][key].append(attn)
            
    
    def save_self_attn(self, k, v, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if is_cross==False:
            #if k.shape[1] <= 64 ** 2:  # avoid memory overhead
            #if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[self.state][key]['k'].append(k)
            self.step_store[self.state][key]['v'].append(v)

    def all_store(self, t):
        self.attention_all_step[f'{t}'] = self.step_store
        self.phi_all_step[f'{t}'] = self.phi
        self.set_empty_store()
        self.place_in_cross=0
    
    def set_empty_store(self):
        self.step_store =  {"ori": {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": {"k":[],"v":[]},  "mid_self": {"k":[],"v":[]},  "up_self": {"k":[],"v":[]}},
                            "edit": {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": {"k":[],"v":[]},  "mid_self": {"k":[],"v":[]},  "up_self": {"k":[],"v":[]}}
                }
        self.phi={}
        
    def set_edit_empty_store(self):
        self.step_store["edit"] =  {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": {"k":[],"v":[]},  "mid_self": {"k":[],"v":[]},  "up_self": {"k":[],"v":[]}}
                
        self.phi={}
        self.place_in_cross=0
        
    def reset(self):
        self.set_empty_store()
        self.attention_all_step = {}
        self.phi={"ori":[], "edit":[]}
        self.phi_all_step = {}
        self.noise_to_add = {}
        self.swap_attn=False
        self.self_attn=False
        self.place_in_cross = 0
        self.cur_t = 999
        self.state = "ori"
        
    def save_noise_to_add(self, noise_to_add, t):
        self.noise_to_add[f'{t}'] = noise_to_add
    

def normalize(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min())


def bilinear_interpolate_torch(im, y, x):
    """
    im : B,C,H,W
    y : 1,numPoints -- pixel location y float
    x : 1,numPOints -- pixel location y float
    """
    device = im.device
    
    x0 = torch.floor(x).long().to(device)
    x1 = x0 + 1

    y0 = torch.floor(y).long().to(device)
    y1 = y0 + 1

    wa = ((x1.float() - x) * (y1.float() - y)).to(device)
    wb = ((x1.float() - x) * (y - y0.float())).to(device)
    wc = ((x - x0.float()) * (y1.float() - y)).to(device)
    wd = ((x - x0.float()) * (y - y0.float())).to(device)
    # Instead of clamp
    x1 = x1 - torch.floor(x1 / im.shape[3]).int().to(device)
    y1 = y1 - torch.floor(y1 / im.shape[2]).int().to(device)
    Ia = im[:, :, y0, x0]
    Ib = im[:, :, y1, x0]
    Ic = im[:, :, y0, x1]
    Id = im[:, :, y1, x1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd
    



def tensor_centroid(tensor):  
    tensor = tensor.squeeze(0)
    height, width = tensor.shape  
    y_indices, x_indices = torch.meshgrid(torch.arange(height), torch.arange(width))  
    x_indices = x_indices.to(tensor.device)  
    y_indices = y_indices.to(tensor.device)  
  
    x_mass = torch.sum(x_indices * tensor)  
    y_mass = torch.sum(y_indices * tensor)  
    total_mass = torch.sum(tensor)  
  
    centroid_x = x_mass / total_mass  
    centroid_y = y_mass / total_mass  
  
    return centroid_x, centroid_y  






class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98], object_id_d=[8], object_id_s=[8]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.ddpm_scheduler = DDPMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.mse = nn.MSELoss()
        

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        # self.min_step = int(self.num_train_timesteps * t_range[0])
        # self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas)
        
        self.l1_loss = torch.nn.L1Loss()
        self.sigmod = nn.Sigmoid()
        
        print(f'[INFO] loaded stable diffusion!')
        
        self.save_attention = Save_Attention(object_id_d=object_id_d, object_id_s=object_id_s )        
        
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents       


    def pre_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = timestep, max(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 0)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] 
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_timestep >=0 else self.scheduler.alpha_cumprod[0]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

   
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = timestep, min(timestep + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] 
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_timestep <= 999 else self.scheduler.alpha_cumprod[-1]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def ddim_invert(self,img, text_embeds,latent,num_inference_steps,guidance_scale=7.5):
        
        all_latent = [latent]
        latent = latent.clone().detach()
        with torch.no_grad():
            for i in range(num_inference_steps):
                t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
                noise_pred = self.unet(latent, t, encoder_hidden_states=text_embeds).sample
                # model_input = torch.concat([latent]*2, dim=0)
                # noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeds)["sample"]
                # noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
                latent = self.next_step(noise_pred, t, latent)
                all_latent.append(latent)
        return all_latent

        
    @torch.no_grad()
    def drag_diffusion(
        self,
        img,
        prompt: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
        height = 512,
        width = 512,
        guidance_step=801
    ):

        ddim_idx=40
        t_idx=781
        
        def neighbor(x, y, d):
            points = []
            for i in range(x - d, x + d):
                for j in range(y - d, y + d):
                    points.append(torch.tensor([i, j]).float().to(latent.device))
            return points        

        batch_size = len(prompt)
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        context = text_embeddings #[uncond_embeddings, text_embeddings]

            
        
        # set timesteps
        #extra_set_kwargs = {"offset": 1}
        self.scheduler.set_timesteps(num_inference_steps)#, **extra_set_kwargs)
        self.ddpm_scheduler.set_timesteps(num_inference_steps)#, **extra_set_kwargs)
        
        ### ddim inversion
        print('ddim inversion')
        latent = self.encode_imgs(img.half()).detach().requires_grad_(False)
        ddim_latent = self.ddim_invert(img, context, latent, num_inference_steps, guidance_scale)
        print('ddim inversion done')
        
        #prepare to save original cross attention
        self.register_attention_control()

    
        ### ddim_latent[41] is x_781
        x_t = ddim_latent[ddim_idx].clone().detach().requires_grad_(True)
        optimizer = Adam([x_t], lr=1e-2, eps=1e-4)
    
    
        step=0
        self.save_attention.state='ori'
        self.save_attention.self_attn=False
        self.unet(ddim_latent[ddim_idx], t_idx, encoder_hidden_states=context).sample
        f_0 = self.save_attention.phi['ori']
        f_0 = F.interpolate(f_0, size=(512, 512), mode='bilinear', align_corners=False)
        
        
        point= [[168,317],[161,248]]#[[180,207], [186,289]] #[[249,296],[180,299]] 244
        #point= [[97,242],[141,243]]
        point=torch.tensor(point,dtype=torch.int)
        p_c = point[0]
        p_t = point[1]
        p_c0 = point[0]
        
    
        ### reconstrut loss on x_t_1
        x_t_512_ori = F.interpolate(ddim_latent[ddim_idx], size=(512, 512), mode='bilinear', align_corners=False)
        x_t_1_512 = F.interpolate(ddim_latent[ddim_idx-1], size=(512, 512), mode='bilinear', align_corners=False)

        video=img
        mask = torch.ones_like(x_t_512_ori,device=x_t_512_ori.device)
        mask[:,:,16:234,84:436]=0
        #mask[:,:,87:150,205:276]=0
        
        
        self.save_attention.state='edit'
        self.save_attention.self_attn=True
        while(torch.norm((p_t-p_c).type(torch.float))>3):
            d = (p_t - p_c) / torch.norm((p_t.float() - p_c.float()))

            
            with torch.enable_grad():
                
                #### motion  supervision
                self.save_attention.state='edit'
                self.save_attention.self_attn=True
                noise_pred = self.unet(x_t, t_idx, encoder_hidden_states=context).sample
                x_t_pre = self.scheduler.step(noise_pred, t_idx, x_t)['prev_sample']
                f = self.save_attention.phi['edit']
                f =  F.interpolate(f, size=(512, 512), mode='bilinear', align_corners=False)

                loss = 0
                for qi in neighbor(int(p_c[0]), int(p_c[1]), 1):
                    f1 = bilinear_interpolate_torch(f, qi[0], qi[1]).detach()
                    f2 = bilinear_interpolate_torch(f, qi[0] + d[0], qi[1] + d[1])
                    loss += self.l1_loss(f2, f1) 
                
                
                x_t_pre_512 =  F.interpolate(x_t_pre, size=(512, 512), mode='bilinear', align_corners=False)
                # mask = torch.ones_like(x_t_pre_512,device=x_t_pre_512.device)
                # mask[:,:,16:234,84:436]=0
                
                loss += self.l1_loss(x_t_pre_512  * mask, x_t_1_512  * mask) * 3000  #.1
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.save_attention.set_edit_empty_store()

            
            ### point tracking
            self.unet(x_t.clone().detach(), t_idx, encoder_hidden_states=context).sample
            f = self.save_attention.phi['edit']
            f =  F.interpolate(f, size=(512, 512), mode='bilinear', align_corners=False)
            pi = p_c0
            # f = F0[..., int(pi[0]), int(pi[1])]
            f0 = bilinear_interpolate_torch(f_0, pi[0].float(), pi[1].float())
            minv = 1e4
            minx = 1e4
            miny = 1e4
            for qi in neighbor(int(p_c[0]), int(p_c[1]), 3):
                f2 = bilinear_interpolate_torch(f, qi[0], qi[1])
                v = torch.norm(f2 - f0, p=1)
                if v < minv:
                    minv = v
                    minx = int(qi[0])
                    miny = int(qi[1])
            p_c[0] = minx
            p_c[1] = miny
            self.save_attention.set_edit_empty_store()
        
            ######### save image
            print(f'step:{step}, loss:{loss},  target point:{p_t}, current point: {p_c}')
            if step % 100 ==0:
                self.save_attention.set_empty_store()
                latents_ori = ddim_latent[ddim_idx]
                latents_edit = x_t.clone().detach()
                for t in self.scheduler.timesteps:
                    if t > t_idx:
                        continue
                    self.save_attention.state = "ori"
                    self.save_attention.self_attn = False
                    latents_ori = self.diffusion_step(latents_ori, context, t, guidance_scale,  low_resource, save_noise=False)
                    #self.save_attention.all_store(t)
                    
                    self.save_attention.state = "edit"
                    if t > guidance_step:
                        plus_guidance = True 
                        self.save_attention.self_attn = True
                    else:
                        plus_guidance = False
                        self.save_attention.self_attn = False
                    latents_edit = self.diffusion_step(latents_edit, context, t, guidance_scale,  low_resource, save_noise=False)
                    
                    self.save_attention.set_empty_store()
                    
                img_ori = self.decode_latents(latents_ori)
                img_edit = self.decode_latents(latents_edit)
                torchvision.utils.save_image(torch.concat((img_ori,img_edit),dim=0),'1.png')
                
                self.save_attention.state='ori'
                self.save_attention.self_attn=False
                self.unet(ddim_latent[ddim_idx], t_idx, encoder_hidden_states=context).sample
            step+=1      
            
        
        
        latents_ori = ddim_latent[ddim_idx]
        latents_edit = x_t.clone().detach()
        for t in self.scheduler.timesteps:
            if t > t_idx:
                continue
            self.save_attention.state = "ori"
            self.save_attention.self_attn = False
            latents_ori = self.diffusion_step(latents_ori, context, t, guidance_scale,  low_resource, save_noise=False)
            #self.save_attention.all_store(t)
            
            self.save_attention.state = "edit"
            if t > guidance_step:
                plus_guidance = True 
                self.save_attention.self_attn = True
            else:
                plus_guidance = False
                self.save_attention.self_attn = False
            latents_edit = self.diffusion_step(latents_edit, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance, save_noise=False)
            
            self.save_attention.set_empty_store()
            
        img_ori = self.decode_latents(latents_ori)
        img_edit = self.decode_latents(latents_edit)
        torchvision.utils.save_image(torch.concat((img_ori,img_edit),dim=0),'1.png')
        self.save_attention.reset()
        return img_edit, latent
        
        



    def register_attention_control(self):
        save_attention = self.save_attention
        
        def unet_forward(self):
            def forward(
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                return_dict: bool = True,
            ) -> Union[UNet2DConditionOutput, Tuple]:
            
                default_overall_up_factor = 2**self.num_upsamplers

                # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
                forward_upsample_size = False
                upsample_size = None

                if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                    logger.info("Forward upsample size to force interpolation output size.")
                    forward_upsample_size = True

                # prepare attention_mask
                if attention_mask is not None:
                    attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                    attention_mask = attention_mask.unsqueeze(1)

                # 0. center input if necessary
                if self.config.center_input_sample:
                    sample = 2 * sample - 1.0

                # 1. time
                timesteps = timestep
                if not torch.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = sample.device.type == "mps"
                    if isinstance(timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(sample.device)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0])

                t_emb = self.time_proj(timesteps)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # but time_embedding might actually be running in fp16. so we need to cast here.
                # there might be better ways to encapsulate this.
                t_emb = t_emb.to(dtype=self.dtype)

                emb = self.time_embedding(t_emb, timestep_cond)

                if self.class_embedding is not None:
                    if class_labels is None:
                        raise ValueError("class_labels should be provided when num_class_embeds > 0")

                    if self.config.class_embed_type == "timestep":
                        class_labels = self.time_proj(class_labels)

                        # `Timesteps` does not contain any weights and will always return f32 tensors
                        # there might be better ways to encapsulate this.
                        class_labels = class_labels.to(dtype=sample.dtype)

                    class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

                    if self.config.class_embeddings_concat:
                        emb = torch.cat([emb, class_emb], dim=-1)
                    else:
                        emb = emb + class_emb

                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)
                    emb = emb + aug_emb

                if self.time_embed_act is not None:
                    emb = self.time_embed_act(emb)

                if self.encoder_hid_proj is not None:
                    encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

                # 2. pre-process
                sample = self.conv_in(sample)

                # 3. down
                down_block_res_samples = (sample,)
                for downsample_block in self.down_blocks:
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                        )
                    else:
                        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                    down_block_res_samples += res_samples

                if down_block_additional_residuals is not None:
                    new_down_block_res_samples = ()

                    for down_block_res_sample, down_block_additional_residual in zip(
                        down_block_res_samples, down_block_additional_residuals
                    ):
                        down_block_res_sample = down_block_res_sample + down_block_additional_residual
                        new_down_block_res_samples += (down_block_res_sample,)

                    down_block_res_samples = new_down_block_res_samples

                # 4. mid
                if self.mid_block is not None:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )

                if mid_block_additional_residual is not None:
                    sample = sample + mid_block_additional_residual

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                        )
                
                save_attention.phi[save_attention.state]=sample
                
                # 6. post-process
                if self.conv_norm_out:
                    sample = self.conv_norm_out(sample)
                    sample = self.conv_act(sample)
                    
                sample = self.conv_out(sample)

                if not return_dict:
                    return (sample,)

                return UNet2DConditionOutput(sample=sample)
            return forward
        
        self.unet.forward = unet_forward(self.unet)
        
        def ca_forward(self, place_in_unet):
            to_out = self.to_out
            if type(to_out) is torch.nn.modules.container.ModuleList:
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            def forward(x, encoder_hidden_states=None, attention_mask=None):
                batch_size, sequence_length, dim = x.shape
                h = self.heads
                q = self.to_q(x)
                is_cross = encoder_hidden_states is not None
                encoder_hidden_states = encoder_hidden_states if is_cross else x
                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.head_to_batch_dim(v)
                
                
                if is_cross==False:
                    if save_attention.self_attn==True:
                        k = save_attention.step_store["ori"][place_in_unet+'_self']['k'][save_attention.place_in_cross]
                        v = save_attention.step_store["ori"][place_in_unet+'_self']['v'][save_attention.place_in_cross]
                        
                        if save_attention.place_in_cross < (len(save_attention.step_store["ori"][place_in_unet+'_self']['k'])-1):
                            save_attention.place_in_cross += 1
                        else:
                            save_attention.place_in_cross = 0 
                

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

                if attention_mask is not None:
                    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    sim = torch.baddbmm(
                        attention_mask,
                        q,
                        k.transpose(-1, -2),
                        beta=1,
                        alpha=self.scale,
                    )

                # attention, what we cannot get enough of
                attn = sim.softmax(dim=-1)
                
                if is_cross:
                    if attn.shape[1] ** 0.5 <= 64:
                        save_attention.save_attn(attn, is_cross, place_in_unet)
    
                else:
                    save_attention.save_self_attn(k, v, is_cross, place_in_unet)
                #################################################################               
                
                
                if is_cross :  
                    if save_attention.swap_attn==True:
                        cross_attn_ori = save_attention.step_store["ori"][place_in_unet+'_cross'][save_attention.place_in_cross]
                        
                        mask = torch.zeros_like(attn,device=attn.device)
                        mask[:,:,save_attention.object_id_d]=1
                        attn = attn * mask * 1 + cross_attn_ori * (1-mask) 
                        
                        if save_attention.place_in_cross < (len(save_attention.attention_all_step[f'{save_attention.cur_t}'][place_in_unet+'_cross'])-1):
                            save_attention.place_in_cross += 1
                        else:
                            save_attention.place_in_cross = 0

                
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.batch_to_head_dim(out)
                return to_out(out)

            return forward


        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'Attention':
                net_.forward = ca_forward(net_, place_in_unet)
                
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    register_recr(net__, count, place_in_unet)
        

        sub_nets = self.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                register_recr(net[1], 0, "mid")

    def init_latent(self, latent, height, width, generator, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                dtype=torch.float16
            )
        
        #latent.requires_grad_(True)
        
        latents = latent.expand(batch_size,  self.unet.in_channels, height // 8, width // 8).to(self.device)
        return latent, latents

    def diffusion_step(self, latents, context, t, guidance_scale,  low_resource=False, plus_guidance=False, save_noise=True):
        #with torch.no_grad():
        if plus_guidance:
            latents.requires_grad_(True)
            with torch.enable_grad():
               
                latents_input = latents
                noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            
                loss = self.guidance(t)
                # loss.backward()
                # gradients = latents.grad
                gradients = torch.autograd.grad(loss, latents, retain_graph=False, create_graph=False)[0] 
                print(f'time_step:{t}, loss:{loss}, gradients_max:{gradients.max()}')
                noise_pred = noise_pred + gradients #*  50
        
        else:
            

            latents_input = latents
            noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
           
        latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        #latents = self.next_step(noise_pred, t, latents, self.save_attention.noise_to_add.get(f'{t}'))
        
        if save_noise:
            self.save_attention.save_noise_to_add(noise_pred, t)
        latents_next = latents.clone().detach()
  
        return latents_next

    def guidance(self, t):
        object_id_d=self.save_attention.object_id_d
        object_id_s=self.save_attention.object_id_s

       
        attn_ori=[]
        attn_edit=[]
        for location in ["down","mid","up"]:
            for item in self.save_attention.step_store['edit'][f"{location}_cross"]:
                item=item[item.shape[0]//2:]
                
                res= int(item.shape[1] ** 0.5)
                item = item.reshape(item.shape[0],res,res,item.shape[-1])
                item = item.permute(0, 3, 1, 2)  
                item = torch.nn.functional.interpolate(item,(64,64),mode='bilinear', align_corners=False)
                attn_edit.append(item)
        
        
        for location in ["down","mid","up"]:
            for item in self.save_attention.step_store['ori'][f"{location}_cross"]:
                item=item[item.shape[0]//2:]
                
                res= int(item.shape[1] ** 0.5)
                item = item.reshape(item.shape[0],res,res,item.shape[-1])
                item = item.permute(0, 3, 1, 2)  
                item = torch.nn.functional.interpolate(item,(64,64),mode='bilinear', align_corners=False)
                
                attn_ori.append(item)
                
        cross_num=len(attn_ori)
        attn_ori=torch.cat(attn_ori,dim=0)
        attn_ori=attn_ori.sum(0)/cross_num
        
        attn_edit=torch.cat(attn_edit,dim=0)
        attn_edit=attn_edit.sum(0)/cross_num
        
        
        #### for static object
        
        ### shape
        attn_ori_obj_s = attn_ori[object_id_s]
        attn_ori_thresh_s = normalize(self.sigmod(10*(normalize(attn_ori_obj_s)-0.5)))
        
        attn_edit_obj_s = attn_edit[object_id_s]
        attn_edit_thresh_s = normalize(self.sigmod(10*(normalize(attn_edit_obj_s)-0.5)))

        
        
        
        
        
        #### for target object        
        ### shape
        
        attn_ori_obj = attn_ori[object_id_d]
        attn_ori_thresh = normalize(self.sigmod(1*(normalize(attn_ori_obj)-0.5)))
        mask_ori = attn_ori_thresh > 0.5
        attn_ori_thresh = attn_ori_thresh * mask_ori
        
        attn_edit_obj = attn_edit[object_id_d]
        attn_edit_thresh = normalize(self.sigmod(1*(normalize(attn_edit_obj)-0.5)))
        mask_edit = attn_edit_thresh > 0.5
        attn_edit_thresh = attn_edit_thresh * mask_edit
        
        #### object size
        ori_size = torch.sum(attn_ori_thresh) / torch.numel(attn_ori_thresh)
        edit_size = torch.sum(attn_edit_thresh) / torch.numel(attn_edit_thresh)
        
        
        
        ### object position
        cur_cen = tensor_centroid(attn_edit_thresh)#_thresh)
        ori_cen = tensor_centroid(attn_ori_obj)#thresh)
        target_cen=torch.tensor([20,20],device=attn_edit.device)


        
        
        loss =  (torch.abs(target_cen[0]  - cur_cen[0] ) + torch.abs(target_cen[1] - cur_cen[1]))*20
        #loss = self.l1_loss(target_cen[0], cur_cen[0]) + self.l1_loss(target_cen[1], cur_cen[1])       
        
        #loss += self.l1_loss(edit_size,ori_size)  * 2500

        # #loss = self.l1_loss(attn_ori_thresh_s, attn_edit_thresh_s)*30 
        
        
        ### appearance   
        phi_edit = self.save_attention.phi[1][1:]
        a_edit = ((phi_edit * attn_edit_thresh.repeat(1,320,1,1)).sum(-1).sum(-1)) /  torch.sum(attn_edit_thresh)
        
        phi_ori = self.save_attention.phi[0][1:]
        #a_ori = torch.einsum('ijml,knml->ijkn',cross_final_ori[:,object_id_s,:,:],phi_ori)  
        a_ori = ((phi_ori * attn_ori_thresh.repeat(1,320,1,1)).sum(-1).sum(-1)) /  torch.sum(attn_ori_thresh)
        
        #loss += self.l1_loss(a_edit,a_ori)  * 2500
        
        

        
        loss = loss * ((1-self.scheduler.alphas_cumprod[t]) ** 0.5)
        
        return loss








if __name__=='__main__':


    init_seed = 279
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    

    
    device = torch.device('cuda')
    # object_id_d: index of word that needs to be edited, begin with 1, 0 is start text.
    # to do: process other word
    sd = StableDiffusion(device, fp16=True, vram_O=True, sd_version='2.1',object_id_d=[5], object_id_s=[8])
    
    prompts = ["a sitting dog"]
    
    rgb_path = '3.png'
    
    img = cv2.imread(rgb_path)
    img = cv2.resize(img,(512,512))
    
        
    img = Image.open(rgb_path).convert('RGB')
    img = torch.tensor(np.array(img),device=device).permute(2,0,1).unsqueeze(0)/255
    img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)

    # num_inference_steps: step of sampling
    # guidance_step: t > guidance_step, add disturbance to the noise
    images, x_t = sd.drag_diffusion(img, prompts, latent=None, num_inference_steps=50, guidance_scale=7.5, generator=None, low_resource=False,  guidance_step=101)
