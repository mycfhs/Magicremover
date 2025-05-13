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
            self.step_store[key].append(attn)

    def all_store(self, t):
        self.attention_all_step[f'{t}'] = self.step_store
        self.phi_all_step[f'{t}'] = self.phi
        self.set_empty_store()
        self.place_in_cross=0
    
    def set_empty_store(self):
        self.step_store =  {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
        
    def reset(self):
        self.set_empty_store()
        self.attention_all_step = {}
        self.phi=None 
        self.phi_all_step = {}
        self.noise_to_add = {}
        self.swap_attn=False
        self.place_in_cross = 0
        self.cur_t = 999
        
    def save_noise_to_add(self, noise_to_add, t):
        self.noise_to_add[f'{t}'] = noise_to_add
    

def normalize(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min())




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
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
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

        
    @torch.no_grad()
    def text2image_ldm_stable(
        self,
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
        self.register_attention_control()
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
        
        context = [uncond_embeddings, text_embeddings]
        if not low_resource:
            context = torch.cat(context)
    

        # set timesteps
        #extra_set_kwargs = {"offset": 1}
        self.scheduler.set_timesteps(num_inference_steps)#, **extra_set_kwargs)
        self.ddpm_scheduler.set_timesteps(num_inference_steps)#, **extra_set_kwargs)
        
        ########ddim invert
        img_path = "dreambooth/dataset/colorful_sneaker/03.jpg"
        img = Image.open(img_path).convert('RGB')
        img = torch.tensor(np.array(img),device=device).permute(2,0,1).unsqueeze(0)/255
        #crop_transform = torchvision.transforms.CenterCrop((512, 512)) 
        #img = crop_transform(img)
        img = torch.nn.functional.interpolate(img,(512,512),mode='bilinear', align_corners=False)
        latent = self.encode_imgs(img).detach().requires_grad_(False)     
        
        ddim_latent = self.ddim_invert(img, context, latent, num_inference_steps, guidance_scale)
        latent = ddim_latent[-1]
        
        
        
    
        latent, init_latents = self.init_latent(latent, height, width, generator, batch_size)
        latents=init_latents
        
        for t in self.scheduler.timesteps:
            latents = self.diffusion_step(latents, context, t, guidance_scale,  low_resource, save_noise=False)
            self.save_attention.all_store(t)
        img_ori = self.decode_latents(latents)
        
        
        self.visualize(t=20)
        
        latents = init_latents
        self.save_attention.swap_attn=False
        plus_guidance=True
        for t in self.scheduler.timesteps:
            if t == guidance_step:
                plus_guidance=False 
                
            self.save_attention.cur_t = t
            latents = self.diffusion_step(latents, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance, save_noise=False)
            self.save_attention.set_empty_store()
            
        img_edit = self.decode_latents(latents)
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

                # 6. post-process
                if self.conv_norm_out:
                    sample = self.conv_norm_out(sample)
                    sample = self.conv_act(sample)
                    save_attention.phi = sample
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
                
                #################################################################               
                
                
                if is_cross :  
                    if save_attention.swap_attn==True:
                        cross_attn_ori = save_attention.attention_all_step[f'{save_attention.cur_t}'][place_in_unet+'_cross'][save_attention.place_in_cross]
                        cross_attn_ori = cross_attn_ori[cross_attn_ori.shape[0]//2:]
                        
                        
                        # res=int(cross_attn_ori.shape[1]**0.5)
                        # cross_attn_ori = cross_attn_ori.reshape(cross_attn_ori.shape[0],res,res,cross_attn_ori.shape[-1])
                        # cross_attn_ori[:,:,:,8] = torch.flip(cross_attn_ori[:,:,:,8],[2]) * 2
                        # cross_attn_ori = cross_attn_ori.reshape(cross_attn_ori.shape[0],int(res*res),cross_attn_ori.shape[-1])
                        
                        # mask = torch.zeros_like(cross_attn_ori,device=attn.device)
                        # attn[attn.shape[0]//2:] = attn[attn.shape[0]//2:] * mask  + cross_attn_ori * (1-mask) 
                
                        
                        mask = torch.zeros_like(cross_attn_ori,device=attn.device)
                        mask[:,:,save_attention.object_id_d]=1
                        attn = attn * mask * 1 + cross_attn_ori * (1-mask) 
                        
                        mask = torch.zeros_like(attn,device=attn.device)
                        mask[:,:,save_attention.object_id_d]=1
                        attn[attn.shape[0]//2:] = attn[attn.shape[0]//2:] * mask * 1 + cross_attn_ori * (1-mask) 
                        
                        if save_attention.place_in_cross < (len(save_attention.attention_all_step[f'{save_attention.cur_t}'][place_in_unet+'_cross'])-1):
                            save_attention.place_in_cross += 1
                        else:
                            save_attention.place_in_cross = 0 
                        
                        
                        
                        
                        
                        # attn_conditional=attn[attn.shape[0]//2:]
                        # h = attn_conditional.shape[0] // (batch_size//2)  
                        # attn_conditional = attn_conditional.reshape(batch_size//2, h, *attn_conditional.shape[1:])  
                        # attn_base, attn_replace = attn_conditional[:1], attn_conditional[1:]  
                        # #alpha_words = controller.cross_replace_alpha[controller.cur_step]  
                        # #attn_replace = attn_base  # controller.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace  
                        # masks = torch.ones_like(attn_base, device = attn_base.device)
                        # masks[:,:,:,2] = 0
                        # attn_replace = attn_base * masks + attn_replace * (1-masks)
                        # attn_conditional = torch.cat([attn_base, attn_replace], dim=0)  
                        # attn_conditional = attn_conditional.reshape(batch_size//2 * h, *attn_conditional.shape[2:]) 
                
                        # attn[attn.shape[0]//2:]=attn_conditional

                
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
                if low_resource:
                    noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=context[0])["sample"]
                    noise_prediction_text = self.unet(latents, t, encoder_hidden_states=context[1])["sample"]
                else:
                    latents_input = torch.cat([latents] * 2)
                    noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
                    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                
                #noise_pred = noise_prediction_text + guidance_scale * (noise_pred_uncond - noise_prediction_text)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
            
                loss = self.guidance(t)
                # loss.backward()
                # gradients = latents.grad
                gradients = torch.autograd.grad(loss, latents, retain_graph=False, create_graph=False)[0] 

                noise_pred = noise_pred + gradients # *  1000
        
        else:
            
            if low_resource:
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=context[0])["sample"]
                noise_prediction_text = self.unet(latents, t, encoder_hidden_states=context[1])["sample"]
            else:
                latents_input = torch.cat([latents] * 2)
                noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            
            #noise_pred = noise_prediction_text + guidance_scale * (noise_pred_uncond - noise_prediction_text)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        #latents = self.next_step(noise_pred, t, latents, self.save_attention.noise_to_add.get(f'{t}'))
        
        if save_noise:
            self.save_attention.save_noise_to_add(noise_pred, t)
        latents_next = latents.clone().detach()
  
        return latents_next

    def visualize(self,t):
        for i in range(981,0,-20):
            phi = self.save_attention.phi_all_step[f"{i}"][1:]
            phi = phi.sum(1)
            torchvision.utils.save_image(phi, f'visualize/phi/{i}.png')

    
        obj_id=5
        for t in self.scheduler.timesteps:
            item_all=torch.tensor([],device='cuda')
            for location in ["down","mid","up"]:
                for item in  self.save_attention.attention_all_step[f'{t}'][f"{location}_cross"]:
                    #if item.shape[1]==(32**2):
                        item=item[item.shape[0]//2:]
                        res= int(item.shape[1] ** 0.5)
                        item = item.reshape(item.shape[0],res,res,item.shape[-1])
                        item = item.permute(0, 3, 1, 2)  
                        item = item.sum(0)
                        item = torch.nn.functional.interpolate(item[obj_id,:,:].unsqueeze(0).unsqueeze(0),(64,64),mode='bilinear', align_corners=False)
                        item_all = torch.concat((item_all,item),dim=0)
            item_all = torch.nn.functional.interpolate(item_all,(512,512),mode='bilinear', align_corners=False)
            item_all = item_all.sum(0)/item_all.shape[0]
            item_all = normalize(item_all)
            torchvision.utils.save_image(item_all,f'visualize/attn/{t}.png')
            #item_thresh = self.sigmod(100 * (normalize(item_all) - 0.5))
            #torchvision.utils.save_image(torch.concat((item_all, item_thresh),dim=2),f'visualize/attn/{t}.png')
        pass

    def guidance(self, t):
        object_id_d=self.save_attention.object_id_d
        object_id_s=self.save_attention.object_id_s

       
        attn_ori=[]
        attn_edit=[]
        for location in ["down","mid","up"]:
            for item in self.save_attention.step_store[f"{location}_cross"]:
                item=item[item.shape[0]//2:]
                
                res= int(item.shape[1] ** 0.5)
                item = item.reshape(item.shape[0],res,res,item.shape[-1])
                item = item.permute(0, 3, 1, 2)  
                item = torch.nn.functional.interpolate(item,(64,64),mode='bilinear', align_corners=False)
                attn_edit.append(item)
        
        ori_step_store = self.save_attention.attention_all_step[f'{t}']
        
        for location in ["down","mid","up"]:
            for item in ori_step_store[f"{location}_cross"]:
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
        
        ### appearance    
        cross_final_edit  =  self.save_attention.step_store['up_cross'][-1]
        res = int(cross_final_edit.shape[1] ** 0.5)
        cross_final_edit = cross_final_edit.reshape(cross_final_edit.shape[0],res,res,cross_final_edit.shape[-1])
        cross_final_edit = cross_final_edit.permute(0,3,1,2)
        cross_final_edit = torch.nn.functional.interpolate(cross_final_edit,(64,64),mode='bilinear', align_corners=False)
        #cross_final_edit = torch.concat((cross_final_edit[:,:object_id,:,:], cross_final_edit[:,object_id+1:,:,:]),dim=1)
        
        phi_edit = self.save_attention.phi[1:]
        #a_edit = torch.einsum('ijml,knml->ijkn',cross_final_edit[:,object_id_s,:,:],phi_edit)  
        a_edit = torch.einsum('ijml,knml->ijkn',cross_final_edit,phi_edit)  
        
        cross_final_ori = ori_step_store['up_cross'][-1]
        cross_final_ori = cross_final_ori.reshape(cross_final_ori.shape[0],res,res,cross_final_ori.shape[-1])
        cross_final_ori = cross_final_ori.permute(0,3,1,2)
        cross_final_ori = torch.nn.functional.interpolate(cross_final_ori,(64,64),mode='bilinear', align_corners=False)
        #cross_final_ori = torch.concat((cross_final_ori[:,:object_id,:,:], cross_final_ori[:,object_id+1:,:,:]),dim=1)
        phi_ori = self.save_attention.phi_all_step[f'{t}'][1:]
        #a_ori = torch.einsum('ijml,knml->ijkn',cross_final_ori[:,object_id_s,:,:],phi_ori)  
        a_ori = torch.einsum('ijml,knml->ijkn',cross_final_ori,phi_ori)  
        
        
        
        
        
        #### for target object        
        ### shape
        attn_ori_obj = attn_ori[object_id_d]
        attn_ori_thresh = normalize(self.sigmod(1*(normalize(attn_ori_obj)-0.5)))
        
        attn_edit_obj = attn_edit[object_id_d]
        attn_edit_thresh = normalize(self.sigmod(1*(normalize(attn_edit_obj)-0.5)))
        
        #### object size
        ori_size = torch.sum(attn_ori_thresh) / torch.numel(attn_ori_thresh)
        edit_size = torch.sum(attn_edit_thresh) / torch.numel(attn_edit_thresh)
        
        
        
        ### object position
        cur_cen = tensor_centroid(attn_edit_obj)#_thresh)
        ori_cen = tensor_centroid(attn_ori_obj)#thresh)
        target_cen=30,30
        
        
        # phi_ori = phi_ori.sum(1)
        # phi_edit = phi_edit.sum(1)
        # attn_ori_obj = attn_ori_obj * phi_ori
        # attn_edit_obj = attn_edit_obj * phi_edit
        
        cur_point = torch.tensor([296,240])#[45,57]#[38,30]#[39,30] #[30,39]  #[57,45]
        target_point = torch.tensor([266,159])#torch.tensor([287,328]) #[45,30] # [39,33] # [34,16]
        # cur_point = torch.tensor([180,207])#[45,57]#[38,30]#[39,30] #[30,39]  #[57,45]
        # target_point = torch.tensor([186,249])#torch.tensor([287,328]) #[45,30] # [39,33] # [34,16]
        d=target_point-cur_point
        phi_ori = torch.nn.functional.interpolate(phi_ori,(512,512),mode='bilinear', align_corners=False)
        phi_edit = torch.nn.functional.interpolate(phi_edit,(512,512),mode='bilinear', align_corners=False)
        # attn_ori_obj = torch.nn.functional.interpolate(attn_ori_obj.unsqueeze(0), (512,512),mode='bilinear', align_corners=False)
        # attn_edit_obj = torch.nn.functional.interpolate(attn_edit_obj.unsqueeze(0), (512,512),mode='bilinear', align_corners=False)
        
        # ori_a = phi_ori * attn_ori_obj.unsqueeze(1).repeat(1,phi_ori.shape[1],1,1)
        # edit_a = phi_edit * attn_edit_obj.unsqueeze(1).repeat(1,phi_edit.shape[1],1,1)
        # ori_a = torch.nn.functional.interpolate(ori_a, (512,512),mode='bilinear', align_corners=False)
        # edit_a = torch.nn.functional.interpolate(edit_a, (512,512),mode='bilinear', align_corners=False)
        loss = 0
        for qi in neighbor(int(cur_point[0]), int(cur_point[1]), 1):
            f1 = bilinear_interpolate_torch(phi_ori, qi[0], qi[1]).detach()
            f2 = bilinear_interpolate_torch(phi_edit, qi[0] + d[0], qi[1] + d[1])
            loss += self.l1_loss(f2, f1)
        
        mask = torch.ones_like(phi_ori,device=phi_ori.device)
        mask[:,:,96:367,80:383]=0
        #mask[:,:,39:240,128:349]=0
        loss += self.l1_loss(phi_ori*mask, phi_edit*mask) * 1
        
        #loss = self.l1_loss(phi_ori[:,:,cur_point[0]-1:cur_point[0]+2,cur_point[1]-1:cur_point[1]+2], phi_edit[:,:,target_point[0]-1:target_point[0]+2,target_point[1]-1:target_point[1]+2] ) 
        #loss = self.l1_loss(attn_ori_obj[:,cur_point[0]-2:cur_point[0]+3,cur_point[1]-2:cur_point[1]+3], attn_edit_obj[:,target_point[0]-2:target_point[0]+3,target_point[1]-2:target_point[1]+3] ) * 200
        #loss = self.l1_loss(attn_ori_obj[:,cur_point[0]-1:cur_point[0]+2,cur_point[1]-1:cur_point[1]+2], attn_edit_obj[:,target_point[0]-1:target_point[0]+2,target_point[1]-1:target_point[1]+2] ) * 200
        #loss =  (torch.abs(ori_cen[0]  - cur_cen[0] ) + torch.abs(ori_cen[1] -10 - cur_cen[1]))*7
        #loss =  (torch.abs(target_cen[0]  - cur_cen[0]) + torch.abs(target_cen[1]  - cur_cen[1])) * 0.1
        #loss += torch.abs(edit_size-ori_size) * 10
        # #loss = self.l1_loss(attn_ori_thresh_s, attn_edit_thresh_s)*30 
        #loss += self.l1_loss(a_edit,a_ori)  * 100
        
        # mask = torch.ones_like(attn_ori, device=attn_ori.device)
        # mask[object_id_d] = 0 
        #loss += self.l1_loss(attn_ori*mask, attn_edit*mask) * 1000
        #loss += self.l1_loss(attn_ori_thresh, attn_edit_thresh) * 10
        # #loss = l1_loss(cur_cen,target_cen)
        # #loss += self.l1_loss(attn_ori, attn_edit)
        # #loss += self.l1_loss(a_edit,a_ori) * 30
        
        loss = loss * ((1-self.scheduler.alphas_cumprod[t]) ** 0.5)  * 300
        
        return loss


    def next_step(self, noise_pred, timestep, sample, noise_to_add=None):
        timestep, next_timestep =  timestep, max(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 0)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        
        if noise_to_add is None:
            noise_to_add = noise_pred
        
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 *  noise_to_add 
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    
    def pre_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
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
                #noise_pred = self.unet(latent, t, encoder_hidden_states=text_embeds).sample
                model_input = torch.concat([latent]*2, dim=0)
                noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeds)["sample"]
                noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
                latent = self.pre_step(noise_pred, t, latent)
                all_latent.append(latent)
        return all_latent

def visualize(self,t):
    for i in range(981,0,-20):
        phi = sd.save_attention.phi_all_step[f"{i}"][1:]
        phi = phi.sum(1)
        torchvision.utils.save_image(phi, f'visualize/phi/{i}.png')


    obj_id=5
    for t in range(981,0,-20):
        item_all=torch.tensor([],device='cuda')
        for location in ["down","mid","up"]:
            for item in  self.save_attention.attention_all_step[f'{t}'][f"{location}_cross"]:
                    item=item[item.shape[0]//2:]
                    res= int(item.shape[1] ** 0.5)
                    item = item.reshape(item.shape[0],res,res,item.shape[-1])
                    item = item.permute(0, 3, 1, 2)  
                    item = item.sum(0)
                    item_all = torch.concat((item_all,item[obj_id].unsqueeze(0)),dim=0)
        torchvision.utils.save_image(item_all.sum(0),f'visualize/attn_32/{t}.png')
        
    pass



def neighbor(x, y, d):
    points = []
    for i in range(x - d, x + d):
        for j in range(y - d, y + d):
            points.append(torch.tensor([i, j]).float().to('cuda'))
    return points

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





if __name__=='__main__':


    init_seed = 287
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    
    
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    # parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    # parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    # parser.add_argument('-H', type=int, default=512)
    # parser.add_argument('-W', type=int, default=512)
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--steps', type=int, default=50)
    # opt = parser.parse_args()
    
    device = torch.device('cuda')
    sd = StableDiffusion(device, fp16=False, vram_O=False,sd_version='2.1',object_id_d=[5], object_id_s=[8])
    
    #prompts = ["a photo of a dog and a ball on the grass"]
    prompts = ["a photo of a sneaker"]
    #["a photo of a burger and an ice cream cone floating in the ocean"]
    #[ "a photo of a dog and a ball on the grass"]
    #while True:
    images, x_t = sd.text2image_ldm_stable(prompts, latent=None, num_inference_steps=100, guidance_scale=1, generator=None, low_resource=False,  guidance_step=201)
    #x_t = torch.randn((1,4,64,64),device=device)
    
    
    #images, x_t = text2image_ldm_stable(ldm_stable, prompts, latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=None, low_resource=LOW_RESOURCE)