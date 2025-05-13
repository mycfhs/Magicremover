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
from torch.optim.adam import Adam
import time 
import torch.nn.functional as F
import time  
import argparse

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
            self.step_store[self.state][key]['k'].append(k.clone().detach())
            self.step_store[self.state][key]['v'].append(v.clone().detach())

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
        self.phi={"ori":[], "edit":[]}
    
    
    def set_edit_empty_store(self):
        self.step_store['edit'] =   {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": {"k":[],"v":[]},  "mid_self": {"k":[],"v":[]},  "up_self": {"k":[],"v":[]}}
        self.phi['edit'] = []
        
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
    #return (tensor-tensor.min())/(tensor.max()-tensor.min())
    return (tensor-tensor.min().item())/(tensor.max().item()-tensor.min().item())



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
        self.l1_loss = torch.nn.L1Loss()
        self.sigmod = nn.Sigmoid()        

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        # self.min_step = int(self.num_train_timesteps * t_range[0])
        # self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas)
        

        
        print(f'[INFO] loaded stable diffusion!')
        
        self.save_attention = Save_Attention(object_id_d=object_id_d, object_id_s=object_id_s )        

    def reset_unet(self):
        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
        pipe.to(self.device)
        self.unet = pipe.unet
        del pipe    
        
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


    
    def get_cross_attn(self, state='ori'):   
        #mask = 0
        item_all_loc = 0
        item_all=torch.tensor([],device=self.device)
        
        for location in ["down","mid","up"]:
            for item in  self.save_attention.step_store[state][f"{location}_cross"]:
                #if item.shape[1]==(32**2):
                    item=item[item.shape[0]//2:]
                    res= int(item.shape[1] ** 0.5)
                    item = item.reshape(item.shape[0],res,res,item.shape[-1])
                    item = item.permute(0, 3, 1, 2)  
                    item = item.sum(0)
                    
                    # item_all = 0
                    # for obj_id in self.save_attention.object_id_d:
                    #     # item_norm =  F.normalize(item[obj_id], p=2, dim=(1, 2))  
                    #     # item_norm = item_norm.prod(0)
                    #     #item_norm = F.normalize(item[obj_id].prod(0), p=2, dim=(0,1))
                    #     item_norm = item[obj_id].prod(0)  #normalize(item[obj_id].prod(0))
                    #     item_all += F.interpolate(item_norm.unsqueeze(0).unsqueeze(0),(64, 64),mode='bilinear', align_corners=False)
                    
                    # item_all_loc += item_all
                    
                    item_all_word = 1
                    for obj_id in  self.save_attention.object_id_d:
                        item_norm = normalize(item[obj_id])
                        item_all_word  *= item_norm.prod(0)
                    item_all_word = torch.nn.functional.interpolate(item_all_word.unsqueeze(0).unsqueeze(0),(64, 64),mode='bilinear', align_corners=False)
                    item_all = torch.concat((item_all, item_all_word),dim=0)
        
        item_all = item_all.sum(0)/item_all.shape[0]    
                    
        return item_all
   
    @torch.no_grad()
    def real_image_ldm_stable(
        self,
        img_path,
        prompt: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        low_resource: bool = False,
        guidance_step=801,
        t_ratio=0.8
    ): 
        
        
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
        
        t_idx = self.scheduler.timesteps[num_inference_steps- int(t_ratio*num_inference_steps)]
        ddim_idx = int(t_ratio*num_inference_steps)

        
        img = Image.open(img_path).convert('RGB')
        img = torch.tensor(np.array(img),device=device).permute(2,0,1).unsqueeze(0)/255
        
        
        # img = Image.open("10.png").convert('RGB')
        # img = torch.tensor(np.array(img),device=device).permute(2,0,1).unsqueeze(0)/255
        # img = torch.nn.functional.interpolate(img,(512,512),mode='bilinear', align_corners=False)
        # mask = Image.open("12.jpg").convert('RGB')
        # mask = torch.tensor(np.array(mask),device=device).permute(2,0,1).unsqueeze(0)/255       
        # img = img * (1-mask) + 0.05 * mask
        
        
        # crop_transform = torchvision.transforms.CenterCrop((512, 512)) 
        # img = crop_transform(img)
        

        img = torch.nn.functional.interpolate(img,(512,512),mode='bilinear', align_corners=False)
        latent = self.encode_imgs(img).detach().requires_grad_(False)     
        #latent, init_latents = self.init_latent(latent, height, width, generator, batch_size)
        
        print('ddim inversion')
        ddim_latent = self.ddim_invert(img, context, latent, num_inference_steps, guidance_scale)

        print('uncond text optimiztion')
    
        uncond_embeds_list = self.null_optimization(ddim_latent, context, num_inner_steps=10, epsilon=1e-5, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        torch.save(uncond_embeds_list,'null_text_embeddings.pt')
        uncond_embeds_list = torch.load('null_text_embeddings.pt')
        print('done')  

        
        latent = ddim_latent[-1]
        
        for i, t in enumerate(self.scheduler.timesteps):           
            if t > t_idx:
                context = torch.cat([uncond_embeds_list[i], text_embeddings])
                latent = self.diffusion_step(latent, context, t, guidance_scale,  low_resource)
        
        latents_ori = latent.clone().detach()
        latents_edit = latent.clone().detach()
        # prepare to save original cross attention
        self.register_attention_control()

       
                
        
        
        t=1
        object_id_d = self.save_attention.object_id_d
        self.save_attention.self_attn = False
        self.save_attention.state = "ori"
        latents_input = torch.cat([ddim_latent[1]] * 2)
        context = torch.cat([uncond_embeds_list[-1], text_embeddings])
        self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        
        mask = self.get_cross_attn(state='ori')
        mask_max = mask.max()
        mask = normalize(mask)
        self.mask=mask
        #self.mask_0 =mask.gt(0.25).to(torch.float32)
        self.mask_0 =mask.gt(0.25).to(torch.float32)
        mask =  torch.nn.functional.interpolate(mask.unsqueeze(0),(512,512),mode='bilinear', align_corners=False)
        # torchvision.utils.save_image(torch.concat((img+mask,mask.repeat(1,3,1,1)),dim=-1), 'mask_1.png')
        # torchvision.utils.save_image(self.mask_0,'2.png')    
        self.save_attention.set_empty_store()    
        # area = self.mask_0.sum() / torch.numel(self.mask)
        # if area>0.25:
        #     t_idx = self.scheduler.timesteps[num_inference_steps- int(0.9*num_inference_steps)]
        #     repeat_n = 2
        # elif area>0.2:
        #     t_idx = self.scheduler.timesteps[num_inference_steps- int(0.9*num_inference_steps)]
        #     repeat_n = 1
        # elif area>0.1:
        #     t_idx = self.scheduler.timesteps[num_inference_steps- int(0.8*num_inference_steps)]
        #     repeat_n = 1

        torch.cuda.empty_cache()             
        
        for i, t in enumerate(self.scheduler.timesteps):
            if t > t_idx:
                continue
            
            context = torch.cat([uncond_embeds_list[i], text_embeddings])
            self.save_attention.state = "ori"
            self.save_attention.self_attn = False
            latents_ori = self.diffusion_step(latents_ori, context, t, guidance_scale,  low_resource, blend=False)
            #latents_ori = self.diffusion_step(ddim_latent[-i-1], context, t, guidance_scale,  low_resource)
            #self.save_attention.all_store(t)
            torch.cuda.empty_cache() 
            
            self.save_attention.state = "edit"
            if t > guidance_step:
                plus_guidance = True 
                self.save_attention.self_attn = True
                

                if t>500:
                    for iter in range(1):    
                        latents_edit = self.diffusion_step(latents_edit, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance, remain_step=True, latents_ori=latents_ori)
                        self.save_attention.set_edit_empty_store()

                     
                        
                latents_edit = self.diffusion_step(latents_edit, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance, latents_ori=latents_ori)
                self.save_attention.set_empty_store()
  
                
            else:
                plus_guidance = False
                self.save_attention.self_attn = True
                latents_edit = self.diffusion_step(latents_edit, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance,latents_ori=latents_ori, blend=True)
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
                    
                    # if i == 0 :
                    #     save_attention.phi[save_attention.state] = torch.tensor([],device=sample.device)#torch.nn.functional.interpolate(sample, size=(512, 512), mode='bilinear', align_corners=False)  * 0.03
                    # if i in [1,2,3]:
                    #     if i==1:
                    #         tmp = torch.nn.functional.interpolate(sample, size=(512, 512), mode='bilinear', align_corners=False)  * 2
                    #     if i == 2:
                    #         tmp = torch.nn.functional.interpolate(sample, size=(512, 512), mode='bilinear', align_corners=False) * 1
                    #     if i == 3:
                    #         tmp = torch.nn.functional.interpolate(sample, size=(512, 512), mode='bilinear', align_corners=False) * 1
                    #     save_attention.phi[save_attention.state] = torch.concat([save_attention.phi[save_attention.state],tmp],dim=1) 
                        
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
               # dtype=torch.float16
                device = self.device
            )
        
        #latent.requires_grad_(True)
        
        latents = latent.expand(batch_size,  self.unet.in_channels, height // 8, width // 8).to(self.device)
        return latent, latents

    def diffusion_step(self, latents, context, t, guidance_scale,  low_resource=False, plus_guidance=False, remain_step=False, latents_ori=None, blend=False):
        #with torch.no_grad():
        if plus_guidance:
            latents.requires_grad_(True)
            with torch.enable_grad():
                latents_input = torch.cat([latents] * 2)
                noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                
                #noise_pred = noise_prediction_text + guidance_scale * (noise_pred_uncond - noise_prediction_text)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
            
                loss, attn_edit_thresh = self.guidance(t, latents, latents_ori)
                
                # if not remain_step:
                #     loss += self.mse(latents * (1-self.mask),  latents_ori * (1-self.mask))*100
                
                
                attn_edit_thresh = torch.nn.functional.interpolate(attn_edit_thresh.unsqueeze(0),(64,64),mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

                gradients = torch.autograd.grad(loss, latents, retain_graph=False, create_graph=False)[0] 
                
                #torchvision.utils.save_image(torch.concat((normalize(attn_edit_thresh).unsqueeze(0),normalize(gradients.sum(1)), normalize(gradients.sum(1)) *  attn_edit_thresh.unsqueeze(0), self.mask),dim=2),f'attn_thresh/{t}.png')
                
                print(f'time_step:{t}, loss:{loss}, gradients_max:{gradients.max()}')
                

        
                
                #noise_pred_edit = noise_pred - gradients * attn_edit_thresh #* grad_scale#*  50
                noise_pred_edit = noise_pred - gradients * self.mask #* grad_scale#*  50

                #latents = self.rev_step(noise_pred, noise_pred_edit, t, latents)
                
                if remain_step:
                    latents = self.remain_step(noise_pred, noise_pred_edit, t, latents)
                else:
                    latents = self.rev_step(noise_pred, noise_pred_edit, t, latents)
                    # if t>500:
                    latents = latents * self.mask + latents_ori * (1-self.mask)
                    #loss += self.mse(latents * (1-self.mask),  latents_ori * (1-self.mask))*100
                    #latents = self.scheduler.step(noise_pred_edit, t, latents)["prev_sample"]
        else:
        
            latents_input = torch.cat([latents] * 2)
            noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            
            #noise_pred = noise_prediction_text + guidance_scale * (noise_pred_uncond - noise_prediction_text)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
            #latents = self.rev_step(noise_pred, noise_pred_edit, t, latents)
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
            
            if blend:
                latents = latents * self.mask + latents_ori * (1-self.mask)
                #loss += self.mse(latents * (1-self.mask),  latents_ori * (1-self.mask))*100
        #latents = self.next_step(noise_pred, t, latents, self.save_attention.noise_to_add.get(f'{t}'))
        
        # if save_noise:
        #     self.save_attention.save_noise_to_add(noise_pred, t)
        
        latents_next = latents.clone().detach()
  
        return latents_next
    
    def guidance(self, t, latents, latents_ori):
        object_id_d=self.save_attention.object_id_d
       
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
        

                
        cross_num=len(attn_edit)
        
        attn_edit=torch.cat(attn_edit,dim=0)
        attn_edit=attn_edit.sum(0)#/cross_num
        

        attn_edit_obj = 0
       
        for obj_id in object_id_d:
            attn_edit_obj = attn_edit_obj + attn_edit[obj_id].prod(0)  
            # #torch.concat((attn_edit_obj,attn_edit[obj_id]),dim=0)
            # a = normalize(attn_edit[obj_id][:1])
            # b = normalize(attn_edit[obj_id][1:2])
            # c = normalize(attn_edit[obj_id][2:])
            # d = normalize(attn_edit[obj_id].prod(0) )
            # e = a*b*c
            # torchvision.utils.save_image(torch.concat((a,b,c,d.unsqueeze(0),e),dim=-1),'1.png')


        attn_edit_obj = torch.nn.functional.interpolate(attn_edit_obj.unsqueeze(0).unsqueeze(0),(512,512),mode='bilinear', align_corners=False).squeeze(0)
        
        
        #grad_mask = normalize(attn_edit_obj)
        grad_mask = attn_edit_obj
        #attn_edit_thresh = normalize(self.sigmod(5*(normalize(attn_edit_obj)-0.2)) -0.3)
        
     
        
       
        
        if hasattr(self, 'init_max'):  
            #self.init_max = max(attn_edit_obj.max(), self.init_max)
            pass
        else:
            self.init_max = attn_edit_obj.max().item()
        
        # if hasattr(self, 'init_min'):  
        #     pass
        # else:
        #     self.init_min = attn_edit_obj.min()
        
        # attn_edit_obj = (attn_edit_obj - self.init_min) / (self.init_max - self.init_min) * 10
        #if self.init_max < 10:
        
        attn_edit_obj = attn_edit_obj   / self.init_max  * 15
        #attn_edit_obj = normalize(attn_edit_obj)  * 8
        
        # with open('file.txt', 'a') as file:  
        #     file.write(f"max: {attn_edit_obj.max()}. min: {attn_edit_obj.min()} \n")  
        
        #lmbda = max(min(t.to(self.device)/1000-0.5,0.7),0.)
        lmbda = max(min((t.to(self.device)-300)/600-0.2, 0.6),0.3)
        tmp = torch.quantile(attn_edit_obj.flatten(start_dim=1),  0.8, dim=1, keepdim=False).item()
        # attn_edit_obj = torch.where(
        #     attn_edit_obj >= tmp
        #     , attn_edit_obj
        #     , torch.zeros_like(attn_edit_obj)
        # )
        
        mask = torch.where(
            attn_edit_obj >= tmp
            , torch.ones_like(attn_edit_obj)
            , torch.zeros_like(attn_edit_obj)
        )
        attn_min = attn_edit_obj.min()
        
        loss = self.mse( attn_edit_obj * mask, mask * attn_min) * 1

        self.mask =  torch.nn.functional.interpolate(mask.unsqueeze(0),(64,64),mode='bilinear', align_corners=False).squeeze(0).to(torch.int) | self.mask_0.to(torch.int)
        #loss = loss * ((1+self.scheduler.alphas_cumprod[t]) ** 0.5)
        return loss, mask.squeeze(0) * grad_mask * 1 #* mask


    def rev_step(self, noise_pred, noise_pred_edit,  timestep, sample):
        timestep, next_timestep = timestep, max(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 0)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] 
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_timestep >=0 else self.scheduler.alpha_cumprod[0]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred_edit
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def remain_step(self, noise_pred, noise_pred_edit,  timestep, sample):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] 
        beta_prod_t = 1 - alpha_prod_t
        original_sample = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t) ** 0.5 * noise_pred_edit
        next_sample = alpha_prod_t ** 0.5 * original_sample + next_sample_direction
        return next_sample

   
    def pre_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):      
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def ddim_invert(self,img, text_embeds,latent,num_inference_steps,guidance_scale=7.5):
        
        all_latent = [latent]
        latent = latent.clone().detach()
        uncond_embeddings, cond_embeddings = text_embeds.chunk(2)
        
        with torch.no_grad():
            for i in range(num_inference_steps):
                t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
                noise_pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings ).sample
                #model_input = torch.concat([latent]*2, dim=0)
                #noise_pred = self.unet(model_input, t, encoder_hidden_states=cond_embeddings)["sample"]
                latent = self.pre_step(noise_pred, t, latent)
                all_latent.append(latent)
        return all_latent

    
    
    def null_optimization(self, ddim_latents, context, num_inner_steps, epsilon, num_inference_steps, guidance_scale=7.5):
        uncond_embeddings, cond_embeddings = context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = ddim_latents[-1]
        #bar = tqdm(total=num_inner_steps * num_inference_steps)
        for i in range(num_inference_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / (2*num_inference_steps)))
            latent_prev = ddim_latents[len(ddim_latents) - i - 2]
            t = self.scheduler.timesteps[i]
            
            noise_pred_cond = self.unet(latent_cur, t, encoder_hidden_states=cond_embeddings)["sample"]
            
            with torch.enable_grad():
                for j in range(num_inner_steps):
                    noise_pred_uncond = self.unet(latent_cur, t, encoder_hidden_states=uncond_embeddings)["sample"]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.scheduler.step(noise_pred, t, latent_cur)["prev_sample"]
                    loss = self.mse(latents_prev_rec, latent_prev)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    #bar.update()
                    if loss_item < epsilon + i * 2e-5:
                        break
            # for j in range(j + 1, num_inner_steps):
            #     bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
           
            context = torch.cat([uncond_embeddings, cond_embeddings])
                   
            latents_input =  torch.cat([latent_cur] * 2)
            noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
            
            latent_cur = self.scheduler.step(noise_pred, t, latent_cur)["prev_sample"]
                
        #bar.close()
        return uncond_embeddings_list

    @torch.no_grad()
    def null_text_inversion(
        self,
        prompt: List[str],
        image,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,

    ): 
        
        
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
        
        latent = self.encode_imgs(image).detach().requires_grad_(False)     
        #latent, init_latents = self.init_latent(latent, height, width, generator, batch_size)
        
        print('ddim inversion')
        ddim_latent = self.ddim_invert(image, context, latent, num_inference_steps, guidance_scale)
        print('uncond text optimiztion')
        uncond_embeds_list = self.null_optimization(ddim_latent, context, num_inner_steps=10, epsilon=1e-5, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        uncond_embeds_list.append(ddim_latent[-1])
        return uncond_embeds_list


    @torch.no_grad()
    def produce_latents_se(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 3, self.unet.in_channels, height // 8, width // 8), device=self.device).half()

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 3)

            # Save input tensors for UNet
            #torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
            #torch.save(t, "produce_latents_t.pt")
            #torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond , noise_pred_se= noise_pred.chunk(3)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) - 1 * (noise_pred_se - noise_pred_uncond)
            #noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device).half()

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # Save input tensors for UNet
            #torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
            #torch.save(t, "produce_latents_t.pt")
            #torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) 
            #noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents


    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        inputs = self.tokenizer(prompts, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        pos_embeds = self.text_encoder(inputs.input_ids.to(self.device))[0]

        
        inputs = self.tokenizer(negative_prompts, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        neg_embeds = self.text_encoder(inputs.input_ids.to(self.device))[0]
        
        inputs = self.tokenizer("smiling dog", padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        se_embeds = self.text_encoder(inputs.input_ids.to(self.device))[0]
        
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]
        text_embeds_se = torch.cat([neg_embeds, pos_embeds, se_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents_ori = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device).half()

        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents_ori, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        latents_se = self.produce_latents_se(text_embeds_se, height=height, width=width, latents=latents_ori, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=-guidance_scale) # [1, 4, 64, 64]
        
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        imgs_se = self.decode_latents(latents_se) # [1, 3, 512, 512]
        torchvision.utils.save_image(torch.concat((imgs,imgs_se),dim=0),'6.png')

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs



if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--idx', type=int, default=5, help='index of target word')
    parser.add_argument('--t_ratio', type=float, default=0.85)
    parser.add_argument('--num_inference_steps',type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default= 7.5)
    
    parser.add_argument('--img_path', type=str, default= "1.png")
    parser.add_argument('--prompts', type=str, default= "a photo of a woman .")

    
    args = parser.parse_args()    

    device = torch.device(args.device)
    sd = StableDiffusion(device, fp16=False, vram_O=False, sd_version='2.1',object_id_d=[[args.idx]], object_id_s = None)
  
    
    images, x_t = sd.real_image_ldm_stable(
        img_path = args.img_path,
        prompt=[args.prompts], 
        num_inference_steps=args.num_inference_steps, 
        guidance_scale=args.guidance_scale, 
        low_resource=False,  
        guidance_step=11, 
        t_ratio=args.t_ratio,
        )
    