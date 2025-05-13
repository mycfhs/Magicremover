import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import DDPMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput, logger
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import torchvision
from PIL import Image
import numpy as np
# suppress partial model loading warning
logging.set_verbosity_error()
from typing import Optional, Union, Tuple, List, Callable, Dict, Any, Mapping
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd

from torch import Tensor

import torchvision.transforms.functional as transforms_F
from PIL import Image, ImageDraw
import torch.optim as optim  

from torch.autograd import grad
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
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
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
        #self.unet.load_attn_procs("results/dog/checkpoint-400/pytorch_model.bin")

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.ddpm_scheduler = DDPMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.mse = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.sigmod = nn.Sigmoid()
        self.feature=[None]

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas)

        self.save_attention = Save_Attention(object_id_d=[5], object_id_s=[6] )  
        print(f'[INFO] loaded stable diffusion!')



    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.ddpm_scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # if t>100:
            #     for num in range(10):
            #         if num==0:
            #             self.save_attention.swap_attn=True
            #         else:
            #             self.save_attention.swap_attn=False
            #         # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            #         latent_model_input = torch.cat([latents] * 2)
                    
            #         self.save_attention.cur_t = t
            #         # predict the noise residual
            #         noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            #         self.save_attention.set_empty_store()
                    
            #         # perform guidance
            #         noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
            #         # add disturbance to x_t
            #         latents = self.disturb(latents, t, noise_pred)
            
            if t > 600:
                self.save_attention.swap_attn=True
            else:
                self.save_attention.swap_attn=False
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            
            self.save_attention.cur_t = t
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            self.save_attention.all_store(t)
            
            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.ddpm_scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

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
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
        
    @torch.no_grad()
    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        if latents is None:
            latents_ori = torch.randn((text_embeds.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.ddpm_scheduler.set_timesteps(num_inference_steps)

        latents = latents_ori.clone()
        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            
            self.save_attention.cur_t = t
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
      
        # Img latents -> imgs
        imgs_1 = self.decode_latents(latents) # [1, 3, 512, 512]

        # latents = latents_ori.clone()
        # for i, t in enumerate(self.scheduler.timesteps):
        #     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        #     if t ==881:
        #         latents = latents.flip(2)
        #     latent_model_input = torch.cat([latents] * 2)
            
        #     self.save_attention.cur_t = t
        #     # predict the noise residual
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

        #     # perform guidance
        #     noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            
        #     # compute the previous noisy sample x_t -> x_t-1
        #     latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        # imgs_2 = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        torchvision.utils.save_image(imgs_1.cpu(),'img.png')
        # imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        # imgs = (imgs * 255).round().astype('uint8')

        return None
    



    def register_attention_control(self):
        save_attention = self.save_attention
        feature = self.feature
        dog_attn_path = "masks/dog.png"
        ball_attn_path = "masks/ball.png"
        dog_attn = Image.open(dog_attn_path).convert('RGB')
        dog_attn= torch.tensor(np.array(dog_attn),device=self.device).permute(2,0,1)[0]/255
        #dog_attn = dog_attn.flip(1)
        
        ball_attn = Image.open(ball_attn_path).convert('RGB')
        ball_attn = torch.tensor(np.array(ball_attn),device=self.device).permute(2,0,1)[0]/255
        
        fin_attn = dog_attn + ball_attn
        
        # ball_attn[300:450,:] = ball_attn[150:300,:]
        # ball_attn[150:300,:] = 0
        ball_attn = ball_attn.flip(1)
        
        bag_attn = Image.open("masks/bag_scribble.png").convert('RGB')
        bag_attn= 1 - torch.tensor(np.array(bag_attn),device=self.device).permute(2,0,1)[0]/255
   
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

                feature[0] = sample
                
                
                # 6. post-process
                if self.conv_norm_out:
                    sample = self.conv_norm_out(sample)
                    sample = self.conv_act(sample)
                    #feature[0] = sample
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
                        res = int(attn.shape[1] ** 0.5 )
                        dog_attn_t = torch.nn.functional.interpolate(dog_attn.unsqueeze(0).unsqueeze(0),(res,res),mode='bilinear', align_corners=False) 
                        ball_attn_t = torch.nn.functional.interpolate(ball_attn.unsqueeze(0).unsqueeze(0),(res,res),mode='bilinear', align_corners=False) 
                        bag_attn_t = torch.nn.functional.interpolate(bag_attn.unsqueeze(0).unsqueeze(0),(res,res),mode='bilinear', align_corners=False) 
                        attn = attn.reshape(attn.shape[0],res,res,attn.shape[-1])
                        
                        # fin_attn_t = torch.nn.functional.interpolate(fin_attn.unsqueeze(0).unsqueeze(0),(res,res),mode='bilinear', align_corners=False)
                        # attn[attn.shape[0]//2:, :, :, 1:10] = fin_attn_t.repeat(attn.shape[0]//2,9,1,1).permute(0,2,3,1) * 1
                        #grass_attn = attn[attn.shape[0]//2:, :, :, 11]
                        #attn[attn.shape[0]//2:] = attn[:attn.shape[0]//2] 
                        #attn[attn.shape[0]//2:] = attn[attn.shape[0]//2:]*0.5
                        
                        # attn[attn.shape[0]//2:, :, :, 5]=dog_attn_t.squeeze(0).repeat(attn.shape[0]//2,1,1) * 1
                        # attn[attn.shape[0]//2:, :, :, 8]=ball_attn_t.squeeze(0).repeat(attn.shape[0]//2,1,1) * 0.5
                        attn[attn.shape[0]//2:, :, :, 1]=bag_attn_t.squeeze(0).repeat(attn.shape[0]//2,1,1) * 1
                        
                        #attn[attn.shape[0]//2:, :, :, 11] = grass_attn * 100
                        

                        
                        mask = torch.nn.functional.normalize(attn[attn.shape[0]//2:], p=2.0, dim=-1, eps=1e-12)          
                        #attn[attn.shape[0]//2:] = attn[attn.shape[0]//2:] * mask
                        attn = attn.reshape(attn.shape[0], res*res, attn.shape[-1])

                        

                        
                        

                
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
          
   
   
    @torch.no_grad()
    def set_cross_attn(self, img, point, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
        


        
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        self.scheduler.set_timesteps(num_inference_steps)


        # Prompts -> text embeds
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]
        #latent = self.encode_imgs(img).detach().requires_grad_(False)
        
        latents = torch.randn((text_embeds.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        # Text embeds -> img latents
        self.register_attention_control()
        self.save_attention.swap_attn=True
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        torchvision.utils.save_image(torch.tensor(imgs).cpu(),'img.png')
        # imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        # imgs = (imgs * 255).round().astype('uint8')

        return imgs

    
    def visualize(self,t):
        for i in range(981,0,-20):
            phi = self.save_attention.phi_all_step[f"{i}"][1:]
            phi = phi.sum(1)
            torchvision.utils.save_image(phi, f'visualize/phi/{i}.png')

    
        obj_id=8
        for t in range(981,0,-20):
            item_all=torch.tensor([],device='cuda')
            for location in ["down","mid","up"]:
                for item in  self.save_attention.attention_all_step[f'{t}'][f"{location}_cross"]:
                    if item.shape[1]==(64**2):
                        item=item[item.shape[0]//2:]
                        res= int(item.shape[1] ** 0.5)
                        item = item.reshape(item.shape[0],res,res,item.shape[-1])
                        item = item.permute(0, 3, 1, 2)  
                        item = item.sum(0)
                        item_all = torch.concat((item_all,item[obj_id].unsqueeze(0)),dim=0)
            torchvision.utils.save_image(item_all.sum(0),f'visualize/attn_64/{t}.png')
            
        pass            

    def guidance(self, attn_ori, phi_ori, t):
        object_id_d=self.save_attention.object_id_d
        object_id_s=self.save_attention.object_id_s


        
        
        attn_edit=[]
        for location in ["down","mid","up"]:
            for item in self.save_attention.step_store[f"{location}_cross"]:
                item=item[item.shape[0]//2:]
                
                res= int(item.shape[1] ** 0.5)
                item = item.reshape(item.shape[0],res,res,item.shape[-1])
                item = item.permute(0, 3, 1, 2)  
                item = torch.nn.functional.interpolate(item,(64,64),mode='bilinear', align_corners=False)
                attn_edit.append(item)
        

        attn_edit=torch.cat(attn_edit,dim=0)
        cross_num=len(attn_edit)
        cross_final_edit = attn_edit[-1][object_id_d]
        attn_edit=attn_edit.sum(0)/cross_num
        
        attn_edit_obj = attn_edit[object_id_d]
        
        cur_cen = tensor_centroid(attn_edit_obj)#_thresh)
        #ori_cen = tensor_centroid(attn_ori_obj)#thresh)
        target_cen= torch.tensor([40,40],device=attn_edit.device)
        loss = (self.l1_loss(cur_cen[0], target_cen[0]) +  self.l1_loss(cur_cen[1],target_cen[1]) ) * 0.05
        
        
        
        attn_ori=torch.cat(attn_ori,dim=0)
        cross_final_ori  =  attn_ori[-1][object_id_d]
        attn_ori=attn_ori.sum(0)/cross_num
        a_ori = (phi_ori * cross_final_ori.unsqueeze(0).repeat(1,320,1,1)).sum(-1).sum(-1) / torch.sum(attn_ori)
        
        phi_edit = self.feature[0]
        a_edit = (phi_edit * cross_final_edit.unsqueeze(0).repeat(1,320,1,1)).sum(-1).sum(-1) / torch.sum(attn_edit)
        loss += self.l1_loss(a_edit, a_ori) * 1
        
        return loss
    
       
        
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def ddim_invert(self,img,text_embeds,latent,num_inference_steps,guidance_scale=7.5):
        
        all_latent = [latent]
        latent = latent.clone().detach()
        with torch.no_grad():
            for i in range(num_inference_steps):
                t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
                noise_pred = self.unet(latent, t, encoder_hidden_states=text_embeds).sample
                #noise_pred = self.unet(latent, t, encoder_hidden_states=text_embeds)["sample"]
                #noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
                #noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
                latent = self.next_step(noise_pred, t, latent)
                all_latent.append(latent)
        return all_latent
        
    def pre_step(self, latent_noisy, t, noise_pred):
        prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latent_noisy - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def disturb(self, latent_noisy, t, noise_pred):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latent_noisy - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t) ** 0.5 * noise_pred
        sample = alpha_prod_t ** 0.5 * pred_original_sample + pred_sample_direction
        return sample

        
    def invert(self, img, prompts, negative_prompts='', guidance_scale=7.5, num_inference_steps=50, num_inner_steps=10):
        
        latent = self.encode_imgs(img).detach().requires_grad_(False)
        
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        # neg_embeds = self.get_text_embeds(negative_prompts)
        # text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]
        text_embeds = pos_embeds
        
        self.scheduler.set_timesteps(num_inference_steps)
        embedding_list = []
        
        ddim_latent = self.ddim_invert(img,text_embeds,latent, num_inference_steps, guidance_scale)
        
        #latent_noisy = torch.randn_like(latent,device=latent.device)
    
        for i in range(num_inference_steps):
            text_embeddings = text_embeds.clone().detach().requires_grad_(True)
            optimizer = Adam([text_embeddings], lr=1e-1) # * (1. - i / 100.))
            t = self.scheduler.timesteps[i]
            

            for j in range(int(num_inner_steps*10*(1-i/num_inference_steps))):
                if i == 0:
                    with torch.no_grad():
                        noise = torch.rand_like(latent,device=latent.device)
                        latent_noisy = self.scheduler.add_noise(latent, noise, self.scheduler.timesteps[0])
            #for j in range(num_inner_steps):
                
                latent_model_input = latent_noisy #torch.cat([latent_noisy] * 2)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # perform guidance (high scale from paper!)
                # noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
                latent_prev = self.pre_step(latent_noisy, t, noise_pred)
                loss = self.mse(ddim_latent[-i-2], latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
               
                print(f'step: {i}, inner step: {j}, loss: {loss}')
            
            embedding_list.append(text_embeddings)
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                latent_noisy = self.pre_step(latent_noisy, t, noise_pred)
        
        #self.scheduler.set_timesteps(num_inference_steps)
        #embedding_list = torch.load("1.pt")
        #latents = torch.randn((1,4,64,64), device=self.device)
        with torch.no_grad():
            latents = self.scheduler.add_noise(latent, noise, self.scheduler.timesteps[0])
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = latents
                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=embedding_list[i])['sample']

                # perform guidance
                # noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        img = self.decode_latents(latents)
        torchvision.utils.save_image(img, 'inver.jpg')
        
            
        return embedding_list

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
    



if __name__ == '__main__':

    init_seed = 9998
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    torch.cuda.manual_seed(init_seed)


    import argparse
    import cv2
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="a photo of a dog playing a ball on the grass")
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()


    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)
    
    prompts = ["a photo of a dog playing a ball on the grass"]
    #prompts = ["bag"]
    sd.prompt_to_img(prompts, num_inference_steps=50)


    rgb_path = '2.png'
    
    img = cv2.imread(rgb_path)
    img = cv2.resize(img,(512,512))
    point=[[167,298],[170,315]]

        
    img = Image.open(rgb_path).convert('RGB')
    img = torch.tensor(np.array(img),device=device).permute(2,0,1).unsqueeze(0)/255
    img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
    
    #ddim_latents = sd.ddim_invert(img)
    #sd.prompt_to_img(opt.prompt,num_inference_steps=100)
    sd.set_cross_attn(img, point, prompts, opt.negative, num_inference_steps=100)
    

    


    






