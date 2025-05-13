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
from torch.optim.sgd import SGD


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    

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
        
        
        ########### lora fintinue
        self.unet.load_attn_procs("log/checkpoint-400/pytorch_model.bin")

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.mse = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        self.feature=[None]

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas)

        
        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


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

    
    #### save feature
    def register_attention_control(self):
        feature = self.feature
        
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

                
                ###########   save feature
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
          
          
    @torch.no_grad()
    def drag_diffusion(self, img, point, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
        ddim_idx=40
        t_idx=781
        
        def neighbor(x, y, d):
            points = []
            for i in range(x - d, x + d):
                for j in range(y - d, y + d):
                    points.append(torch.tensor([i, j]).float().to(latent.device))
            return points
        
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        
        
        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        text_embeds = pos_embeds
        self.scheduler.set_timesteps(num_inference_steps)
        
               
        
        ### save feature
        self.register_attention_control()
        
        
        ### ddim inversion
        latent = self.encode_imgs(img).detach().requires_grad_(False)
        ddim_latent = self.ddim_invert(img,text_embeds,latent, num_inference_steps, guidance_scale)
        
        ### ddim_latent[41] is x_781
        x_t = ddim_latent[ddim_idx].clone().detach()
        
        neg_embeds = self.get_text_embeds(negative_prompts)
        null_embeds = neg_embeds.clone().detach().requires_grad_(True)
        optimizer = Adam([null_embeds], lr=1e-2, eps=1e-4)
        
        noise_pred_cond = self.unet(x_t, t_idx, encoder_hidden_states=text_embeds).sample
        
        step = 0
        while step < 500:
            with torch.enable_grad():
                noise_pred_uncond = self.unet(x_t, t_idx, encoder_hidden_states=null_embeds).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                x_start_predict = self.predict_start(noise_pred, t_idx, x_t)
                loss = self.l1_loss(x_start_predict, ddim_latent[0])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f'rec optimiztion, step:{step}, loss:{loss}')
            step +=1
            if loss < 1e-2:
                break
            
        
        noise_pred_uncond = self.unet(x_t, t_idx, encoder_hidden_states=null_embeds).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        x_start_predict = self.predict_start(noise_pred, t_idx, x_t)
        dec_img = self.decode_latents(x_start_predict)
        torchvision.utils.save_image(dec_img, f'rec.jpg') 

        #SGD([x_t], lr=1e-2)
        latent_0_512 = F.interpolate(ddim_latent[0], size=(512, 512), mode='bilinear', align_corners=False)
        point= [[168,317],[161,360]]#[[180,207], [186,289]] #[[249,296],[180,299]] 244
        point=torch.tensor(point,dtype=torch.int)
        p_c = point[0]
        p_t = point[1]
        p_c0 = point[0]
        
        while(torch.norm((p_t-p_c).type(torch.float))>1):
            d = p_t - p_c
            with torch.enable_grad():
                noise_pred_uncond = self.unet(x_t, t_idx, encoder_hidden_states=null_embeds).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                x_start_predict = self.predict_start(noise_pred, t_idx, x_t)
                f = F.interpolate(x_start_predict, size=(512, 512), mode='bilinear', align_corners=False)
                loss = 0
                for qi in neighbor(int(p_c[0]), int(p_c[1]), 1):
                    f1 = bilinear_interpolate_torch(latent_0_512, qi[0], qi[1]).detach()
                    f2 = bilinear_interpolate_torch(f, qi[0] + d[0], qi[1] + d[1])
                    loss += self.l1_loss(f2, f1)
                    
                mask = torch.ones_like(f,device=f.device)
                mask[:,:,16:234,84:436]=0
                loss += self.l1_loss(f * mask,  latent_0_512  * mask) * 1  #.1
   
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            
            #point tracking
            noise_pred_uncond = self.unet(x_t, t_idx, encoder_hidden_states=null_embeds).sample
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            x_start_predict = self.predict_start(noise_pred, t_idx, x_t)
            f = F.interpolate(x_start_predict, size=(512, 512), mode='bilinear', align_corners=False)
            pi = p_c0
            # f = F0[..., int(pi[0]), int(pi[1])]
            f0 = bilinear_interpolate_torch(latent_0_512, pi[0].float(), pi[1].float())
            minv = 1e9
            minx = 1e9
            miny = 1e9
            for qi in neighbor(int(p_c[0]), int(p_c[1]), 3):
                f2 = bilinear_interpolate_torch(f, qi[0], qi[1])
                v = torch.norm(f2 - f0, p=1)
                if v < minv:
                    minv = v
                    minx = int(qi[0])
                    miny = int(qi[1])
            p_c[0] = minx
            p_c[1] = miny
            
            ######### info log
            print(f'step:{step}, loss:{loss},  target point:{p_t}, current point: {p_c}')
            
            if step % 100 ==0:
                embeds = null_embeds.clone().detach()
                noise_pred_uncond = self.unet(x_t, t_idx, encoder_hidden_states=embeds).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                x_start_predict = self.predict_start(noise_pred, t_idx, x_t)
                dec_img = self.decode_latents(x_start_predict)
                torchvision.utils.save_image(dec_img, f'results/{step}.jpg') 
            step+=1      
        
        
        embeds = null_embeds.clone().detach()
        noise_pred_uncond = self.unet(x_t, t_idx, encoder_hidden_states=embeds).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        x_start_predict = self.predict_start(noise_pred, t_idx, x_t)
        dec_img = self.decode_latents(x_start_predict)
        torchvision.utils.save_image(dec_img, f'results/{step}.jpg') 
        
  

        # step=0
        # self.unet(ddim_latent[ddim_idx], t_idx, encoder_hidden_states=text_embeds).sample
        # f_0 = self.feature[0]
        # f_0 = F.interpolate(f_0, size=(512, 512), mode='bilinear', align_corners=False)
        
                
        # point= [[168,317],[161,360]]#[[180,207], [186,289]] #[[249,296],[180,299]] 244
        # point=torch.tensor(point,dtype=torch.int)
        # p_c = point[0]
        # p_t = point[1]
        # p_c0 = point[0]
        
        
        
        # ### reconstrut loss on x_t_1
        # x_t_1_512 = F.interpolate(ddim_latent[40], size=(512, 512), mode='bilinear', align_corners=False)
        
        # while(torch.norm((p_t-p_c).type(torch.float))>1):
        #     d = (p_t - p_c) / torch.sum((p_t.float() - p_c.float())**2)
            
        #     with torch.enable_grad():
                
        #         #### motion  supervision
        #         noise_pred = self.unet(x_t, t_idx, encoder_hidden_states=text_embeds).sample
        #         x_t_pre = self.scheduler.step(noise_pred, t_idx, x_t)['prev_sample']
        #         f = self.feature[0]
        #         f =  F.interpolate(f, size=(512, 512), mode='bilinear', align_corners=False)

        #         loss = 0
        #         for qi in neighbor(int(p_c[0]), int(p_c[1]), 1):
        #             f1 = bilinear_interpolate_torch(f, qi[0], qi[1]).detach()
        #             f2 = bilinear_interpolate_torch(f, qi[0] + d[0], qi[1] + d[1])
        #             loss += self.l1_loss(f2, f1)
                
                
        #         x_t_pre_512 =  F.interpolate(x_t_pre, size=(512, 512), mode='bilinear', align_corners=False)
        #         mask = torch.ones_like(x_t_pre_512,device=x_t_pre_512.device)
        #         mask[:,:,16:234,84:436]=0
                
        #         loss += self.l1_loss(x_t_pre_512  * mask, x_t_1_512  * mask) * 40  #.1
        #         loss.backward()
        #         optimizer.step()
        #         optimizer.zero_grad()
            
            
        #     ### point tracking
        #     self.unet(x_t.clone().detach(), t_idx, encoder_hidden_states=text_embeds).sample
        #     f = self.feature[0]
        #     f =  F.interpolate(f, size=(512, 512), mode='bilinear', align_corners=False)
        #     pi = p_c0
        #     # f = F0[..., int(pi[0]), int(pi[1])]
        #     f0 = bilinear_interpolate_torch(f_0, pi[0].float(), pi[1].float())
        #     minv = 1e9
        #     minx = 1e9
        #     miny = 1e9
        #     for qi in neighbor(int(p_c[0]), int(p_c[1]), 3):
        #         f2 = bilinear_interpolate_torch(f, qi[0], qi[1])
        #         v = torch.norm(f2 - f0, p=1)
        #         if v < minv:
        #             minv = v
        #             minx = int(qi[0])
        #             miny = int(qi[1])
        #     p_c[0] = minx
        #     p_c[1] = miny
                
                

            
        #     ######### info log
        #     print(f'step:{step}, loss:{loss},  target point:{p_t}, current point: {p_c}')
            
        #     if step % 100 ==0:
        #         latents = x_t.clone().detach()
        #         for i, t in enumerate(self.scheduler.timesteps):
        #             if t > t_idx:
        #                 continue
        #             # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        #             latent_model_input = latents
                    
        #             # predict the noise residual
        #             noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

        #             # compute the previous noisy sample x_t -> x_t-1
        #             latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        #         dec_img = self.decode_latents(latents)
        #         torchvision.utils.save_image(dec_img, f'results/{step}.jpg') 
        #     step+=1      
        
        
        # latents = x_t.clone().detach()
        # for i, t in enumerate(self.scheduler.timesteps):
        #     if t > t_idx:
        #         continue
        #     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        #     latent_model_input = latents
            
        #     # predict the noise residual
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']

        #     # compute the previous noisy sample x_t -> x_t-1
        #     latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        # dec_img = self.decode_latents(latents)
        # torchvision.utils.save_image(dec_img, f'results/{step}.jpg') 
        
        # return x_t     
                
    def predict_start(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray] ):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        return original_sample
    
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

    import argparse
    import cv2
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="a sitting dog")   ### fintune lora with prompt "a sitting dog"
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

    
    seed_everything(opt.seed)


    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)
    
 
    rgb_path = '3.png'
    
    img = cv2.imread(rgb_path)
    img = cv2.resize(img,(512,512))
    point= [[168,317],[161,244]]#[[180,207], [186,289]] #[[249,296],[180,299]]

        
    img = Image.open(rgb_path).convert('RGB')
    img = torch.tensor(np.array(img),device=device).permute(2,0,1).unsqueeze(0)/255
    img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
    

    sd.drag_diffusion(img, point,opt.prompt, opt.negative)
   
    


    






