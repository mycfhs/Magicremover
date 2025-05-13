import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
from torch.autograd import grad
import torch.nn as nn
import torchvision
from diffusers import DDPMScheduler, DDIMScheduler, StableDiffusionPipeline
import random

class Save_Attention(nn.Module):
    def __init__(self):
        super(Save_Attention, self).__init__() 
        self.set_empty_store()
        self.swap_attn=False

    def save_attn(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
        #if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)

    def set_empty_store(self):
        self.step_store =  {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
        
        
    def del_store(self):
        del self.step_store
        self.set_empty_store()
    

save_attention = Save_Attention()
l1_loss = torch.nn.L1Loss()
sigmod = nn.Sigmoid()
def normalize(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min())

GUIDANCE_STEP=801


def guidance(batch_size):
    object_id=2

    rec_loss=0
    attn_ori=[]
    attn_edit=[]
    for location in ["down","mid","up"]:
        for item in save_attention.step_store[f"{location}_cross"]:
            item=item[item.shape[0]//2:]
            
            res= int(item.shape[1] ** 0.5)
            item = item.reshape(item.shape[0],res,res,item.shape[-1])
            item = item.permute(0, 3, 1, 2)  
            item = torch.nn.functional.interpolate(item,(64,64),mode='bilinear', align_corners=False)
            
            item_ori = item[:item.shape[0]//2]
            item_edit = item[item.shape[0]//2:]
            
            rec_loss += l1_loss(item_ori[:,:object_id,:,:], item_edit[:,:object_id,:,:]) + l1_loss(item_ori[:,object_id+1:, :, :],item_edit[:,object_id+1:,:,:])

            
            attn_ori.append(torch.nn.functional.interpolate(item_ori,(64,64),mode='bilinear', align_corners=False))
            attn_edit.append(torch.nn.functional.interpolate(item_edit,(64,64),mode='bilinear', align_corners=False))
            
    cross_num=len(attn_ori)
    attn_ori=torch.cat(attn_ori,dim=0)
    attn_ori=attn_ori.sum(0)/cross_num
    
    attn_edit=torch.cat(attn_edit,dim=0)
    attn_edit=attn_edit.sum(0)/cross_num
    
    ### shape
    attn_ori_obj = attn_ori[object_id]
    attn_ori_thresh = normalize(sigmod(10*(normalize(attn_ori_obj)-0.5)))
    
    attn_edit_obj = attn_edit[object_id]
    attn_edit_thresh = normalize(sigmod(10*(normalize(attn_edit_obj)-0.5)))
    
    #### object size
    ori_size = torch.sum(attn_ori_thresh) / torch.numel(attn_ori_thresh)
    edit_size = torch.sum(attn_edit_thresh) / torch.numel(attn_edit_thresh)
    
    
    ### object position
    cur_cen = tensor_centroid(attn_edit_thresh)
    target_cen=10,10
    loss =  torch.abs(cur_cen[0] - target_cen[0]) + torch.abs(cur_cen[1] - target_cen[1]) + torch.abs(edit_size-ori_size) * 10 #+ rec_loss #* 50 #+ torch.abs(edit_size-ori_size) #* 5
    
    #loss = l1_loss(cur_cen,target_cen)
    
    
    
    return loss



def tensor_centroid(tensor):  
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




def diffusion_step(model, latents, context, t, guidance_scale,  low_resource=False, plus_guidance=False):
    #with torch.no_grad():
    if plus_guidance:
        latents.requires_grad_(True)
        with torch.enable_grad():
            if low_resource:
                noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
                noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
            else:
                latents_input = torch.cat([latents] * 2)
                noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            
            #noise_pred = noise_prediction_text + guidance_scale * (noise_pred_uncond - noise_prediction_text)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        
            loss = guidance(batch_size=latents.shape[0])
            # loss.backward()
            # gradients = latents.grad
            gradients = torch.autograd.grad(loss, latents, retain_graph=False, create_graph=False)[0]  
            
            noise_pred[1:] = noise_pred[1:] + gradients[1:] * 10
    
    else:
        
        if low_resource:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
        else:
            latents_input = torch.cat([latents] * 2)
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        
        #noise_pred = noise_prediction_text + guidance_scale * (noise_pred_uncond - noise_prediction_text)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)



    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents_next = latents.clone().detach()
    
    save_attention.set_empty_store()
   
    # save_attention.del_store()
    # latents = latents.detach()
    # del noise_pred,  gradients
    # model.unet.zero_grad()
    # torch.cuda.empty_cache()

    


    return latents_next





            


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    # image = image.cpu().permute(0, 2, 3, 1).numpy()
    # image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
            dtype=torch.float16
        )
    
    #latent.requires_grad_(True)
    
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
   

    # set timesteps
    #extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)#, **extra_set_kwargs)
  
    latent, init_latents = init_latent(latent, model, height, width, generator, batch_size)
    latents=init_latents
    
    plus_guidance=True
    for t in tqdm(model.scheduler.timesteps):
        if t == GUIDANCE_STEP:
            plus_guidance=False        
            save_attention.swap_attn=True    
        latents = diffusion_step(model, latents, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance)

    image_w_guid = latent2image(model.vae, latents)
    torchvision.utils.save_image(image_w_guid,'1.png')

    # latents=init_latents
    # for t in tqdm(model.scheduler.timesteps):        
    #     latents = diffusion_step(model, latents, context, t, guidance_scale,  low_resource, plus_guidance=False)
    # image_o_guid = latent2image(model.vae, latents)
    # torchvision.utils.save_image(torch.concat((image_w_guid,image_o_guid),dim=0),'2.png')
  
    return image_w_guid, latent





def register_attention_control(model):
    
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
            save_attention.save_attn(attn, is_cross, place_in_unet)
            
            #################################################################               
            
            
            # if is_cross :  
            #     if save_attention.swap_attn==True:
            #         attn_conditional=attn[attn.shape[0]//2:]
            #         h = attn_conditional.shape[0] // (batch_size//2)  
            #         attn_conditional = attn_conditional.reshape(batch_size//2, h, *attn_conditional.shape[1:])  
            #         attn_base, attn_replace = attn_conditional[:1], attn_conditional[1:]  
            #         #alpha_words = controller.cross_replace_alpha[controller.cur_step]  
            #         #attn_replace = attn_base  # controller.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace  
            #         masks = torch.ones_like(attn_base, device = attn_base.device)
            #         masks[:,:,:,2] = 0
            #         attn_replace = attn_base * masks + attn_replace * (1-masks)
            #         attn_conditional = torch.cat([attn_base, attn_replace], dim=0)  
            #         attn_conditional = attn_conditional.reshape(batch_size//2 * h, *attn_conditional.shape[2:]) 
            
            #         attn[attn.shape[0]//2:]=attn_conditional

            
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
       

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            register_recr(net[1], 0, "mid")

 



if __name__=='__main__':
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    model_key = "stabilityai/stable-diffusion-2-1-base"
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to(device)
    ldm_stable.scheduler = DDPMScheduler.from_pretrained(model_key,  subfolder="scheduler", torch_dtype=torch.float16)
    tokenizer = ldm_stable.tokenizer

    # init_seed = 8886
    # torch.manual_seed(init_seed)
    # random.seed(init_seed)
    # torch.cuda.manual_seed(init_seed)
    
    prompts = [ "A dog playing a ball on the grass",
               "A dog playing a ball on the grass"]
    
    #x_t = torch.randn((1,4,64,64),device=device)
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=None, low_resource=LOW_RESOURCE)