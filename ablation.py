import os  
import numpy as np
import skimage.io as io  
from pycocotools.coco import COCO  
from IPython.display import Image  
import torch 
import torchvision
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F  
import cv2
import random
#from inpating_word_sum_time import StableDiffusion
from inpating_muti_word import StableDiffusion,normalize


random.seed(1)  


# 设置图片和注释文件夹路径  
data_dir = '/home/ubuntun/yyc_workspace/remover/coco2017'  
image_type = 'val2017'  
annotation_file = 'instances_{}.json'.format(image_type)  
  
# 初始化 COCO 对象  
coco = COCO(os.path.join(data_dir, 'annotations', annotation_file))  
  
  
categories = coco.loadCats(coco.getCatIds())  
category_names = [category['name'] for category in categories]   
random.shuffle(category_names)  

device='cuda:0'

def check_file_exists(directory,filename):
    file_path = os.path.join(directory,filename)
    if os.path.exists(file_path):
        return True
    else:
        return False

# with torch.no_grad():
#     sd = StableDiffusion(device, fp16=False, vram_O=False, sd_version='2.1',object_id_d=[[9],], object_id_s=[8])
   

#     i = 0
#     for category in category_names:
#         cat_id = coco.getCatIds(catNms=category )[0]  
#         image_ids = coco.getImgIds(catIds=[cat_id])  
#         prompt = ["a photo of a " + category]
#         for image_id in image_ids:
#             if 0<=i<1250:
#                 try:
#                     image_data = coco.loadImgs(image_id)[0]
                    
#                     image = io.imread(os.path.join(data_dir, image_type, image_data['file_name']))  
#                     file_name = image_data['file_name'].split('.')[0]


                    
#                     image = torch.tensor(image/255,dtype=torch.float32).permute(2,0,1).to(device)
                    
#                     image = F.interpolate(image.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
                    
#                     null_text_embdding = sd.null_text_inversion(prompt, image, latent=None, num_inference_steps=50, guidance_scale=7.5)
                    
                    
#                     torch.save(null_text_embdding,f'null_text_embeddings_coco/embeddings_2/{file_name}_{cat_id}.pt')
                                        
#                 except:
#                     pass
#             i+=1
#             print(i)





######### 加了 mask  843行noise_pred_edit = noise_pred - gradients * attn_edit_thresh * self.mask 

num_inference_steps=50
t_ratio=0.85
guidance_scale=7.5
guidance_step=11
low_resource = False

category_names = {category['id']: category['name'] for category in categories}  

with torch.no_grad():
    sd = StableDiffusion(device, fp16=False, vram_O=False, sd_version='2.1',object_id_d=[[5],], object_id_s=[8])
    #embeds = os.listdir("/home/siyuan/embeddings_2")
    embeds = os.listdir("null_text_embeddings_coco/embeddings")
    
    test_i = 0
    for embed_name in embeds:
        if 1000<=test_i:
            cat_id = int(embed_name.split('.')[0].split('_')[-1])
            category = category_names[cat_id]

            image_id = int(embed_name.split('_')[0].lstrip('0'))

            
            prompt = ["a photo of a " + category]
            
            sd.save_attention.object_id_d = [[5+i for i in range(len(category.split(' ')))],]
            

       
            #try:
            image_data = coco.loadImgs(image_id)[0]
            
            image = io.imread(os.path.join(data_dir, image_type, image_data['file_name']))  
            file_name = image_data['file_name'].split('.')[0]


            
            image = torch.tensor(image/255,dtype=torch.float32).permute(2,0,1).to(device)
            
            # 获取图像的注释  
            annotation_ids = coco.getAnnIds(imgIds=image_data['id'], catIds=[cat_id], iscrowd=None)  
            annotations = coco.loadAnns(annotation_ids)  
            
            # 创建一个空的 mask 图像  
            mask = np.zeros((image_data['height'], image_data['width']))  
            
            # 根据注释为每个实例添加 mask  
            for annotation in annotations:  
                mask += coco.annToMask(annotation)  
            
            kernel = np.ones((30, 30), np.uint8) 
            mask = cv2.dilate(mask, kernel, iterations=1)  
            mask = torch.tensor(mask).to(device)
            mask = (mask>0.5).to(torch.float32)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
            
            
            
            
            image = F.interpolate(image.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)



            batch_size = len(prompt)
            text_input = sd.tokenizer(
                prompt,
                padding="max_length",
                max_length=sd.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = sd.text_encoder(text_input.input_ids.to(sd.device))[0]
            max_length = text_input.input_ids.shape[-1]
            uncond_input = sd.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = sd.text_encoder(uncond_input.input_ids.to(sd.device))[0]
            
            context = [uncond_embeddings, text_embeddings]
            context = torch.cat(context)
            # set timesteps
            #extra_set_kwargs = {"offset": 1}
            sd.scheduler.set_timesteps(num_inference_steps)#, **extra_set_kwargs)
            sd.ddpm_scheduler.set_timesteps(num_inference_steps)#, **extra_set_kwargs)
            
            t_idx = sd.scheduler.timesteps[num_inference_steps- int(t_ratio*num_inference_steps)]
            ddim_idx = int(t_ratio*num_inference_steps)

            img = image
            latent_0 = sd.encode_imgs(img).detach().requires_grad_(False)     
            
            print('ddim inversion')
            ddim_latent = sd.ddim_invert(img, context, latent_0, num_inference_steps, guidance_scale)
    

            uncond_embeds_id = file_name+'_'+str(cat_id)
            #uncond_embeds_list = torch.load(f'/home/siyuan/embeddings/{uncond_embeds_id}.pt')
            uncond_embeds_list = torch.load(f'null_text_embeddings_coco/embeddings/{uncond_embeds_id}.pt',map_location=device)
            

            uncond_embeds_list = [tensor.to(device) for tensor in uncond_embeds_list] 
            print('done')
            
            latent = uncond_embeds_list[-1]
            
            for i, t in enumerate(sd.scheduler.timesteps):           
                if t > t_idx:
                    context = torch.cat([uncond_embeds_list[i], text_embeddings])
                    latent = sd.diffusion_step(latent, context, t, guidance_scale,  False)
            
            latents_ori = latent.clone().detach()
            latents_edit = latent.clone().detach()
            # prepare to save original cross attention
            sd.register_attention_control()

            torch.cuda.empty_cache() 
            
            
            mask=None
            if mask is not None:
                #sd.mask = mask
                sd.mask = torch.nn.functional.interpolate(mask,(64,64),mode='bilinear', align_corners=False).squeeze(0).to(torch.int)
                sd.mask_source = 'ground'
            else:
                sd.mask_source = None
                t=1
                object_id_d = sd.save_attention.object_id_d
                sd.save_attention.self_attn = False
                sd.save_attention.state = "ori"
                latents_input = torch.cat([ddim_latent[1]] * 2)
                context = torch.cat([uncond_embeds_list[-2], text_embeddings])
                sd.unet(latents_input, t, encoder_hidden_states=context)["sample"]
                
                mask = sd.get_cross_attn(state='ori')
                mask = normalize(mask)
                sd.mask=mask
                sd.mask_0 =mask.gt(0.3).to(torch.float32)
                mask =  torch.nn.functional.interpolate(mask.unsqueeze(0),(512,512),mode='bilinear', align_corners=False)
                torchvision.utils.save_image(torch.concat((img+mask,mask.repeat(1,3,1,1)),dim=-1), 'mask_1.png')
                torchvision.utils.save_image(sd.mask_0,'2.png')    
                sd.save_attention.set_empty_store()    
                
                torch.cuda.empty_cache()       
                    

            
            for i, t in enumerate(sd.scheduler.timesteps):
                if t > t_idx:
                    continue
                
                context = torch.cat([uncond_embeds_list[i], text_embeddings])
                sd.save_attention.state = "ori"
                sd.save_attention.self_attn = False
                latents_ori = sd.diffusion_step(latents_ori, context, t, guidance_scale,  low_resource)
                #latents_ori = sd.diffusion_step(ddim_latent[-i-1], context, t, guidance_scale,  low_resource)
                #sd.save_attention.all_store(t)
                torch.cuda.empty_cache()
                
                sd.save_attention.state = "edit"
                if t > guidance_step:
                    plus_guidance = True 
                    sd.save_attention.self_attn = True
                    
                    
                    if t>500:
                        for iter in range(1):    
                            latents_edit = sd.diffusion_step(latents_edit, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance, remain_step=True, latents_ori=latents_ori)
                            sd.save_attention.set_edit_empty_store()
                            torch.cuda.empty_cache() 
                    # elif t > 580:  
                    
                        # for iter in range(2):    
                        #     latents_edit = sd.diffusion_step(latents_edit, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance, remain_step=True, latents_ori=latents_ori)
                        #     sd.save_attention.set_edit_empty_store()
                    
                    latents_edit = sd.diffusion_step(latents_edit, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance, latents_ori=latents_ori)
                    sd.save_attention.set_empty_store()
                    torch.cuda.empty_cache()      
                else:
                    plus_guidance = False
                    sd.save_attention.self_attn = True
                    latents_edit = sd.diffusion_step(latents_edit, context, t, guidance_scale,  low_resource, plus_guidance=plus_guidance)
                    sd.save_attention.set_empty_store() 
                    torch.cuda.empty_cache()      
                
            img_ori = sd.decode_latents(latents_ori)
            img_edit = sd.decode_latents(latents_edit)
            torchvision.utils.save_image(torch.concat((img_ori,img_edit),dim=0),f'ablation_data/full/{file_name}_{cat_id}.png')
            #torchvision.utils.save_image(img_edit,'1.png')
            sd.save_attention.reset()
            
            sd.reset_unet()
            torch.cuda.empty_cache()      
            
            
            #return img_edit, latent

        # except:
        #     pass
        test_i+=1
        print(test_i)


