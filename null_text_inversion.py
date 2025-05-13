import os  
import numpy as np
import skimage.io as io  
from pycocotools.coco import COCO  
from IPython.display import Image  
import torch 
import torchvision
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F  

from main import StableDiffusion

# 设置图片和注释文件夹路径  
data_dir = '/home/dataset/coco'  
image_type = 'val2017'  
annotation_file = 'instances_{}.json'.format(image_type)  
  
# 初始化 COCO 对象  
coco = COCO(os.path.join(data_dir, 'annotations', annotation_file))  
  
  
categories = coco.loadCats(coco.getCatIds())  
category_names = [category['name'] for category in categories]   

device='cuda'

with torch.no_grad():
    sd = StableDiffusion(device, fp16=False, vram_O=False, sd_version='2.1',object_id_d=[[9],], object_id_s=[8])
   

    i = 0
    for category in category_names:
        cat_id = coco.getCatIds(catNms=category )[0]  
        image_ids = coco.getImgIds(catIds=[cat_id])  
        prompt = ["a photo of a " + category]
        for image_id in image_ids:
            #try:
                image_data = coco.loadImgs(image_id)[0]
                
                image = io.imread(os.path.join(data_dir, image_type, image_data['file_name']))  
                image = torch.tensor(image/255,dtype=torch.float32).permute(2,0,1).to('cuda')
                
                image = F.interpolate(image.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
                
                null_text_embdding = sd.null_text_inversion(prompt, image, latent=None, num_inference_steps=50, guidance_scale=7.5)
                
                file_name = image_data['file_name'].split('.')[0]
                torch.save(null_text_embdding,f'null_text_embeddings_coco/{file_name}.pt')
                                
            # except:
            #     pass
                i+=1
                # print(i)


