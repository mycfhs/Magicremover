import os  
import numpy as np
import skimage.io as io  
from pycocotools.coco import COCO  
from IPython.display import Image  
import torch 
import torchvision
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F  
from PIL import Image
import cv2

################ coco val
# 设置图片和注释文件夹路径  
data_dir = '/home/v-siyuanyang/coco'  
image_type = 'val2017'  
annotation_file = 'instances_{}.json'.format(image_type)  
  
# 初始化 COCO 对象  
coco = COCO(os.path.join(data_dir, 'annotations', annotation_file))  
  
  
categories = coco.loadCats(coco.getCatIds())  
category_names = [category['name'] for category in categories]   

with torch.no_grad():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    prompt = " " 

    i = 0
    for category in category_names:
        cat_id = coco.getCatIds(catNms=category )[0]  
        image_ids = coco.getImgIds(catIds=[cat_id])  
        
        for image_id in image_ids:
            try:
                if i<7000:
                    continue
                image_data = coco.loadImgs(image_id)[0]
                
                image = io.imread(os.path.join(data_dir, image_type, image_data['file_name']))  
                if len(image.shape)==2:
                    print('shape error')
                else:
                    image = torch.tensor(image/255).permute(2,0,1)
                    
                    # 获取图像的注释  
                    annotation_ids = coco.getAnnIds(imgIds=image_data['id'], catIds=[cat_id], iscrowd=None)  
                    annotations = coco.loadAnns(annotation_ids)  
                    
                    # 创建一个空的 mask 图像  
                    mask = np.zeros((image_data['height'], image_data['width']))  
                    
                    # 根据注释为每个实例添加 mask  
                    for annotation in annotations:  
                        mask += coco.annToMask(annotation)  
                    
                    kernel = np.ones((20, 20), np.uint8) 
                    mask = cv2.dilate(mask, kernel, iterations=1)  
                    
                    mask = torch.tensor(mask)
                    mask = (mask>0.5).to(torch.float32)
                    
                    image = F.interpolate(image.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
                    image_inpaint = pipe(prompt=prompt, image=(image-0.5) * 2, mask_image=mask).images[0]
                    
                    image_inpaint = torch.tensor(np.array(image_inpaint)/255).permute(2,0,1).unsqueeze(0)
                    
                    result = torch.concat((image,mask.repeat(1,3,1,1),image_inpaint),dim=3)
                    
                    image_id = str(image_data['id']) + '_' + str(cat_id)
                    torchvision.utils.save_image(result, f'sd_inpaint_coco/images_2/{image_id}.png')
            
            except:
                pass
            i+=1
            print(i)



################  user val

data_path = 'new_test_image'



with torch.no_grad():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    prompt = " " 
    
    i=0
    for root, dirs, files in os.walk(data_path+'/image'):  
        for image_path in files:
                    
            image = Image.open(data_path+'/image/'+image_path).resize((512,512))
            mask_path =data_path+'/mask/'+ image_path.split('.')[0] +'.png'
            mask = Image.open(mask_path).resize((512,512))
            
            image = torch.tensor(np.array(image)/255).permute(2,0,1).unsqueeze(0)
            mask = torch.tensor(np.array(mask)/255).permute(2,0,1)[:1].unsqueeze(0)
            
            image = image[:,0:3,:,:]
            
            
            image_inpaint = pipe(prompt=prompt, image=(image-0.5) * 2, mask_image=mask).images[0]            
            image_inpaint = torch.tensor(np.array(image_inpaint)/255).permute(2,0,1).unsqueeze(0)            
            result = torch.concat((image,mask.repeat(1,3,1,1),image_inpaint),dim=3)
            save_path = data_path+'/sd_inpaint/'+ image_path.split('.')[0] +'.png'
            torchvision.utils.save_image(result, save_path)
            
            save_path = data_path+'/sd_inpaint/'+ image_path.split('.')[0] +'_1.png'
            torchvision.utils.save_image(image+mask*0.3, save_path )
            
            save_path = data_path+'/sd_inpaint/'+ image_path.split('.')[0] +'_2.png'
            torchvision.utils.save_image(image_inpaint, save_path )

            i+=1
            print(i)
            
    
