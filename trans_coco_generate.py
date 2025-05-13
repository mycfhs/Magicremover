import os  
import numpy as np
import skimage.io as io  
from pycocotools.coco import COCO  
from IPython.display import Image  
import torch 
import torchvision
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F  
import torchvision.transforms as TF
from PIL import Image
import cv2
from collections import defaultdict
import random
import copy
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--trans_ratio', type=int)
    
    args = parser.parse_args()    

    random.seed(0)
    ################ coco val
    # 设置图片和注释文件夹路径  
    data_dir = args.data_dir
    image_type = 'val2017'  
    annotation_file = 'instances_{}.json'.format(image_type)  
    
    # 初始化 COCO 对象  
    coco = COCO(os.path.join(data_dir, 'annotations', annotation_file))  
    
    
    categories = coco.loadCats(coco.getCatIds())  
    category_names = [category['name'] for category in categories]   

    to_tensor = TF.ToTensor()

    with torch.no_grad():
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        prompt = " " 
        
        files_a = defaultdict(list)
        for category in category_names:
            cat_id = coco.getCatIds(catNms=category )[0]  
            image_ids = coco.getImgIds(catIds=[cat_id])  
            files_a[cat_id] = image_ids

        cat_s = list(files_a.keys())
        files_b = copy.deepcopy(files_a)
        
        cat_a_list = [4,17,18,19,20,21,22,23,24,44,47,52,54,59,76,78,85]


        i=0
        while i < 1010:
            
            cat_a = random.choice(cat_a_list)
            cat_b_list = [i for i in cat_s if i != cat_a] 
            cat_b = random.choice(cat_b_list)
            img_a_id = random.choice(files_a[cat_a])
            # files_a[cat_a].remove(img_a_id)
            # if files_a[cat_a]==[]:
            #     cat_s.remove(cat_a)
            
            img_b_id = random.choice(files_b[cat_b])
            # files_b[cat_b].remove(img_b_id)
            # if files_b[cat_b]==[]:
            #     cat_s.remove(cat_b)

            img_a_data = coco.loadImgs(img_a_id)[0]
            img_a = io.imread(os.path.join(data_dir, image_type, img_a_data['file_name']))  
            img_a = to_tensor(img_a)
            
            img_b_data = coco.loadImgs(img_b_id)[0]
            img_b = io.imread(os.path.join(data_dir, image_type, img_b_data['file_name']))  
            img_b = to_tensor(img_b)

            
            # 获取图像的注释  
            annotation_ids = coco.getAnnIds(imgIds=img_a_data['id'], catIds=[cat_a], iscrowd=None)  
            annotations = coco.loadAnns(annotation_ids)  
            
            # 创建一个空的 mask 图像  
            mask_a = np.zeros((img_a_data['height'], img_a_data['width']))  
            
            # 根据注释为每个实例添加 mask  
            for annotation in annotations:  
                mask_a += coco.annToMask(annotation)  
            mask_a = torch.tensor(mask_a)
            mask_a = (mask_a>0.5).to(torch.float32)

        
            img_a = F.interpolate(img_a.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
            img_b = F.interpolate(img_b.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
            mask_a = F.interpolate(mask_a.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)   
            
            if mask_a.sum()/(mask_a.shape[-2]*mask_a.shape[-1]) < 0.1 or mask_a.sum()/(mask_a.shape[-2]*mask_a.shape[-1]) > 0.4:
                continue


            # random.uniform(0.5, 0.8)  
            # img_c = img_a * mask_a * transp + img_b * mask_a * (1-transp) +img_b * (1-mask_a)
            # torchvision.utils.save_image(mask_a,f'trans_coco/non_trans/mask/{img_b_id}_{cat_a}.png')
            # torchvision.utils.save_image(img_b,f'trans_coco/non_trans/gt/{img_b_id}_{cat_a}.png')
            # torchvision.utils.save_image(img_c,f'trans_coco/non_trans/masked_image/{img_b_id}_{cat_a}.png')
            # torchvision.utils.save_image(img_a,f'trans_coco/non_trans/object/{img_b_id}_{cat_a}.png')

            transp = args.trans_ratio
            img_c = img_a * mask_a * transp + img_b * mask_a * (1-transp) +img_b * (1-mask_a)
            torchvision.utils.save_image(mask_a,f'trans_coco/non_trans/mask/{img_b_id}_{cat_a}.png')
            torchvision.utils.save_image(img_b,f'trans_coco/non_trans/gt/{img_b_id}_{cat_a}.png')
            torchvision.utils.save_image(img_c,f'trans_coco/non_trans/masked_image/{img_b_id}_{cat_a}.png')
            torchvision.utils.save_image(img_a,f'trans_coco/non_trans/object/{img_b_id}_{cat_a}.png')
            print(i)
            i+=1
            #torchvision.utils.save_image(img_c,'img_c.png')
            #print(1)

