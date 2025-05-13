import json  
from PIL import Image  
import os  
import torchvision
import torch
import torch.functional as F
import torchvision.transforms as TF
to_tensor = TF.ToTensor()

# 加载标注信息  
with open('/home/v-siyuanyang/visual_genome/objects.json', 'r') as f:  
    data = json.load(f)  
  
# 遍历标注信息并处理每个样本  
for item in data:  
    # 获取图片ID  
    image_id = item['id']  
    # 构建图片文件路径  
    image_file = os.path.join('/home/v-siyuanyang/visual_genome/VG_100K', f'{image_id}.jpg')  

  
    # 遍历图片中的所有物体  
    for obj in item['objects']:  
        image = Image.open(image_file)  
        # 获取类别信息和bounding box  
        category = obj['names'][0]  # 物体可能有多个名字  
        h,w,x,y = obj['h'],obj['w'],obj['x'],obj['y']

        image = to_tensor(image)
        a = image.clone()
        a[:,y:y+h,x:x+w]=0
        torchvision.utils.save_image(a,'image1.png')
        # 输出结果  

        torchvision.utils.save_image(torch.concat((image,a),dim=-1),f'visual_genome/{image_id}_{category}.png')
        
        print(1)


  

