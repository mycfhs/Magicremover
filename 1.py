import torchvision
from PIL import Image,ImageSequence
import numpy as np
import torch
img_path = 'gif/1.png'

# img = Image.open(img_path).resize((512,512))
# img.save('4.png')

gif = Image.open('ezgif-4-0db51df695a8.gif')

crop_transform = torchvision.transforms.CenterCrop((557, 557)) 
gi = iter(ImageSequence.Iterator(gif))
i=0
while gi:
    f = next(gi)
    img = torch.tensor(np.array(f.convert('RGB'))).permute(2,0,1).unsqueeze(0)/255
    img = crop_transform(img)
    torchvision.utils.save_image(img,f'gif/{i}.png')
    i+=1




# img = Image.open("10.png").convert('RGB')
# img = torch.tensor(np.array(img),device=device).permute(2,0,1).unsqueeze(0)/255
# img = torch.nn.functional.interpolate(img,(512,512),mode='bilinear', align_corners=False)
# mask = Image.open("11.png").convert('RGB')
# mask = torch.tensor(np.array(mask),device=device).permute(2,0,1).unsqueeze(0)/255       
# img = img * (1-mask) + 0.05 * mask



