import torch
import torchvision
from PIL import Image
import numpy as np
import cv2
def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std


def calculate_mean_std_vector(image, mask):
    # Get non-masked pixels
    masked_pixels = image[mask == 0]
 
    # Calculate mean and std for each channel
    channel_mean = torch.mean(masked_pixels)
    channel_std = torch.std(masked_pixels)
 
    return channel_mean, channel_std

def adaptive_instance_normalization(content_feat, style_feat, mask):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calculate_mean_std_vector(style_feat, mask)
    content_mean, content_std = calculate_mean_std_vector(content_feat,mask)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)



# img = Image.open('16_ori.png').convert('RGB')
# img = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0)/255

# ori_img = img[:,:,:,:512]
# mask = img[:,:,:,512:1024]

# inpaint_img = Image.open('16_m.png').convert('RGB')
# inpaint_img = torch.tensor(np.array(inpaint_img)).permute(2,0,1).unsqueeze(0)/255
# inpaint_img = torch.nn.functional.interpolate(inpaint_img,(512,512),mode='bilinear', align_corners=False)


img = Image.open('64.png').convert('RGB')
img = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0)/255
ori_img = torch.nn.functional.interpolate(img,(512,512),mode='bilinear', align_corners=False)


mask = Image.open('64_mask.png').convert('RGB')
mask  = torch.tensor(np.array(mask )).permute(2,0,1).unsqueeze(0)/255
mask  = torch.nn.functional.interpolate(mask ,(512,512),mode='bilinear', align_corners=False)

inpaint_img = Image.open('64_m.png').convert('RGB')
inpaint_img = torch.tensor(np.array(inpaint_img)).permute(2,0,1).unsqueeze(0)/255
inpaint_img = torch.nn.functional.interpolate(inpaint_img,(512,512),mode='bilinear', align_corners=False)



# mask = Image.open('2.png').convert('RGB')
# mask = torch.tensor(np.array(mask)).permute(2,0,1).unsqueeze(0)/255
# mask = torch.nn.functional.interpolate(mask,(512,512),mode='bilinear', align_corners=False)
# #mask = mask[:,:1,:,:]
# mask = mask.gt(0).to(torch.float32)

img_torch = adaptive_instance_normalization(inpaint_img, ori_img, mask) # color correction (1, 3, H, W)




mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()  
kernel = np.ones((15,15),np.uint8)  
mask_np = cv2.dilate(mask_np, kernel, iterations = 1)  

blurred_mask_np = cv2.GaussianBlur(mask_np, (21, 21), 0)  
# 将处理后的 mask 转回 PyTorch tensor  
mask = torch.from_numpy(blurred_mask_np).float().unsqueeze(0).unsqueeze(0)  
result_img = img_torch * (1.0-mask) + inpaint_img * mask

torchvision.utils.save_image(result_img,'5.png')





import cv2
import numpy as np
 
def normalize_image(image):
    # Imagenet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (image / 255.0 - mean) / std
    return normalized_image
 

 
# Load image and mask
image_path = 'path/to/your/image.jpg'
mask_path = 'path/to/your/mask.png'
 
image = cv2.imread(image_path)[:, :, ::-1]  # Read BGR image and convert to RGB
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
 
# Normalize image
normalized_image = normalize_image(image)
 
# Calculate mean and std for non-masked pixels
channel_mean, channel_std = calculate_mean_std(normalized_image, mask)
 
print("Mean for each channel:", channel_mean)
print("Standard deviation for each channel:", channel_std)
