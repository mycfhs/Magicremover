import os
import sys
sys.path.append('/home/v-siyuanyang/workspace/drag_diffusion')
sys.path.append('/home/v-siyuanyang/workspace/drag_diffusion/fid_scores')
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import clip
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import json  
import cv2
import random
import argparse

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from evaluation.inception import InceptionV3

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.to_tensor)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):

    path = pathlib.Path(path)
    files = sorted([file for ext in ["jpg","png"]
                    for file in path.glob('*.{}'.format(ext))])
    m, s = calculate_activation_statistics(files, model, batch_size,
                                            dims, device, num_workers)

    return m, s


def calculate_fid_given_paths(real_paths, fake_paths, batch_size, device, dims, num_workers=1):
   


    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(real_paths, model, batch_size,
                                        dims, device, num_workers)


    m2, s2 = compute_statistics_of_path(fake_paths, model, batch_size,
                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value






class CLIP(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()

        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device, jit=False)
        self.clip_model.eval()
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def get_text_embeds(self, prompt, **kwargs):

        text = clip.tokenize(prompt).to(self.device)
        text_z = self.clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True).to(torch.float32)

        return text_z

    def get_img_embeds(self, image, **kwargs):

        image_z = self.clip_model.encode_image(self.aug(image))
        image_z = image_z / image_z.norm(dim=-1, keepdim=True).to(torch.float32)

        return image_z

    
    def calcualte_clip(self,img_path, coco_label_path):

       
        # 加载类别标签文件  
        coco_labels_file = coco_label_path
        
        # 读取 JSON 数据  
        with open(coco_labels_file, 'r') as f:  
            coco_labels_data = json.load(f)  
        
        # 获取类别与 ID 的映射关系  
        category_id_name_map = {category['id']: category['name'] for category in coco_labels_data['categories']}  

        path= img_path 
    
        path = pathlib.Path(path)
        files = sorted([file for ext in ["jpg","png"]
                        for file in path.glob('*.{}'.format(ext))])

        files = random.sample(files,1000)
        
        #files = files[:1000]
        ##  part 1
        #files = files[:1857]+files[3000:9000]
        ## part 2
        #files = files[1857:3000] + files[9000:]


       # original image part global 0.2270
       #sd:    all local  # part local   #all global 0.2116  # part global 0.2117  
       #lama:  all local  # part local   #all global 0.2148  # part global 0.2149  
       #inst:  all local  # part local   #all global         # part global 0.2199
       #m                                                    # part global 0.2181
        
        clip_sim = []
        i=0
        
        for file in files:
            try:
                ########### date form
                # sd-inpaint： image.shape is [3,512,1536]  image[：，：，0：512] is original image。 image[：,：,512：1024] is mask。  image[：，：，1024：1536] is result image
                # lama： image.shape is [3,512,1536]  image[：，：，0：512] is original image。 image[：,：,512：1024] is mask。  image[：，：，1024：1536] is result image
                # Inst-inpaint： image.shape is [3,512,1024]  image[：,：,0：512] is original image   image[：,：.512：1024] is result image

                image = TF.to_tensor(Image.open(file).convert('RGB'))
                # mask_path = '/home/v-siyuanyang/workspace/lama/lama_coco_inpaint/images/' + str(file).split('/')[-1].lstrip('0')
                # mask = TF.to_tensor(Image.open(mask_path).convert('RGB'))
                # mask = mask[:,:,512:1024]
                # _, rows, cols = torch.nonzero(mask, as_tuple=True)  




                #image = image[:,:,1024:].to(self.device)
                #image = image[:,:,1024:].to(self.device)
                image = image[:,:,512:].to(self.device)
                
                #image = image[:,rows.min():rows.max(), cols.min():cols.max()]
                
                #coco_object_id  = int(str(file).split('_')[-1].split('.')[0])
                text= category_id_name_map[int(str(file).split('_')[-1].split('.')[0])]

                image_emb = self.get_img_embeds(image.unsqueeze(0))
                text_emb = self.get_text_embeds(text)

                clip_score = (image_emb * text_emb).sum(-1)
                clip_sim.append(clip_score)
                print(i)
                i+=1
                
            except:
                pass

        return sum(clip_sim)/len(clip_sim)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_image_path', type=str)
    parser.add_argument('--fake_image_path', type=str)
    parser.add_argument('--coco_label_path', type=str)
    parser.add_argument('--task', type=str)
    
    args = parser.parse_args()    

    if args.task=='fid':
        #####. calcualte fid
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')


        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0

        fid_value = calculate_fid_given_paths(args.real_image_path,   
                                              args.fake_image_path,
                                              batch_size=50,                                     
                                              device=device,
                                              dims=2048,                                    
                                              num_workers=num_workers)
        print('FID: ', fid_value)



    elif args.task=='cs':
    ######### calculate clip
        device = 'cuda'
        with torch.no_grad():
            clip = CLIP(device)
            clip_sim = clip.calcualte_clip(args.fake_path,
                                           args.coco_label_path)
        print(clip_sim)

    else:
        print('wrong task')



if __name__ == '__main__':
    main()
