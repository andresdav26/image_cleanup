import os
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms


# mean and std of ImageNet to use pre-noiseed VGG
# Grayscale 
# IMAGENET_MEAN = 0.44531356896770125
# IMAGENET_STD = 0.2692461874154524
# RGB
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                 std=IMAGENET_STD)

denormalize = transforms.Normalize(mean=[-mean/std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/std for std in IMAGENET_STD])
# denormalize = transforms.Normalize(mean = -IMAGENET_MEAN/IMAGENET_STD, std = 1/IMAGENET_STD)

class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, noise_paths, clean_paths, transform, use_cache = False):
        self.clean_paths = clean_paths
        self.noise_paths = noise_paths     
        self.transform = transform
        self.cached_data = []
        self.use_cache = use_cache
        
    def __len__(self):
        return len(self.clean_paths)
    
    def __getitem__(self, index):
        if not self.use_cache:
            imgC = Image.open(self.clean_paths[index])
            imgN = Image.open(self.noise_paths[index])
            
            imgC = self.transform(imgC)
            imgN = self.transform(imgN)
            self.cached_data.append((imgC,imgN))
        else: 
            imgC, imgN = self.cached_data[index]
        return imgC, imgN 

    def set_use_cache(self, use_cache):
            if use_cache:
                self.cached_data = self.cached_data
            else:
                self.cached_data = []
            self.use_cache = use_cache

def get_transformer(cropsize=None):
    transformer = []
    if cropsize:
        transformer.append(transforms.RandomCrop(cropsize))
    transformer.append(transforms.ToTensor())
    # transformer.append(normalize)
    return transforms.Compose(transformer)

def imsave(tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)    
    # torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
    torchvision.utils.save_image(tensor.clamp_(0.0, 1.0), path)
    return None
    
def imload(path, cropsize=None):
    transformer = get_transformer(cropsize)
    return transformer(Image.open(path).convert("L")).unsqueeze(0)



