from pathlib import Path
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image
from itertools import product

import os
import argparse

def split_data(src_path): 

    noise_path = src_path + "noise3/"
    img_paths = [p for p in Path(noise_path).glob('*') if p.suffix in ('.png', '.jpg', '.jpeg')] # noisy images path
    train_paths, test_paths = train_test_split(img_paths, train_size=11, random_state=3)

    name_train = []
    for p in train_paths:  
        name_train.append(str(p.name))

    name_test = []
    for p in test_paths:  
        name_test.append(str(p.name))

    return name_train, name_test

def augmentation(src_path, name_img_train, name_img_test):

    d = 256
    # TRAIN DATA
    for name_file in name_img_train: 
        
        imgC = Image.open(os.path.join(src_path + "clean3/" + name_file)).convert('L')
        imgN = Image.open(os.path.join(src_path + "noise3/" + name_file)).convert('L')

        name_file, f_ext = os.path.splitext(name_file)
        # resize 
        for rs in [0.7,1.0,1.4]:
            imgN_rs = imgN.resize((int(imgN.size[0]*rs),int(imgN.size[1]*rs)), Image.Resampling.LANCZOS)
            imgC_rs = imgC.resize((int(imgC.size[0]*rs),int(imgC.size[1]*rs)), Image.Resampling.LANCZOS)
            
            # crop in overlapping blocks
            w, h = imgN_rs.size
            s = h%d 
            yi = 0
            xi = 0
            for i in range(0, h-h%d, d): 
                for j in range(0, w-w%d, d):
                    if i == 0 and j == 0: 
                        xi = 0
                        yi = 0
                    elif i == 0: 
                        yi = 0
                        xi = 1
                    elif j == 0: 
                        xi = 0
                        yi = 1
                    else: 
                        xi = 1
                        yi = 1

                    box = (j+s*xi, i+s*yi, j+s*xi+d, i+s*yi+d)

                    imgN_rs_crop = imgN_rs.crop(box)
                    imgC_rs_crop = imgC_rs.crop(box)
                    imgN_rs_crop.save(os.path.join(src_path + "train3/noise/" + name_file + str(j) + '_' + str(i) + f_ext))
                    imgC_rs_crop.save(os.path.join(src_path + "train3/clean/" + name_file + str(j) + '_' + str(i) + f_ext))
                
                    # Blur each block  
                    Blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
                    for b in range(4):
                        imgN_b = Blur(imgN_rs_crop)
                        imgC_b = Blur(imgC_rs_crop)

                        imgN_b.save(os.path.join(src_path + "train3/noise/" + name_file + '_blur_' + str(b) + '_' + str(j) + '_' + str(i) + f_ext))
                        imgC_b.save(os.path.join(src_path + "train3/clean/" + name_file + '_blur_' + str(b) + '_' + str(j) + '_' + str(i) + f_ext))

                    # change perspective each block 
                    perspective = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
                    for p in range(4): 
                        imgN_p = perspective(imgN_rs_crop)
                        imgC_p = perspective(imgC_rs_crop)
                        imgN_p.save(os.path.join(src_path + "train3/noise/" + name_file + '_pers_' + str(p) + '_' + str(j) + '_' + str(i) + f_ext))
                        imgC_p.save(os.path.join(src_path + "train3/clean/" + name_file + '_pers_' + str(p) + '_' + str(j) + '_' + str(i) + f_ext))
    # TEST DATA 
    for name_file in name_img_test: 
        
        imgC = Image.open(os.path.join(src_path + "clean3/" + name_file)).convert('L')
        imgN = Image.open(os.path.join(src_path + "noise3/" + name_file)).convert('L')
        for rs in [0.7,1.0,1.4]:
            imgN_rs = imgN.resize((int(imgN.size[0]*rs),int(imgN.size[1]*rs)), Image.Resampling.LANCZOS)
            imgC_rs = imgC.resize((int(imgC.size[0]*rs),int(imgC.size[1]*rs)), Image.Resampling.LANCZOS)
            
            # crop in overlapping blocks
            w, h = imgN_rs.size
            s = h%d 
            yi = 0
            xi = 0
            for i in range(0, h-h%d, d): 
                for j in range(0, w-w%d, d):
                    if i == 0 and j == 0: 
                        xi = 0
                        yi = 0
                    elif i == 0: 
                        yi = 0
                        xi = 1
                    elif j == 0: 
                        xi = 0
                        yi = 1
                    else: 
                        xi = 1
                        yi = 1

                    box = (j+s*xi, i+s*yi, j+s*xi+d, i+s*yi+d)

                    imgN_rs_crop = imgN_rs.crop(box)
                    imgC_rs_crop = imgC_rs.crop(box)
                    imgN_rs_crop.save(os.path.join(src_path + "test3/noise/" + name_file + str(j) + '_' + str(i) + f_ext))
                    imgC_rs_crop.save(os.path.join(src_path + "test3/clean/" + name_file + str(j) + '_' + str(i) + f_ext))
    

# Args 
parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str,
                    help="source image path") 
args = parser.parse_args()

# split data 
name_train, name_test = split_data(args.src_path)

# data augmentation 
augmentation(args.src_path,name_train,name_test)