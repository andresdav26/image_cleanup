import torch
from pathlib import Path

from network import Model
from image_utils import imload, imsave, padd
import time

def validation(args):
    device = torch.device("cuda" if args.cuda_device_no >= 0 else 'cpu')

    model = Model()
    checkpoint = torch.load(args.models_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    img_paths = [p for p in Path(args.val_path).glob('*') if p.suffix in ('.png', '.jpg', '.jpeg')] # noise
    d = args.cropsize
    for path in img_paths:
        t0 = time.time()
        input_image = imload(path).to(device)

        imNoisy_padd, r, c, padr, padc = padd(input_image,d) # padding
        patchN = imNoisy_padd.unfold(2, d, d).unfold(3, d, d)
        reconT = torch.ones(1,1,r+padr,c+padc).to(device) 
        with torch.no_grad():
            model.eval()
            for i in range(patchN.shape[2]): 
                for j in range(patchN.shape[3]):
                    imgN_p = patchN[:,:,i,j,:,:]
                    output_image = model(imgN_p)
                    reconT[:,:,i*d:(i+1)*d,j*d:(j+1)*d] = output_image 

        imsave(reconT[:,:,0:r,0:c], args.output_path + path.name)
        print(f'Time: {time.time() - t0} seconds')
    return None