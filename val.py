import torch
from pathlib import Path

from network import Model
from image_utils import imload, imsave
import time

def validation(args):
    device = torch.device("cuda" if args.cuda_device_no >= 0 else 'cpu')

    model = Model()
    checkpoint = torch.load(args.models_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    img_paths = [p for p in Path(args.val_path).glob('*') if p.suffix in ('.png', '.jpg', '.jpeg')] # noise
    for path in img_paths:
        t0 = time.time()
        input_image = imload(path,cropsize=512).to(device)

        with torch.no_grad():
            model.eval()
            output_image = model(input_image)

        imsave(output_image, args.output_path + path.name)
        print(f'Time: {time.time() - t0} seconds')
    return None