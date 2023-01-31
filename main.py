import os
import argparse
import time

from train import training
from val import validation

def build_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected !!!')

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda-device-no', type=int,
                    help='cpu: -1, gpu: 0 ~ n ', default=0)

    parser.add_argument('--train-flag', type=str2bool,
                    help='Train flag', required=True)

    parser.add_argument('--epochs', type=int,
                    help='epochs amount', default=100)

    parser.add_argument('--batch', type=int,
                    help='Batch size', default=4)

    parser.add_argument('--cropsize', type=int,
                    help='Size for crop image durning training', default=256)

    parser.add_argument('--vgg-flag', type=str,
                    help='VGG flag for calculating losses', default='vgg19')

    parser.add_argument('--content-layers', type=int, nargs='+', 
                    help='layer indices to extract content features', default=[2]) ## ? 
    
    parser.add_argument('--style-layers', type=int, nargs='+',
                    help='layer indices to extract style features', default=[0, 5, 10, 19, 28]) ## ?

    parser.add_argument('--l1_weight', type=float,
                    help='l1 pixel-level loss weight', default=10.0)

    parser.add_argument('--content_weight', type=float, 
                    help='content loss weight', default=0.1)
    
    parser.add_argument('--style_weight', type=float,
                    help='style loss weight', default=10.0)

    parser.add_argument('--train_path', type=str,
                    help='noise images path')
    
    parser.add_argument('--test_path', type=str,
                    help="clean image path")
    
    parser.add_argument('--val_path', type=str,
                    help="clean image path")
    
    parser.add_argument('--save_model', type=str,
                    help='Save model', default='./trained_models/')
    
    parser.add_argument('--output_path', type=str,
                    help='output image path', default='./results/')
    
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args= parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_no)

    if args.train_flag:
        transform_network = training(args)
    else:
        validation(args)
        