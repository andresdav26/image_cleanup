import time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from decimal import Decimal
from pathlib import Path
from network import Model
from image_utils import ImageFolder, get_transformer

mse_criterion = torch.nn.MSELoss(reduction='mean')
L1_criterion = torch.nn.L1Loss(reduction='mean')

def extract_features(model, x, layers):
    features = list()
    for index, layer in enumerate(model):
        x = layer(x)
        if index in layers:
            features.append(x)
    return features

def calc_Content_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
    
    content_loss = 0
    for f, t, w in zip(features, targets, weights):
        content_loss += L1_criterion(f, t) * w
        
    return content_loss

def gram(x):
    b ,c, h, w = x.size()
    g = torch.bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1,2))
    return g.div(h*w)

def calc_Gram_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
        
    gram_loss = 0
    for f, t, w in zip(features, targets, weights):
        gram_loss += mse_criterion(gram(f), gram(t)) * w
    return gram_loss

   
def training(args):
    device = torch.device("cuda" if args.cuda_device_no >= 0 else 'cpu')
    print('Using device:', device)
    
    # Dataset
    n_train_paths = [p for p in Path(args.train_path + "noise/").glob('*') if p.suffix in ('.png', '.jpg', '.jpeg')] # noise
    c_train_paths = [p for p in Path(args.train_path + "clean/").glob('*') if p.suffix in ('.png', '.jpg', '.jpeg')] # clean
    train_dataset = ImageFolder(n_train_paths, c_train_paths, get_transformer(args.cropsize), use_cache=False)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    n_test_paths = [p for p in Path(args.test_path + "noise/").glob('*') if p.suffix in ('.png', '.jpg', '.jpeg')] # noise
    c_test_paths = [p for p in Path(args.test_path + "clean/").glob('*') if p.suffix in ('.png', '.jpg', '.jpeg')] # clean
    test_dataset = ImageFolder(n_test_paths, c_test_paths, get_transformer(args.cropsize), use_cache=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    # Loss network
    loss_network = torchvision.models.__dict__[args.vgg_flag](weights='VGG19_Weights.DEFAULT').features.to(device)

    # Transform Network
    model = Model()
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters())

    tb = SummaryWriter()

    # MAIN LOOP 
    worst_loss = 1000
    for epoch in range(args.epochs):
        t0 = time.time()
    
        # Train
        model.train()
        train_loss = 0
        train_l1_loss = 0
        train_cont_loss = 0 
        train_style_loss = 0
        train_mse = 0
        train_psnr = 0 
        
        if epoch == 1:
            trainloader.dataset.set_use_cache(use_cache=True)
            trainloader.num_workers = 4
            
        for pair in trainloader:
            # data 
            imgC, imgN = pair[0].to(device), pair[1].to(device) # [batch,ch,h,w]

            optimizer.zero_grad() 

            # reconstruction
            out = model(imgN)

            # Loss function
            l1_loss = L1_criterion(out,imgC)

            target_content_features = extract_features(loss_network, imgC.expand(-1,3,-1,-1), args.content_layers)
            target_style_features = extract_features(loss_network, imgC.expand(-1,3,-1,-1), args.style_layers) 

            output_content_features = extract_features(loss_network, out.expand(-1,3,-1,-1), args.content_layers)
            output_style_features = extract_features(loss_network, out.expand(-1,3,-1,-1), args.style_layers)

            content_loss = calc_Content_Loss(output_content_features, target_content_features)
            style_loss = calc_Gram_Loss(output_style_features, target_style_features)
        
            total_loss =  l1_loss * args.l1_weight + content_loss * args.content_weight + style_loss * args.style_weight
     
            train_loss += total_loss.item()
            train_l1_loss += l1_loss.item()
            train_cont_loss += content_loss.item()
            train_style_loss += style_loss.item()
            train_mse += mse_criterion(out,imgC).item()
            ## 
            total_loss.backward()
            optimizer.step()

        # metrics
        train_loss /= len(trainloader.dataset)
        train_l1_loss /= len(trainloader.dataset)
        train_cont_loss /= len(trainloader.dataset)
        train_style_loss /= len(trainloader.dataset)
        train_mse /= len(trainloader.dataset)
        train_psnr = 10 * torch.log10(torch.tensor([1]) /train_mse) 
        epoch_time = time.time() - t0

        # Test 
        with torch.no_grad():
            model.eval()
            test_loss = 0
            test_l1_loss = 0
            test_cont_loss = 0
            test_style_loss = 0
            mse_test = 0
            test_psnr = 0
            
            if epoch == 1:
                testloader.dataset.set_use_cache(use_cache=True)
                testloader.num_workers = 4

            for pair in testloader:
                # data 
                imgC, imgN = pair[0].to(device), pair[1].to(device)

                # reconstruction
                out = model(imgN)

                # Loss function
                l1_loss = L1_criterion(out,imgC)

                target_content_features = extract_features(loss_network, imgC.expand(-1,3,-1,-1), args.content_layers)
                target_style_features = extract_features(loss_network, imgC.expand(-1,3,-1,-1), args.style_layers) 

                output_content_features = extract_features(loss_network, out.expand(-1,3,-1,-1), args.content_layers)
                output_style_features = extract_features(loss_network, out.expand(-1,3,-1,-1), args.style_layers)

                content_loss = calc_Content_Loss(output_content_features, target_content_features)
                style_loss = calc_Gram_Loss(output_style_features, target_style_features)
        
                total_loss =  l1_loss * args.l1_weight + content_loss * args.content_weight + style_loss * args.style_weight

                test_loss += total_loss.item()
                test_l1_loss += l1_loss.item()
                test_cont_loss += content_loss.item()
                test_style_loss += style_loss.item()

                mse_test += mse_criterion(out,imgC).item()

            # metrics
            test_loss /= len(testloader.dataset)
            test_l1_loss /= len(testloader.dataset)
            test_cont_loss /= len(testloader.dataset)
            test_style_loss /= len(testloader.dataset)
            mse_test /= len(testloader.dataset)
            test_psnr = 10 * torch.log10(torch.tensor([1]) / mse_test) ## ERROR 
        
        # tensorboard 
        tb.add_scalars('Loss', {'Train loss':train_loss,'Test loss':test_loss}, epoch)
        tb.add_scalars('L1_loss', {'Train l1_loss':train_l1_loss,'Test l1_loss':test_l1_loss}, epoch)
        tb.add_scalars('content_loss', {'Train cont_loss':train_cont_loss,'Test cont_loss':test_cont_loss}, epoch)
        tb.add_scalars('style_loss', {'Train style_loss':train_style_loss,'Test style_loss':test_style_loss}, epoch)
        tb.add_scalars('PSNR', {'Train psnr':train_psnr.item(),'Test psnr':test_psnr.item()}, epoch)

        # Save model 
        if worst_loss > test_loss:
            worst_loss = test_loss
            state = {'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
            torch.save(state, args.save_model + 'best2.pth')

        print('Epoch: {}, Train loss: {:.2f}, Test loss: {:.2f}, Train_psnr: {:,.2f} , Test_psnr: {:,.2f}, time: {:,.2f}'.format(epoch, Decimal(train_loss), 
                Decimal(test_loss), train_psnr.item(), test_psnr.item(), epoch_time))
    
    tb.close()

    return model 


