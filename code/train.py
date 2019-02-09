import argparse
import os
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision import transforms
# import numpy as np
# from training_layers import PriorBoostLayer, NNEncLayer, ClassRebalanceMultLayer, NonGrayMaskLayer
from training_layers import PriorBoostLayer, NNEncLayer, NonGrayMaskLayer
from data_loader import TrainImageFolder
from model import Color_model
from sample_imagenet import validate
import time
import re

original_transform = transforms.Compose([
    #transforms.Resize((64, 64)),
    #transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])


def main(args):
    # ImageNet64 vs ImageNet changes
    if args.dataset == "ImageNet64":
        model_output_size = 32
        upscale = 2
    elif args.dataset == "ImageNet128":
        model_output_size = 64
        upscale = 2
    else:
        model_output_size = 56
        upscale = 4

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    train_set = TrainImageFolder(args.image_dir, original_transform, model_output_size)

    # Build data loader
    data_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

    # Build the models
    model=nn.DataParallel(Color_model(args.new_arch)).cuda()
    num_epochs_load = 0
    if args.ckpt is not None:
        model.load_state_dict(torch.load('{}'.format(args.ckpt)))
        num_epoch_regex = re.search("model-new\d-(\d+)-.*", args.ckpt)
        num_epochs_load = int(num_epoch_regex.group(1))


    encode_layer=NNEncLayer()
    boost_layer=PriorBoostLayer()
    nongray_mask=NonGrayMaskLayer()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.learning_rate, betas=(0.9, 0.99), weight_decay=1e-3)
    

    # Train the models
    total_step = len(data_loader)
    lr_step = int(args.num_epochs / 3)
    if lr_step == 0:
        lr_step = args.num_epochs
    for epoch in range(args.num_epochs):
        if args.update_lr:
            if epoch > 0 and epoch % lr_step == 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5

        epoch_start_time = time.time()
        try:
            for i, (images, img_ab) in enumerate(data_loader):
                try:
                    step_start_time = time.time()
                    # Set mini-batch dataset
                    images = images.unsqueeze(1).float().cuda()
                    img_ab = img_ab.float()
                    encode,max_encode=encode_layer.forward(img_ab)
                    targets=torch.Tensor(max_encode).long().cuda()
                    boost=torch.Tensor(boost_layer.forward(encode)).float().cuda()
                    mask=torch.Tensor(nongray_mask.forward(img_ab)).float().cuda()
                    boost_nongray=boost*mask
                    outputs = model(images)
    
                    loss = (criterion(outputs,targets)*(boost_nongray.squeeze(1))).mean()
                    model.zero_grad()
                    
                    loss.backward()
                    optimizer.step()
    
                    # Print log info
                    if (i+1) % args.log_step == 0:
                        step_time = time.time() - step_start_time
                        step_fps = args.batch_size / step_time
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}, fps: {:.4f}'
                          .format(num_epochs_load+epoch+1, num_epochs_load+args.num_epochs, i+1, total_step, loss.item(), step_time, step_fps))
    
                except Exception as ex:
                    print (str(ex))
                    #import ipdb; ipdb.set_trace()
                    pass
        except Exception as ex:
            print (str("I: {} ".format(i)) + (str(ex)))
            import ipdb; ipdb.set_trace()
            print (images) 
            print (img_ab)
            pass
                    

        epoch_time = time.time() - epoch_start_time
        epoch_fps = args.batch_size * total_step / epoch_time
        print('Epoch END [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}, fps: {:.4f}, Learning Rate: {:.8f}'
              .format(num_epochs_load+epoch+1, num_epochs_load+args.num_epochs, i+1, total_step, loss.item(), epoch_time, epoch_fps, optimizer.param_groups[0]['lr']))

        # Save the model checkpoints
        if (epoch % args.save_step == args.save_step-1):
            ckpt = 'model-new{}-{}-{}.ckpt'.format(args.new_arch,num_epochs_load+epoch + 1, i + 1)
            ckpt = os.path.join(args.model_path, ckpt)
            torch.save(model.state_dict(), ckpt)
            print('Model saved: {}'.format(ckpt))
            validate(dataset=args.dataset, ckpt=ckpt, new_arch=args.new_arch)
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = '../model/models/', help = 'path for saving trained models')
    # parser.add_argument('--crop_size', type = int, default = 224, help = 'size for randomly cropping images')
    parser.add_argument('--image_dir', type=str, default='../data/imagenet128', help='directory for train images')
    parser.add_argument('--log_step', type = int, default = 100, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 1, help = 'step size for saving trained models')

    # Model parameters
    parser.add_argument('--ckpt', type = str, default = None)
    # parser.add_argument('--num_epochs_load', type=int, default=0)
    parser.add_argument('--num_epochs', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 50)
    parser.add_argument('--num_workers', type = int, default = 16)
    parser.add_argument('--learning_rate', type = float, default = 1e-5)
    parser.add_argument('--update_lr', type=int, default=1)
    parser.add_argument('--new_arch', type=int, default=1)
    parser.add_argument('--dataset', type = str, default = 'ImageNet128', help = 'ImageNet128, ImageNet64')
    args = parser.parse_args()
    print(args)
    main(args)
