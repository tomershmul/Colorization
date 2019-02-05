import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from training_layers import PriorBoostLayer, NNEncLayer, ClassRebalanceMultLayer, NonGrayMaskLayer
from data_loader import TrainImageFolder
from model import Color_model
import time

original_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])


def main(args):
    # CIFAR10 vs ImageNet changes
    work_dataset = "CIFAR10"
    if work_dataset == "CIFAR10":
        model_output_size = 32 # TODO 32 CIFAR, 56 in ImageNet
        upscale = 2
    else:
        model_output_size = 56 # TODO 32 CIFAR, 56 in ImageNet
        upscale = 4

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    train_set = TrainImageFolder(args.image_dir, original_transform, model_output_size)

    # Build data loader
    data_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

    # Build the models
    model=nn.DataParallel(Color_model()).cuda()
    if args.num_epochs_load != 0:
        model.load_state_dict(torch.load('../model/models/model-{}-2500.ckpt'.format(args.num_epochs_load))) # TODO take from input arg
    encode_layer=NNEncLayer()
    boost_layer=PriorBoostLayer()
    nongray_mask=NonGrayMaskLayer()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.learning_rate)
    

    # Train the models
    total_step = len(data_loader)
    lr_step = int(args.num_epochs / 4)
    if lr_step == 0:
        lr_step = args.num_epochs
    for epoch in range(args.num_epochs):
        if epoch % lr_step == 0:
            for g in optimizer.param_groups:
                print('Learning Rate: {:.4f}'.format(g['lr']))
                g['lr'] = g['lr'] / 2
                print('Updated Learning Rate: {:.4f}'.format(g['lr']))

        epoch_start_time = time.time()
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
                if i % args.log_step == 0:
                    step_time = time.time() - step_start_time
                    step_fps = args.batch_size / step_time
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}, fps: {:.4f}'
                      .format(args.num_epochs_load+epoch, args.num_epochs_load+args.num_epochs, i, total_step, loss.item(), step_time, step_fps))

            except Exception as ex:
                print (str(ex))
                pass
        # Save the model checkpoints
        if (epoch % args.save_step == args.save_step-1):
            torch.save(model.state_dict(), os.path.join(
                args.model_path, 'model-{}-{}.ckpt'.format(args.num_epochs_load+epoch + 1, i + 1)))
            print('Model saved: ', 'model-{}-{}.ckpt'.format(args.num_epochs_load+epoch + 1, i + 1))

        epoch_time = time.time() - epoch_start_time
        epoch_fps = args.batch_size * total_step / epoch_time
        print('Epoch END [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}, fps: {:.4f}'
              .format(args.num_epochs_load+epoch, args.num_epochs_load+args.num_epochs, i, total_step, loss.item(), epoch_time, epoch_fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = '../model/models/', help = 'path for saving trained models')
    parser.add_argument('--crop_size', type = int, default = 224, help = 'size for randomly cropping images')
    #parser.add_argument('--image_dir', type = str, default = '../data/images256', help = 'directory for resized images')
    parser.add_argument('--image_dir', type=str, default='../data/imagenet64', help='directory for resized images')
    # parser.add_argument('--image_dir', type=str, default='../data/imagenet128', help='directory for resized images')
    parser.add_argument('--log_step', type = int, default = 100, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 5, help = 'step size for saving trained models')

    # Model parameters
    parser.add_argument('--num_epochs_load', type = int, default = 0)
    parser.add_argument('--num_epochs', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_workers', type = int, default = 16)
    parser.add_argument('--learning_rate', type = float, default = 1e-4)
    args = parser.parse_args()
    print(args)
    main(args)
