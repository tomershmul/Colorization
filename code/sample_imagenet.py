import argparse
import torch
from model import Color_model
import numpy as np
from skimage.color import rgb2lab, rgb2gray
import torch.nn as nn 
from PIL import Image
import scipy.misc
from torchvision import datasets, transforms
from training_layers import decode
import os


def load_image(image_path, model_output_size, transform=None):
    image = Image.open(image_path)
    
    if transform is not None:
        image = transform(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_small=transforms.Resize(model_output_size)(image)
    image_small=np.expand_dims(rgb2lab(image_small)[:,:,0],axis=-1)
    image=rgb2lab(image)[:,:,0]-50.
    image=torch.from_numpy(image).unsqueeze(0)
    
    return image,image_small


def validate(dataset, ckpt, new_arch):
    print ("Validate: ",dataset, "\n",ckpt)
    # ImageNet64 vs ImageNet changes
    if dataset == "ImageNet64":
        model_output_size = 32 
        upscale = 2
    elif dataset == "ImageNet128":
        model_output_size = 64 
        upscale = 2
    else:
        model_output_size = 56 
        upscale = 4

    scale_transform = transforms.Compose([
        transforms.Resize((model_output_size*upscale, model_output_size*upscale)),
    ])

    data_dir = "../data/val"
    dirs=os.listdir(data_dir)
    color_model = nn.DataParallel(Color_model(new_arch=new_arch)).cuda().eval()
    color_model.load_state_dict(torch.load(ckpt))
     
    for file in dirs:
        image,image_small=load_image(data_dir+'/'+file, model_output_size=model_output_size, transform=scale_transform) #TODO CIFAR 32, imagenet 56
        image=image.unsqueeze(0).float().cuda()
        img_ab_313=color_model(image)
        # out_max=np.argmax(img_ab_313[0].cpu().data.numpy(),axis=0)
        # print('out_max',set(out_max.flatten()))
        print(file)
        # print('image.shape', image.shape)
        # print('img_ab_313.shape', img_ab_313.shape)
        color_img = decode(image, img_ab_313, upscale=upscale)
        color_name = '../data/colorimg/' + file
        scipy.misc.imsave(color_name, color_img*255.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_arch', type=int, default=0)
    parser.add_argument('--ckpt', type = str, default = '../model/models/128x128_orig_arch/model-new0-100-640.ckpt', help = 'path for ckpt models')
    parser.add_argument('--dataset', type = str, default = 'ImageNet128', help = 'ImageNet128, ImageNet64')
    args = parser.parse_args()
    print(args)
    
    validate(dataset=args.dataset, ckpt=args.ckpt, new_arch=args.new_arch)
