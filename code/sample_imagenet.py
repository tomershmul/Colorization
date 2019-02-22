import argparse
import torch
from model import Color_model
import numpy as np
from skimage.color import rgb2lab, rgb2gray
import torch.nn as nn 
from PIL import Image
import scipy.misc
from torchvision import datasets, transforms
import os
import torch.nn.functional as F
from skimage import color


def load_image(image_path, model_output_size, transform=None):
    ''' Transform image + convert to L-ab color space'''
    image = Image.open(image_path)
    
    if transform is not None:
        image = transform(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image=rgb2lab(image)[:,:,0]-50.
    image=torch.from_numpy(image).unsqueeze(0)
    
    return image

def decode(data_l, conv8_313, upscale=2):
    ''' upscale==4 for imagenet orig, upscale==2 for otherwise'''
    resources_dir = './resources'
    # bias level
    data_l=data_l[0]+50
    data_l=data_l.cpu().data.numpy().transpose((1,2,0))
    conv8_313 = conv8_313[0]
    # Soft-max
    class8_313 = F.softmax(conv8_313,dim=0).cpu().data.numpy().transpose((1,2,0))
    # Take arg-max
    class8=np.argmax(class8_313,axis=-1)
    # Take quantization
    cc = np.load(os.path.join(resources_dir, 'pts_in_hull.npy'))
    data_ab=cc[class8[:][:]]
    # Upscale NN
    data_ab=data_ab.repeat(upscale, axis=0).repeat(upscale, axis=1)
    # Concat with L-channel
    img_lab = np.concatenate((data_l, data_ab), axis=-1)
    img_rgb = color.lab2rgb(img_lab)

    return img_rgb

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
        # Load image
        image=load_image(data_dir+'/'+file, model_output_size=model_output_size, transform=scale_transform)
        image=image.unsqueeze(0).float().cuda()
        # Run CNN model
        print(file)
        img_ab_313=color_model(image)
        # Feed-Forward
        color_img = decode(image, img_ab_313, upscale=upscale)
        # Save image
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
