import torch
from torch.autograd import Variable
from skimage.color import lab2rgb
from model import Color_model
#from data_loader import ValImageFolder
import numpy as np
from skimage.color import rgb2lab, rgb2gray
import torch.nn as nn 
from PIL import Image
import scipy.misc
from torchvision import datasets, transforms
from training_layers import decode
import torch.nn.functional as F
import os

scale_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    # transforms.RandomCrop(224),
])


def load_image(image_path, model_output_size, transform=None,):
    image = Image.open(image_path)
    
    if transform is not None:
        image = transform(image)
    image_small=transforms.Resize(model_output_size)(image)
    image_small=np.expand_dims(rgb2lab(image_small)[:,:,0],axis=-1)
    image=rgb2lab(image)[:,:,0]-50.
    image=torch.from_numpy(image).unsqueeze(0)
    
    return image,image_small

def main():
    # CIFAR10 vs ImageNet changes
    work_dataset = "CIFAR10"
    if work_dataset == "CIFAR10":
        model_output_size = 32 # TODO 32 CIFAR, 56 in ImageNet
        upscale = 2
    else:
        model_output_size = 56 # TODO 32 CIFAR, 56 in ImageNet
        upscale = 4


    data_dir = "../data/val"
    dirs=os.listdir(data_dir)
    color_model = nn.DataParallel(Color_model()).cuda().eval()
    color_model.load_state_dict(torch.load('../model/models/model-10-782.ckpt'))
     
    for file in dirs:
        image,image_small=load_image(data_dir+'/'+file, model_output_size=model_output_size, transform=scale_transform) #TODO CIFAR 32, imagenet 56
        image=image.unsqueeze(0).float().cuda()
        img_ab_313=color_model(image)
        # out_max=np.argmax(img_ab_313[0].cpu().data.numpy(),axis=0)
        # print('out_max',set(out_max.flatten()))
        color_img = decode(image, img_ab_313, upscale=upscale)
        #print(color_img)
        #break
        color_name = '../data/colorimg/' + file
        scipy.misc.imsave(color_name, color_img*255.)

if __name__ == '__main__':
    main()
