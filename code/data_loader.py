from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
from skimage import io
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

scale_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    #transforms.ToTensor()
])


class TrainImageFolder(data.Dataset):
    def __init__(self, data_dir, transform, model_output_size):
        self.file_list=os.listdir(data_dir)
        self.transform=transform
        self.data_dir=data_dir
        self.model_output_size = model_output_size
    def __getitem__(self, index):
        #try:
            img=Image.open(self.data_dir+'/'+self.file_list[index])
            if (img.layers == 3):
                if self.transform is not None:
                    img_original = self.transform(img)
                    img_resize=transforms.Resize(self.model_output_size)(img_original)
                    img_original = np.asarray(img_original)
                    #print('image ', self.file_list[index],img_original.shape)
                    img_lab = rgb2lab(img_resize)
                    #img_lab = (img_lab + 128) / 255
                    img_ab = img_lab[:, :, 1:3]
                    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
                    #print('img_ori',img_original.shape)
                    #print('img_ab',img_ab.size())
                    img_original = rgb2lab(img_original)[:,:,0]-50.
                    img_original = torch.from_numpy(img_original)
                    return img_original, img_ab
        #except:
        #    pass
    def __len__(self):
        return len(self.file_list)


class ValImageFolder(data.Dataset):
    def __init__(self,data_dir):
        self.file_list=os.listdir(data_dir)
        self.data_dir=data_dir

    def __getitem__(self, index):
        img=Image.open(self.data_dir+'/'+self.file_list[index])
        img_scale = scale_transform(img)
        img_scale = np.asarray(img_scale)
        img_scale = rgb2gray(img_scale)
        img_scale = torch.from_numpy(img_scale)
        return img_scale

    def __len__(self):
        return len(self.file_list)
