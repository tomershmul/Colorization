from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray, gray2rgb
from PIL import Image
import torch.utils.data as data
import torch
import numpy as np
import os

scale_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    # transforms.RandomCrop(224),
    #transforms.ToTensor()
])


class TrainImageFolder(data.Dataset):
    def __init__(self, data_dir, transform, model_output_size):
        self.file_list=os.listdir(data_dir)
        self.transform=transform
        self.data_dir=data_dir
        self.model_output_size = model_output_size

    def __getitem__(self, index):
        try:
            img=Image.open(self.data_dir+'/'+self.file_list[index])
            if self.transform is not None:
                # img_original = self.transform(img)
                # img_resize=transforms.Resize(self.model_output_size)(img_original)
                # img_original = np.asarray(img_original)

                img_resize=transforms.Resize(self.model_output_size)(img)
                img_original = np.asarray(img)

                #print('image ', self.file_list[index],img_original.shape)
                if img_resize.mode != 'RGB':
                    img_resize = img_resize.convert('RGB')
                img_lab = rgb2lab(img_resize)
                #img_lab = (img_lab + 128) / 255
                img_ab = img_lab[:, :, 1:3]
                img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
                #print('img_ori',img_original.shape)
                #print('img_ab',img_ab.size())
                if img_original.ndim != 3:
                    img_original = gray2rgb(img_original)
                img_original = rgb2lab(img_original)[:,:,0]-50.
                img_original = torch.from_numpy(img_original)
                return img_original, img_ab
        except Exception as ex:
            print(self.data_dir+'/'+self.file_list[index])
            print(str(ex))
            pass

    def __len__(self):
        return len(self.file_list)

