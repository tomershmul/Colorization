import torch.nn as nn


def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)


class Color_model(nn.Module):
    def __init__(self, new_arch=False):
        super(Color_model, self).__init__()
        if not new_arch:
            self.features = nn.Sequential(
               # conv1
               nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 64),
               # conv2
               nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 128),
               # # conv3
               nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 256),
               # conv4
               nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
               nn.ReLU(),
               nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
               nn.ReLU(),
               nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 512),
               # conv5
               nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 256),
               # conv6
               nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, dilation = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
               nn.ReLU(),
               # conv6_313
               nn.Conv2d(in_channels = 256, out_channels = 313, kernel_size = 1, stride = 1,dilation = 1),

            )
        else:
            self.features = nn.Sequential(
               # conv1
               nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 64),
               # conv2
               nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1 , dilation = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 128),
               # # conv3
               nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 256),
               # conv4
               nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 4, dilation = 4),
               nn.ReLU(),
               nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 4, dilation = 4),
               nn.ReLU(),
               nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 4, dilation = 4),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 512),
               # conv5
               nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
               nn.ReLU(),
               nn.BatchNorm2d(num_features = 256),
               # conv6
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
               nn.ReLU(),
               nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
               nn.ReLU(),
               # conv6_313
               nn.Conv2d(in_channels = 256, out_channels = 313, kernel_size = 1, stride = 1,dilation = 1),
            )           
        self.apply(weights_init)

    def forward(self, gray_image):
        #print('gray_image',gray_image.size())
        features=self.features(gray_image)
        features=features/0.38
        return features
