import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

# from unet import UNet


#encoder block
class enc_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels*2, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels*2, in_channels*4, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels*4, in_channels*8, 3, 1, padding="same")
        self.conv4 = nn.Conv2d(in_channels*8, in_channels*16, 3, 1, padding="same")
        self.conv5 = nn.Conv2d(in_channels*16, in_channels*32, 3, 1, padding="same")
        self.conv6 = nn.Conv2d(in_channels*32, in_channels*64, 3, 1, padding="same")
        self.conv7 = nn.Conv2d(in_channels*64, in_channels*128, 3, 1, padding="same")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) 
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = self.maxpool(self.relu(self.conv4(x)))
        x = self.maxpool(self.relu(self.conv5(x)))
        x = self.maxpool(self.relu(self.conv6(x)))
        x = self.maxpool(self.relu(self.conv7(x)))
        return x


#Linear bottleneck Layer
class linear_bottleneck(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linear = nn.Linear(input, output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        return x


#decoder block
class dec_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(in_channels//2, in_channels//4, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.ConvTranspose2d(in_channels//4, in_channels//8, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.ConvTranspose2d(in_channels//8, in_channels//16, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.ConvTranspose2d(in_channels//16, out_channels, kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        out = self.dropout(x)
        return out


#deep supervision class 
class DS_out(nn.Module):
    def __init__(self, output, in_channels, out_channels):
        super().__init__()
        self.output = output
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(in_channels//2, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        if self.output == 'mask':
            out = self.relu(self.conv2(x))
        else:
            out = self.relu(self.conv2(x))
        return out



#Fully convolutional transformer
class EncoderDecoder_B(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        filters = [8, 16, 32, 64, 128, 256, 512] 

        self.enc_CNN = enc_Block(in_channels)
        self.dec_CNN = dec_Block(filters[-1], filters[-2])
        self.linear1 = linear_bottleneck(input=2*2*filters[6], output=512)
        self.linear2 = linear_bottleneck(input=512, output=2*2*filters[6])
        self.dsMask = DS_out("mask", filters[1], 1)
        self.dsImage = DS_out("image", filters[1], 3)
        # self.decoderLayer = nn.TransformerDecoderLayer(filters[-1], nhead=1)
        # self.decoder = nn.TransformerDecoder(self.decoderLayer, num_layers=2)
        # self.prev = torch.randn(1, 1, filters[-1])

    def forward(self, x):
        x = self.enc_CNN(x)
        x =  x.view(-1, 512*2*2)
        l = self.linear1(x)
        # l = l.unsqueeze(dim=1)
        # l_dash = self.decoder(self.prev, l)
        # l_dash = l_dash.view(-1, 512)
        # self.prev = l_dash
        # x = self.linear2(l_dash)
        x = self.linear2(l)
        x = x.view(x.size(0), 512, 2, 2)
        x = self.dec_CNN(x)
        outM = self.dsMask(x)
        outF = self.dsImage(x)

        return outM, outF, l


# data = (torch.rand(size=(1, 4, 256, 256)))
# model = EncoderDecoder_B(in_channels=4)
# out = model(data)
# print(out[0].shape)
# print(out[1].shape)
# print(out[2].shape)
