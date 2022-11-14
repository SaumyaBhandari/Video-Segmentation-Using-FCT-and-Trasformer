import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

from dataset import DataLoader
from EncoderDecoder_A2 import UNet
from loss import DiceLoss, CrossEntropy2d


#encoder block
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 4
        self.unet = self.load_prev_encoder()
        for param in self.unet.parameters():
            param.requires_grad = False
        self.conv1 = nn.Conv2d(6, in_channels*2, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels*2, in_channels*4, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels*4, in_channels*8, 3, 1, padding="same")
        self.conv4 = nn.Conv2d(in_channels*8, in_channels*16, 3, 1, padding="same")
        self.conv5 = nn.Conv2d(in_channels*16, in_channels*32, 3, 1, padding="same")
        # self.conv6 = nn.Conv2d(in_channels*32, in_channels*64, 3, 1, padding="same")
        # self.conv7 = nn.Conv2d(in_channels*64, in_channels*128, 3, 1, padding="same")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) 
        self.linear1 = linear_bottleneck(8192, 4096)
        self.linear2 = linear_bottleneck(4096, 1024)

    def load_prev_encoder(self):
        model = UNet(in_channels=3, out_channels=3)
        check = torch.load('saved_model/road_model_unet.tar')
        model.load_state_dict(check['model_state_dict'])
        return model

    def forward(self, x):
        x1 = self.unet(x)
        x = torch.cat((x, x1), dim=1)
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = self.maxpool(self.relu(self.conv4(x)))
        x = self.maxpool(self.relu(self.conv5(x)))
        # x = self.maxpool(self.relu(self.conv6(x)))
        # x = self.maxpool(self.relu(self.conv7(x)))
        x = x.view(-1, 128*8*8)
        x = self.linear1(x)
        x = self.linear2(x)
        return x1, x


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
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear1 = linear_bottleneck(1024, 4096)
        self.linear2 = linear_bottleneck(4096, 8192)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(in_channels//2, in_channels//4, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.ConvTranspose2d(in_channels//4, in_channels//8, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.ConvTranspose2d(in_channels//8, in_channels//16, kernel_size=2, stride=2, padding=0)
        # self.conv5 = nn.ConvTranspose2d(in_channels//16, out_channels, kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        # x = self.relu(self.conv5(x))
        out = self.dropout(x)
        return out


#deep supervision class 
class DS_out(nn.Module):
    def __init__(self, output, in_channels, out_channels):
        super().__init__()
        self.output = output
        self.conv1 = nn.ConvTranspose2d(in_channels, 3, kernel_size=2, stride=2)
        # self.conv2 = nn.ConvTranspose2d(in_channels//2, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        # if self.output == 'mask':
        #     # out = self.softmax(self.conv2(x))
        #     out = self.relu(self.conv2(x))
        #     # out = self.sigmoid(self.conv2(x))
        # else:
        #     out = self.relu(self.conv2(x))
        return out



#Fully convolutional transformer
class EncoderDecoder_B(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [8, 16, 32, 64, 128, 256, 512] 
        self.encoder = Encoder()
        self.decoder = Decoder(128, 8)
        self.dsMask = DS_out("mask", 8, 3)
        self.dsImage = DS_out("image", 8, 3)
        self.batch_size = 1

    def forward(self, x):
        mask_prev, latent = self.encoder(x)
        decoded = self.decoder(latent)
        image = self.dsImage(decoded)
        mask = self.dsMask(decoded)
        return mask_prev, latent, mask, image





def save_sample(epoch, mask, mask_pred, x, frame):
    path = f'sneakpeeks_EncoderDecoderB/{epoch}'
    elements = [mask, mask_pred, x, frame]
    elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]
    elements[2] = elements[2].save(f"{path}_image.jpg")
    elements[3] = elements[3].save(f"{path}_image_pred.jpg")
    elements[0] = elements[0].save(f"{path}_mask.jpg")
    elements[1] = elements[1].save(f"{path}_mask_pred.jpg")



def train(epochs, batch_size=2, lr=0.0001):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    print("Loading Datasets...")
    train_dataloader = DataLoader().load_data(batch_size)
    print("Dataset Loaded.")

    print("Initializing Parameters...")
    model = EncoderDecoder_B()
    optimizer = optim.AdamW(model.parameters(), lr)
    mseloss = torch.nn.MSELoss()
    # crossentloss = torch.nn.CrossEntropyLoss()
    # diceloss = DiceLoss()
    loss_train = []
    start = 0
    epochs = epochs
    print(f"Parameters Initialized...")

    print(f"Starting to train for {epochs} epochs.")
    for epoch in range(start, epochs):
        print(f"Epoch no: {epoch+1}")
        _loss = 0
        num = random.randint(0, 9)
        for i, (image, mask) in enumerate(tqdm(train_dataloader)):
            mask_prev, latent, mask_pred, image_pred = model(image)
            loss1 = mseloss(mask_pred, mask_prev)
            loss2 = mseloss(image_pred, image)
            loss = loss1 + 2*loss2
            _loss += loss.item()
            loss.backward()
            optimizer.step()
            # if i == num:
            save_sample(epoch+1, mask_prev, mask_pred, image, image_pred)
        loss_train.append(_loss)
        print(f"Epoch: {epoch+1}, Training loss: {_loss}")
        if epoch%50==0:
            print('Saving Model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train
            }, f'saved_model/road_EncoderDecoderB_{epoch}.tar')
        print('\nProceeding to the next epoch...')



train(epochs=1000)
# data = (torch.rand(size=(1, 3, 256, 256)))
# model = UNet(out_channels=3)
# out = model(data)
# print(out.shape)
    


