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


class Linear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linear = nn.Linear(input, output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        return x



class EncoderBlock(nn.Module):
    def __init__(self, blk, in_channels, out_channels):
        super().__init__()
        self.blk = blk
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) 

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))
        if not self.blk == "first":
            x1 = self.relu(self.conv3(x1))
        out = self.maxpool(self.dropout(x1))
        return out


    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.relu(self.conv1(x1))
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv3(x1))
        out = self.dropout(x1)
        return out



class DS_out(nn.Module):
    def __init__(self, output, in_channels, out_channels):
        super().__init__()
        self.output = output
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.relu(self.conv1(x1))
        x1 = self.relu(self.conv2(x1))
        out = self.relu(self.conv3(x1))
        return out




class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.unet = UNet(in_channels=3, out_channels=3)
        check = torch.load('saved_model/road_model_unet.tar')
        self.unet.load_state_dict(check['model_state_dict'])
        for param in self.unet.parameters():
            param.requires_grad = False
        self.block_1 = EncoderBlock("first", 3, 8)
        self.block_2 = EncoderBlock("second", 8, 16)
        self.block_3 = EncoderBlock("third", 16, 32)
        self.block_4 = EncoderBlock("fourth", 32, 64)
        self.block_5 = EncoderBlock("bottleneck", 64, 128)
        self.linear_bottleneck1 = Linear(8192, 4096)
        self.linear_bottleneck2 = Linear(4096, 1024)

    def forward(self, x):
        x1 = self.unet(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = x.view(-1, 128*8*8)
        x = self.linear_bottleneck1(x)
        x = self.linear_bottleneck2(x)
        return x1, x




class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(1024, 4096)
        self.linear2 = Linear(4096, 8192)
        self.block_1 = DecoderBlock(128, 64)
        self.block_2 = DecoderBlock(64, 32)
        self.block_3 = DecoderBlock(32, 16)
        self.block_4 = DecoderBlock(16, 8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.relu(self.block_1(x))
        x = self.relu(self.block_2(x))
        x = self.relu(self.block_3(x))
        x = self.relu(self.block_4(x))
        out = self.dropout(x)
        return out




class EncoderDecoder_B2(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.dsMask = DS_out("mask", 8, 3)
        self.dsImage = DS_out("image", 8, 3)
        
    def forward(self, x):
        mask_prev, latent = self.encoder(x)
        x = self.decoder(latent)
        mask_pred = self.dsMask(x)
        x_pred = self.dsImage(x)
        return mask_prev, latent, mask_pred, x_pred




def save_sample(epoch, mask, mask_pred, x, frame):
    path = f'Training Sneakpeeks/sneakpeeks_EncoderDecoderB/{epoch}'
    elements = [mask, mask_pred, x, frame]
    elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]
    elements[2] = elements[2].save(f"{path}_image.jpg")
    elements[3] = elements[3].save(f"{path}_image_pred.jpg")
    elements[0] = elements[0].save(f"{path}_mask.jpg")
    elements[1] = elements[1].save(f"{path}_mask_pred.jpg")



def train(epochs, batch_size=1, lr=0.0001):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    print("Loading Datasets...")
    train_dataloader = DataLoader().load_data(batch_size)
    print("Dataset Loaded.")

    print("Initializing Parameters...")
    model = EncoderDecoder_B2()
    optimizer = optim.AdamW(model.parameters(), lr)
    mseloss = torch.nn.MSELoss()
    crossentloss = torch.nn.CrossEntropyLoss()
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
            loss2 = crossentloss(image_pred, image)
            loss = loss1 + loss2
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
            }, f'saved_model/road_EncoderDecoderB2_{epoch}.tar')
        print('\nProceeding to the next epoch...')


train(30)