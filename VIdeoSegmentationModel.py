from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm

from dataset import DataLoader
from EncoderDecoder_A2 import UNet
from EncoderDecoder_B import Decoder, DS_out, Encoder, EncoderDecoder_B
from loss import DiceLoss


class TransformerDec(nn.Module):
    def __init__(self):
        super(TransformerDec, self).__init__()
        self.decoderLayer = nn.TransformerDecoderLayer(512, nhead=1)
        self.decoder = nn.TransformerDecoder(self.decoderLayer, num_layers=2)

    def forward(self, target, memory):
        x = self.decoder(target, memory)
        return x



class SegmentationModel():

    def __init__(self):
        self.transdec = TransformerDec()
        self.sequence_length = 8
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        encoderdecoder = EncoderDecoder_B()
        encoderdecoder.load_state_dict(torch.load('saved_model/road_EncoderDecoderB.tar')['model_state_dict'])
        for parameters in encoderdecoder.parameters():
            parameters.requires_grad = False
        self.encoder = encoderdecoder.encoder
        self.decoder = encoderdecoder.decoder
        self.dsOutMask = encoderdecoder.dsMask
        self.dsOutImage = encoderdecoder.dsImage


    def train(self, epochs, batch_size=1, lr=0.0001):
        target = torch.rand(size=(1, 1, 512))
        flag = 1
        buffer = deque()
        # targ = deque()
        # targ.append(target.tolist())
        train_dataloader = DataLoader().load_data(batch_size)
        model = self.transdec
        optimizer = optim.AdamW(model.parameters(), lr)
        mseloss = torch.nn.MSELoss()
        diceloss = DiceLoss()
        loss_train = []

        sequence_length = 4

        for epoch in range(epochs):

            _loss = 0
            print(f"Epoch no: {epoch+1}")

            for i, (image, mask) in enumerate(tqdm(train_dataloader)):

                mask, lat = self.encoder(image)
                buffer.append(lat.tolist())

                if len(buffer) == sequence_length:
                    # if i == sequence_length+1:
                    #     target = sos
                    memory = torch.Tensor(buffer)
                    lat_das = model(target, memory)
                    target = lat_das.detach().clone()
                    x = self.decoder(lat_das)
                    mask_pred = self.dsOutMask(x)   
                    image_pred = self.dsOutImage(x)
                    buffer.pop()

                    # loss = mseloss(lat_das, lat)
                    loss1 = diceloss(mask_pred, mask)
                    loss2 = mseloss(image_pred, image)
                    loss = loss1 + loss2
                    _loss += loss.item()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    loss_train.append(_loss)

            print(f"Epoch: {epoch+1}, Training loss: {_loss}")
            if loss_train[-1] == min(loss_train):
                print('Saving Model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train
                }, f'saved_model/transformer_model.tar')
            print('\nProceeding to the next epoch...')
                


model = SegmentationModel()
model.train(epochs=10)







