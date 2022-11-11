from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
from torch.autograd import variable
from torchvision import transforms

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
        # for parameters in encoderdecoder.parameters():
        #     parameters.requires_grad = False
        self.encoder = encoderdecoder.encoder
        self.decoder = encoderdecoder.decoder
        self.dsOutMask = encoderdecoder.dsMask
        self.dsOutImage = encoderdecoder.dsImage


    def save_sample(self, epoch, mask, mask_pred, x, frame):
        path = f'sneekpeeks_model_training/{epoch}'
        elements = [mask, mask_pred, x, frame]
        elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]
        elements[2] = elements[2].save(f"{path}_image.jpg")
        elements[3] = elements[3].save(f"{path}_image_pred.jpg")
        elements[0] = elements[0].save(f"{path}_mask.jpg")
        elements[1] = elements[1].save(f"{path}_mask_pred.jpg")



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
        crossentropy = torch.nn.CrossEntropyLoss()
        loss_train = []

        sequence_length = 4

        for epoch in range(epochs):

            _loss = 0
            print(f"Epoch no: {epoch+1}")

            for i, (image, mask) in enumerate(tqdm(train_dataloader)):

                mask_prev, lat = self.encoder(image)
                buffer.append(lat.tolist())

                if len(buffer) == sequence_length:
                    # if i == sequence_length+1:
                    #     target = sos
                    memory = torch.Tensor(buffer)
                    target = memory.detach().clone()
                    lat_das = model(target, memory)
                    latent = lat_das[0]
                    # latent = variable(latent.data, requires_grad=True)
                    x = self.decoder(latent)
                    mask_pred = self.dsOutMask(x)   
                    image_pred = self.dsOutImage(x)
                    buffer.pop()
                    # loss = mseloss(lat_das, lat)
                    loss1 = crossentropy(mask_pred, mask_prev)
                    loss2 = mseloss(image_pred, image)
                    loss = loss1 + loss2
                    _loss += loss.item()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    loss_train.append(_loss)
                    self.save_sample(epoch+1, mask_prev, mask_pred, image, image_pred)

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







