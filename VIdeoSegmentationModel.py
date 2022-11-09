from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from dataset import DataLoader
from EncoderDecoder_A2 import UNet
from EncoderDecoder_B import Decoder, DS_out, Encoder, EncoderDecoder_B


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
        self.encoder, self.transdec, self.decoder, self.dsOut = self.load_models()
        self.sequence_length = 8
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_models(self):
        encoder = Encoder()
        check = torch.load_state_dict('saved_model/road_EncoderDecoderB.tar')

        transDec = TransformerDec()
        decoder = Decoder(512, 16)
        dsOut = DS_out('mask', 16, 3)
        return encoder, transDec, decoder, dsOut


    def train(self):
        target = torch.rand(size=(1, 1, 512))
        flag = 1
        buffer = deque()
        mseloss 
        self.train_dataloader = DataLoader().load_data(self.batch_size)
        for epoch in range(epochs):
            for i, (x, y) in enumerate(tqdm(self.train_dataloader)):
                x = torch.rand(size=(3, 256, 256))
                # x, y = x.to(self.device), y.to(self.device)
                print(f"Input Frame {i+1}")
                out = encoder(x)
                buffer.append(out.tolist())
                if len(buffer) == sequence_length:
                    memory = torch.Tensor(buffer)
                    x = transDec(target, memory)
                    target = x
                    x = decoder(x)
                    buffer.pop()

model = SegmentationModel()
model.train()







