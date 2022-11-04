import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from EncoderDecoder_A2 import UNet
from EncoderDecoder_B import (DS_out, EncoderDecoder_B, dec_Block, enc_Block,
                              linear_bottleneck)


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.unet, self.enc_block, self.linear = self.load_model()

    def load_model(self):
        modelA = UNet(in_channels=3)
        modelB = enc_Block(in_channels=4)
        linear = linear_bottleneck(512*2*2, 512)
        # check1 = torch.load('saved_model/model.tar')
        # modelA.load_state_dict(check1['model_state_dict'])
        return modelA, modelB, linear

    def forward(self, x):
        x1 = self.unet(x)
        X = torch.cat((x, x1), dim=1)
        x = self.enc_block(X)
        x = x.view(-1, 512*2*2)
        x = self.linear(x)
        x = x.unsqueeze(dim=1)
        return x
    


class TransformerDec(nn.Module):
    def __init__(self):
        super(TransformerDec, self).__init__()
        self.decoderLayer = nn.TransformerDecoderLayer(512, nhead=1)
        self.decoder = nn.TransformerDecoder(self.decoderLayer, num_layers=2)

    def forward(self, target, memory):
        x = self.decoder(target, memory)
        return x



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = linear_bottleneck(512, 512*4)
        self.decBlock = dec_Block(512, 16)
        self.ds_mask = DS_out("mask", 16, 1)

    def forward(self, x):
        x = x.view(-1, 512)
        x = self.linear(x)
        x = x.view(x.size(0), 512, 2, 2)
        x = self.ds_mask(self.decBlock(x))
        return x





#initializing components
encoder = Encoder()
transDec = TransformerDec()
decoder = Decoder()

#initializing buffer and start of sequence for the transfomer decoder
sequence_length = 8
buffer = torch.ones((8, 1, 512))
# buffer2 = torch.ones((8, 1, 512))
sos = torch.rand(size=(1, 1, 512))

video_length = 1000

flag = 0
index_counter = 0
for i in range(video_length):
    input_frame = torch.rand(size=(1, 3, 256, 256))
    print(f"Input Frame {flag}")
    out = encoder(input_frame)
    buffer2 = buffer[1: , :, :]
    buffer = torch.cat((buffer2, out))
    x = transDec(sos, buffer)
    sos = x
    x = decoder(x)
    print(f"Decoder output {x.shape}")
    # print(buffer.shape)
    # print(f"Latent {flag} added...")
    flag += 1
        








