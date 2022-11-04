import random

import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from dataset import DataLoader
from EncoderDecoder_A1 import FCT
from EncoderDecoder_A2 import UNet
from loss import DiceLoss


class Trainer():

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.network = self.load_network()
        self.batch_size = 1

    def load_network(self):
        if self.model_name == "focusnet":
            model = FCT()
            check = torch.load('saved_model/model_focusnet.tar')
        else:
            model = UNet(in_channels=3, out_channels=1)
            check = torch.load('saved_model/model.tar')
        model.load_state_dict(check['model_state_dict'])
        model = model.to(self.device)
        return model
    
    def save_sample(self, epoch, x, y, y_pred):
        x = x[0:1, :, :, :]
        y = y[0:1, :, :, :]
        y_pred = y_pred[0:1, :, :, :]
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        y_pred = torch.squeeze(y_pred)
        x = transforms.ToPILImage()(x)
        y = transforms.ToPILImage()(y)
        y_pred = transforms.ToPILImage()(y_pred)

        image = x.save(f"sneakpeeks_focusnet/{epoch}_input.jpg")
        actual = y.save(f"sneakpeeks_focusnet/{epoch}_actual.jpg")
        pred = y_pred.save(f"sneakpeeks_focusnet/{epoch}_predicted.jpg")


    def train(self, epochs, lr=0.0001):

        print("Loading Datasets...")
        self.train_dataloader =DataLoader().load_data(self.batch_size)
        print("Dataset Loaded... initializing parameters...")
        model = self.network
        optimizer = optim.AdamW(model.parameters(), lr)
        dsc_loss = DiceLoss()

        loss_train = []
        start = 0
        epochs = epochs
        print(f"Starting to train for {epochs} epochs.")
        for epoch in range(start, epochs):
            print(f"Epoch no: {epoch+1}")
            _loss = 0
            num = random.randint(0, 100)
            for i, (x, y) in enumerate(tqdm(self.train_dataloader)):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = dsc_loss(y_pred, y)
                _loss += loss.item()
                loss.backward()
                optimizer.step()
                if i == num:
                    self.save_sample(epoch+1, x, y, y_pred)

            loss_train.append(_loss)

            print(f"Epoch: {epoch+1}, Training loss: {_loss}")

            if loss_train[-1] == min(loss_train):
                print('Saving Model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train
                }, f'saved_model/model_{self.model_name}.tar')
            print('\nProceeding to the next epoch...')
    

model = "focusnet"
seg = Trainer(model)
seg.train(epochs=60)