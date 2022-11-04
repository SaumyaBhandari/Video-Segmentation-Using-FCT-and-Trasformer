import random

import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from dataset import DataLoader
from EncoderDecoder_A1 import FCT
from EncoderDecoder_A2 import UNet
from EncoderDecoder_B import EncoderDecoder_B
from loss import MSE, DiceLoss


class Trainer():

    def __init__(self, prev_model) -> None:
        self.prev_model = prev_model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.enndec_A, self.enndec_B = self.load_network()
        for param in self.enndec_A.parameters():
            param.requires_grad = False
        self.batch_size = 4

    def load_network(self):
        if self.prev_model == "focusnet":
            model = FCT()
            check = torch.load('saved_model/model_focusnet.tar')
        else:
            modelA = UNet(in_channels=3, out_channels=1)
            modelB = EncoderDecoder_B(in_channels=4)
            check1 = torch.load('saved_model/model.tar')
            # check2 = torch.load('saved_model/model_encoderDecoderB.tar')
        modelA.load_state_dict(check1['model_state_dict'])
        # modelB.load_state_dict(check2['model_state_dict'])
        modelA, modelB = modelA.to(self.device), modelB.to(self.device)
        return modelA, modelB
    
    def save_sample(self, epoch, mask, mask_pred, x, frame):
        mask = mask[0:1, :, :, :]
        mask_pred = mask_pred[0:1, :, :, :]
        x = x[0:1, :, :, :]
        frame = frame[0:1, :, :, :]
        mask = torch.squeeze(mask)
        mask_pred = torch.squeeze(mask_pred)
        x = torch.squeeze(x)
        frame = torch.squeeze(frame)
        mask = transforms.ToPILImage()(mask)
        mask_pred = transforms.ToPILImage()(mask_pred)
        x = transforms.ToPILImage()(x)
        frame = transforms.ToPILImage()(frame)
        image = mask.save(f"sneakpeeks_EncoderDecoderB/{epoch}_mask.jpg")
        actual = mask_pred.save(f"sneakpeeks_EncoderDecoderB/{epoch}_mask_pred.jpg")
        x = x.save(f"sneakpeeks_EncoderDecoderB/{epoch}_x.jpg")
        x_pred = frame.save(f"sneakpeeks_EncoderDecoderB/{epoch}_x_pred.jpg")


    def train(self, epochs, lr=0.0001):

        print("Loading Datasets...")
        self.train_dataloader =DataLoader().load_data(self.batch_size)
        print("Dataset Loaded... initializing parameters...")
        encdec_A = self.enndec_A
        for param in encdec_A.parameters():
            param.requires_grad = False
        encdec_B = self.enndec_B
        optimizer = optim.AdamW(encdec_B.parameters(), lr)
        mseloss = torch.nn.MSELoss()

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
                with torch.no_grad():
                    mask = encdec_A(x)
                optimizer.zero_grad()
                X = torch.cat((x, mask), 1)
                mask_pred, frame, latent = encdec_B(X)
                loss1 = mseloss(mask_pred, mask)
                loss2 = mseloss(frame, x)
                loss = loss1+3*loss2
                _loss += loss.item()
                loss.backward()
                optimizer.step()
                if i == num:
                    self.save_sample(epoch+1, mask, mask_pred, x, frame)

            loss_train.append(_loss)

            print(f"Epoch: {epoch+1}, Training loss: {_loss}")

            if loss_train[-1] == min(loss_train):
                print('Saving Model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encdec_B.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train
                }, f'saved_model/model_encoderDecoderB.tar')
            print('\nProceeding to the next epoch...')
    

model = "unet"
seg = Trainer(model)
seg.train(epochs=60)