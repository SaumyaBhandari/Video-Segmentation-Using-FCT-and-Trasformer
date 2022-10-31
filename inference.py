import os
import random

import torch
from PIL import Image
from torchvision import transforms

from focusnet import FCT
from unet import UNet


class Inference():

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.network = self.__load_network()
        self.batch_size = 1


    def __load_network(self):
        if self.model_name == "focusnet":
            model = FCT()
            check = torch.load('saved_model/model_focusnet.tar')
            model.load_state_dict(check['model_state_dict'])
        else:
            model = UNet(in_channels=3, out_channels=1)
            check = torch.load('saved_model/model.tar')
            model.load_state_dict(check['model_state_dict'])
        model = model.to(self.device)
        return model


    def __view_sample(self, x, y, y_pred):
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        y_pred = torch.squeeze(y_pred)
        x = transforms.ToPILImage()(x)
        y = transforms.ToPILImage()(y)
        y_pred = transforms.ToPILImage()(y_pred)
        images = [x, y_pred, y]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height), (0, 200, 200))
        
        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0] + 4
        new_im.show()

    


    def infer(self, ind):

        images = os.listdir('segmented-images/images_test')
        masks = os.listdir('segmented-images/masks_test')
        trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        x = trans(Image.open(f'segmented-images/images_test/{images[ind]}'))
        y = trans(Image.open(f'segmented-images/masks_test/{masks[ind]}'))
        x, y = x[None, :], y[None, :]  # type: ignore
        x, y = x.to(self.device), y.to(self.device)

        model = self.network
        y_pred = model(x)
        self.__view_sample(x, y, y_pred)
    
model = "focusnet"
seg = Inference(model)
rand = random.randint(0, 100)
seg.infer(rand)