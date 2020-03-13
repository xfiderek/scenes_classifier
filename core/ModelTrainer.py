import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms 
import os

from .model import Model
from .IntelDataset import IntelDataset
from .transforms import training_transforms, eval_transforms


class ModelTrainer:
    def __init__(self, save_path, train_csv, valid_csv, data_dir="",
                  mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                  load_path="", disable_cuda=False):
        
        self.model = Model()
        self.device = torch.device("cuda:0" if (not disable_cuda and torch.cuda.is_available()) else "cpu")
        self.save_path = save_path

        self.loaded_epoch = 0 #if we load model, i.e. resume training it is updated to non-zero value
        self.mean = mean
        self.std = std 
        
        self.model = self.model.to(self.device)
        self.loss_crit = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0007)
                                                
        self.training_dataset = IntelDataset(data_dir, train_csv, self.mean, self.std, 
                                               transforms=training_transforms(self.mean, self.std))
        self.validation_dataset = IntelDataset(data_dir, train_csv, self.mean, self.std, 
                                                transforms=eval_transforms(self.mean, self.std)) if valid_csv else None 

        if load_path:
            self._load_model(load_path)

    def train(self, epochs, batch_size=64, print_loss_every:"num of batches" = 64, 
                    save_every:"num of epochs"=1, validate_every=1, shuffle_batch=True) -> None:
        
        dataloader_t = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=shuffle_batch,  num_workers=8,   pin_memory=True)

        if self.validation_dataset:
            dataloader_v = DataLoader(self.validation_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        accuracy = None
        for epoch in range(epochs):
            epoch += self.loaded_epoch
            self.model.train()
            for batch_num, (images, lbls) in enumerate(dataloader_t):
                images = images.to(self.device)
                lbls = lbls.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.loss_crit(outputs, lbls.reshape(-1))
                loss.backward()
                self.optimizer.step()

                if batch_num % print_loss_every==0:
                    current_loss = loss.item()
                    print("Epoch:{}, batch: {} loss: {}".format(epoch, batch_num, current_loss))
           
            if self.validation_dataset and epoch % validate_every == 0: 
                accuracy = self._validate(dataloader_v)
                print("epoch: {}, accuracy: {}%".format(epoch, round(accuracy, 2)))
                
            if epoch % save_every == 0:
                self._save_model(epoch, current_loss, accuracy)
        
    def _validate(self, dataloader):
        valid = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for _, (images, lbls) in enumerate(dataloader):
                images = images.to(self.device)
                lbls = lbls.to(self.device)

                outputs = self.model(images)
                valid += (torch.argmax(outputs).item() == lbls.item())
                total += 1
        percents = 100. * valid / total
        return percents


    def _save_model(self, epoch:int, loss:float, accuracy:"percents"):
        torch.save({
                    'epoch' : epoch,
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss' : loss,
                    'accuracy' : accuracy,
                    'mean' : self.mean,
                    'std' : self.std}, os.path.join(self.save_path, "weight-{}.pth".format(epoch)))
    
    def _load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path)['model_state_dict'])
        self.loaded_epoch = torch.load(load_path)['epoch']
        self.optimizer.load_state_dict(torch.load(load_path)['optimizer_state_dict'])
        self.mean = torch.load(load_path)['mean']
        self.std = torch.load(load_path)['std']
