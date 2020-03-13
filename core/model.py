import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) #64x64
        self.conv2 = nn.Conv2d(64, 256, 3, padding=1) #64x64

        self.conv3 = nn.Conv2d(256, 128, 3, padding=1) #64x64, then pooling - 32x32
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1) #32x32, then pooling - 16x16
        self.conv5 = nn.Conv2d(64, 32, 1) #reducing filters, 16x16

        self.fc1 = nn.Linear(32*16*16, 2048)
        self.fc2 = nn.Linear(2048, 6)

        self.batch_norm_c1 = nn.BatchNorm2d(64)
        self.batch_norm_c2 = nn.BatchNorm2d(256)
        self.batch_norm_c3 = nn.BatchNorm2d(128)
        self.batch_norm_c4 = nn.BatchNorm2d(64)
        self.batch_norm_c5 = nn.BatchNorm2d(32)
        
        self.plain_dropout = nn.Dropout(p=0.5)
        self.conv_dropout = nn.Dropout(p=0.3)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm_c1(x)
        x = self.conv_dropout(x)

        x = F.relu(self.conv2(x))
        x = self.batch_norm_c2(x)
        x = self.conv_dropout(x)
        
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.batch_norm_c3(x)
        x = self.conv_dropout(x)
        
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = self.batch_norm_c4(x)
        x = self.conv_dropout(x)
        
        x = self.conv5(x)
        x = self.batch_norm_c5(x)
        x = self.conv_dropout(x)

        x = x.view(-1, 32*16*16)
        x = self.plain_dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
