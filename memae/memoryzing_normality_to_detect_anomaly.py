
"""
Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection
https://arxiv.org/pdf/1904.02639.pdf

#https://github.com/VieVie31/cool-papers-in-pytorch/blob/master/memoryzing_normality_to_detect_anomaly.py
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch as T
import torch.nn as nn

from torch.nn.modules import *

from tqdm import tqdm, trange
from torchvision import datasets, transforms

from sklearn.metrics import f1_score, accuracy_score


T.set_default_tensor_type('torch.FloatTensor')

batch_size = 32
nb_epochs  = 1
nb_digits  = 10


train_normals = [
    img for img, lbl in datasets.MNIST(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Normalize((0.1307,), (0.3081,))
        ])
    ) if lbl == 9
]
train_normals = torch.utils.data.TensorDataset(
    torch.tensor([v.numpy() for v in train_normals])
)
train_normals_loader = T.utils.data.DataLoader(
    train_normals, 
    batch_size=batch_size, 
    shuffle=True
) 


train_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=True
)
 
test_loader = T.utils.data.DataLoader(datasets.MNIST(
    './data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=False
) 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  16, 1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),