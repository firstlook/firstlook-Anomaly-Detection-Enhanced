
import os
import torch
from torch.nn import functional as F
from dataset import return_MVTecAD_loader
from network import VAE,loss_function
import matplotlib.pyplot as plt

def eval(model,test_loader,device):
    model.eval()
    x_0 = iter(test_loader).next()
    with torch.no_grad():
        x_vae = model(x_0.to(device)).detach().cpu().numpy()


def EBM(model,test_loader,device):
    model.train()
    x_0 = iter(test_loader).next()
    alpha = 0.05
    lamda = 1
    x_0 = x_0.to(device).clone().detach().requires_grad_(True)
    recon_x = model(x_0).detach()
    loss = F.binary_cross_entropy(x_0, recon_x, reduction='sum')  
    loss.backward(retain_graph=True)