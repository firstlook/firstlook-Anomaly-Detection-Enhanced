
import os
import torch
from torch.nn import functional as F
from dataset import return_MVTecAD_loader
from network import VAE,loss_function
import matplotlib.pyplot as plt

def train(model,train_loader,device,optimizer,epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data, model.mu, model.logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    return train_loss


def eval(model,test_loader,device):
    model.eval()
    x_0 = iter(test_loader).next()
    with torch.no_grad():
        x_vae = model(x_0.to(device)).detach().cpu().numpy()

