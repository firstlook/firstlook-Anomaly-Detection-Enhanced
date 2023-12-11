# -*- coding: utf-8 -*-
"""
Autoencoder module
--------------------------
"""
import torch
import torch.nn as nn
from modules.mahalanobis import MahalanobisLayer

class Autoencoder(nn.Module):

    def __init__(self, layer_dims, mahalanobis=False,
                 mahalanobis_cov_decay=0.1, distort_inputs=False):
        super(Autoencoder, self).__init__()

        self.layer_dims = layer_dims

        self.encoding_layers = torch.nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]),  # 1st hidden layer
            nn.Tanh(),                                # 1st hidden layer
            nn.Linear(layer_dims[1], layer_dims[2])   # Compression layer
        )

        self.decoding_layers = torch.nn.Sequential(
            nn.Linear(layer_dims[2], layer_dims[3]),  # 3rd hidden layer
  