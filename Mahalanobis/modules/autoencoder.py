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
            nn.Tanh(),                                # 3d hidden layer
            nn.Linear(layer_dims[3], layer_dims[4])   # Output layer
        )

        self.mahalanobis = mahalanobis

        if mahalanobis:
            self.mahalanobis_layer = MahalanobisLayer(layer_dims[0],
                                                      mahalanobis_cov_decay)

        self.distort_input = distort_inputs

    def forward(self, x):
        x_in = x + torch.randn_like(x) if self.distort_input else x
        x_enc = self.encoding_layers(x_in)
        x_fit = self.