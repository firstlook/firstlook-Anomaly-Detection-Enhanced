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

        self.encoding_layers =