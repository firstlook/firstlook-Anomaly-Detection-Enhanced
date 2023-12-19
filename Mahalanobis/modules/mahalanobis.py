# -*- coding: utf-8 -*-
"""
Mahalanobis module
--------------------------
"""
import torch
import torch.nn as nn

class MahalanobisLayer(nn.Module):

    def __init__(self, dim, decay = 0.1)