
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