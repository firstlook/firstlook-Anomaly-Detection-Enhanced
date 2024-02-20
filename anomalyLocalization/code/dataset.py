# data loader 
import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F



class MVTecAD(data.Dataset):
    """Dataset class for the MVTecAD dataset."""

    def __init__(self, image_dir, transform):
        """Initialize and preprocess the MVTecAD dataset."""
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__