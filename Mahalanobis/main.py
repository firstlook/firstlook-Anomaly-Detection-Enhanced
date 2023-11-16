
import torch
import argparse

from modules.autoencoder import Autoencoder
from utils.dataloading import load_dataset
from utils.tracking import Tracker
from utils.experiment import train_model

parser = argparse.ArgumentParser(description='Automahalanobis experiment')

# Autoencoder args
parser.add_argument('--mahalanobis', dest='mahalanobis', action='store_true')
parser.set_defaults(mahalanobis=False)
parser.add_argument('--mahalanobis_cov_dec