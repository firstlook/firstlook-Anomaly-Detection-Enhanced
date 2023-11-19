
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
parser.add_argument('--mahalanobis_cov_decay', type=float, default=1E-4)
parser.add_argument('--distort_inputs', dest='distort_inputs',
                    action='store_true')
parser.set_defaults(distort_inputs=False)
parser.add_argument('--distort_targets', dest='distort_targets',
                    act