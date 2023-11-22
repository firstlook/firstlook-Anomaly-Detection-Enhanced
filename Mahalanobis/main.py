
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
                    action='store_true')
parser.set_defaults(distort_targets=False)

# Dataset args
parser.add_argument('--dataset_name', type=str, default='forest_cover',
                    help='name of the dataset')
parser.add_argument('--test_prop', type=str, default=0.2)
parser.add_argument('--val_prop', type=str, default=0.2)

# Training args
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--no_adam',  dest='adam', action='store_false',
                    help='boolean whether to not use adam optimizer but SGD with momentum')
parser.set_defaults(