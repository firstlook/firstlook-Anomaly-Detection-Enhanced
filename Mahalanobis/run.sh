#!/usr/bin/env bash

source activate pytorch

# Forest cover
python3 main.py --dataset_name forest_cover
python3 main.py --mahalanobis --dataset_name forest_cover
python3 main.py --mahalanobis --distort_inputs --dataset_name forest_cover
python3 main