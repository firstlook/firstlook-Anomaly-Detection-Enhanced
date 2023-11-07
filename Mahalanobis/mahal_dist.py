

import numpy as np
from numpy import linalg as LA


def mahal_dist(matrix):
    # 计算样本矩阵的中心向量
    matrix_mean = np.mean(matrix, axis=