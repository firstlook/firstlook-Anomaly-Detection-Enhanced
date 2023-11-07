

import numpy as np
from numpy import linalg as LA


def mahal_dist(matrix):
    # 计算样本矩阵的中心向量
    matrix_mean = np.mean(matrix, axis=0)
    # 计算各样本与中心向量之间的差异
    delta = matrix - matrix_mean
    
    # 求协方差矩阵及其逆矩阵
    cov_matrix = np.cov(