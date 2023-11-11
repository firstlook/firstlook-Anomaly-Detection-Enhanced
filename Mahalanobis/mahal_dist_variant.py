

import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler 


def mahal_dist_variant(matrix):
    # 将数据集标准化
    matrix = StandardScaler().fit_transform(matrix)
    # 对数据集进行主成分分析
    cov_matrix = np.cov(matri