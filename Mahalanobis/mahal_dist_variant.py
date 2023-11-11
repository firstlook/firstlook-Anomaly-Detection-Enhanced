

import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler 


def mahal_dist_variant(matrix):
    # 将数据集标准化
    matrix = StandardScaler().fit_transform(matrix)
    # 对数据集进行主成分分析
    cov_matrix = np.cov(matrix, rowvar=False, ddof=1)
    eigen_values, eigen_vectors = LA.eig(cov_matrix)
        
    # 函数get_score用于返回数据集在单个主成分上的分数
    # 参数pc_i