

## 1. 马氏距离

#### 1.1 马氏距离等价于【规范化的主成分空间内的欧氏距离】
  
- **规范化的主成分空间**
  - 对数据集进行主成分分析，即对数据集的协方差矩阵进行特征值分解，求主成分（特征向量）
  - 对所有主成分进行归一化处理，这些规范化的主成分即构成了规范化主成分空间的坐标轴

- **将样本映射至规范化主成分空间，意味着数据从超椭圆(ellipsoidal)分布转化为超球面(spherical)分布**
  - 样本在规范化主成分空间各坐标轴上的投影(坐标分量)，可通过计算样本向量与规范化主成分的内积求得

- **两个向量的马氏距离等价于两者在规范化的主成分空间内的欧氏距离** 
  - If each of these axes is re-scaled to have unit variance, then the Mahalanobis distance corresponds to standard Euclidean distance in the transformed space. 


#### 1.2 马氏距离的特点
- **特点一：马氏距离是无单位化的、尺度无关的，它内生地考虑到了数据集各坐标轴之间的相关性**
  - The Mahalanobis distance is thus unitless and scale-invariant, and takes into account the correlations of the data set.
 
- **特点二：马氏距离与样本在各主成分上的偏离度成正比**
   - This distance is zero if P is at the mean of D, and grows as P moves away from the