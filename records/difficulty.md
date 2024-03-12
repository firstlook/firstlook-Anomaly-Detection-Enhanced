
异常检测在图像领域困难点

1、维度灾难：图像维度高，传统机器学习领域方法无法有效应对维度灾难问题。

2、特征表征：图像特征包含较高得语义信息，在无监督信息下无法有效得提取，同时还得保持特征空间一致性。

3、理论困乏：目前针对图像领域，缺乏有效手段界定Anomaly Score。


思考点：
1、目前纯无监督学习，无法做到异常检测。参见[Anomaly Detection in Images](http://arxiv.org/pdf/1905.13147v1.pdf)


数据集：
![image](imgs/MVTec.png)
参见[MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf)

有效解决方案：