# -*- coding UTF-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt



def clustering_task(data, target):
    # kmeans
    estimator = KMeans(n_clusters=2)
    kmeans = estimator.fit_predict(data)
    # pca
    estimator2 = PCA(n_components=2)
    pca = estimator2.fit_transform(data)
    # plot
    out = np.insert(pca, 2, values=kmeans, axis=1)
    plt.figure()
    plt.scatter(out[..., 0], out[..., 1], c=out[..., 2])
    # plt.show()
    plt.savefig('output/3.1.kmeans.jpg', dpi=600)


    # test github



def dimension_reduction_task():
    pass



if __name__ == '__main__':
    data, target = load_iris(return_X_y=True)
    clustering_task(data, target)


# 距离的度量
# 欧氏距离，平方和，是直线距离，点间距离
# 曼哈顿距离，差的绝对值之和，是十字路口走过距离，方形距离
# 马氏距离，尺度无关，协方差距离，各个属性标准化，再计算距离，弧形距离
# 余弦相似度距离，向量夹角的余弦值，衡量两者的距离，夹角越接近0，余弦值越接近1，说明两个向量越相似，方向距离

# 输入矩阵
# 1. 样本个数*特征个数定义的矩阵形式
# 2. 样本个数*样本个数定义的矩阵形式，样本相似度矩阵，[0, 1]取值，对角线元素全为1

# 聚类，sklearn.cluster
# K-means，点间距离，参数：聚类个数
# DBSCAN，点间距离，参数：邻域大小
# Gaussian Mixtures，马氏距离，参数，聚类个数及其他超参
# Birch，欧式距离，参数，分支因子，阈值等其他超参

# 降维，sklearn.decomposition
# PCA，信号处理，参数：所降维度及其他超参
# FastICA，图形图像特征提取，适用于超大规模数据，参数：所降维度及其他超参
# NMF，图形图像特征提取，参数：所降维度及其他超参
# LDA，文本数据，主题挖掘，参数：所降维度及其他超参

