# -*- coding UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
# print(plt.get_backend())  # TkAgg


def clustering_task(data, target):
    # pca for plotting
    pcaModel = PCA(n_components=2)
    pca = pcaModel.fit_transform(data)

    # kmeans
    kmeansModel = KMeans(n_clusters=2)
    kmeans = kmeansModel.fit_predict(data)
    print(">>> Kmeans\nSilhouette Coefficient: {:.3f}".format(metrics.silhouette_score(data, kmeans)))
    print("Accuracy score: {:.3f}\n".format(metrics.accuracy_score(target, kmeans)))
    out = np.insert(pca, 2, values=kmeans, axis=1)
    plt.figure()
    plt.scatter(out[..., 0], out[..., 1], c=out[..., 2])
    # plt.show()
    plt.savefig('output/3.1.kmeans.jpg', dpi=600)

    # DBSCAN
    dbscanModel = DBSCAN(eps = 1, min_samples = 10)
    dbscan = dbscanModel.fit_predict(data)
    print(">>> DBSCAN\nSilhouette Coefficient: {:.3f}".format(metrics.silhouette_score(data, dbscan)))
    print("Accuracy score: {:.3f}\n".format(metrics.accuracy_score(target, dbscan)))
    out = np.insert(pca, 2, values=dbscan, axis=1)
    plt.figure()
    plt.scatter(out[..., 0], out[..., 1], c=out[..., 2])
    # plt.show()
    plt.savefig('output/3.1.dbscan.jpg', dpi=600)

    # test roc curve, only allow 2 classes
    fpr, tpr, thresholds = metrics.roc_curve(dbscan, kmeans)
    plt.figure()
    plt.plot(tpr, fpr)
    plt.savefig('output/3.1.roc-curve.pseudo.jpg', dpi=600)


def plot_three_group(x, y, file):
    # plot
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(x)):
        if y[i] == 0:
            red_x.append(x[i][0])
            red_y.append(x[i][1])
        elif y[i] == 1:
            blue_x.append(x[i][0])
            blue_y.append(x[i][1])
        else:
            green_x.append(x[i][0])
            green_y.append(x[i][1])
    plt.figure()
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.savefig(file, dpi=600)



def dimension_reduction_task(data, target):
    # pca for plotting
    pcaModel = PCA(n_components=2)
    pca = pcaModel.fit_transform(data)
    plot_three_group(pca, target, 'output/3.1.pca.jpg')
    # pca manually
    dataAdjust = data - data.mean(axis=0)  # 数据中心化
    covmatrix = np.cov(dataAdjust.T)  # 求协方差矩阵
    featValue, featVec = np.linalg.eig(covmatrix)  # 求特征值，特征向量
    index = np.argsort(-featValue)  # 按特征值从大到小排序
    k = 2 # 选择前两个主成分
    selectVec = featVec.T[index[:k]]  # 特征向量为列向量，转置后，选取特征值靠前的k行
    finalData = dataAdjust * np.matrix(selectVec.T)  # m * n, n* k => m * k
                                                     # 两个np.array相乘，属于element-wise
                                                     # 两个np.matrix相乘，属于矩阵乘法
    reconData = (finalData * selectVec) + data.mean(axis=0)  # 还原到原始空间
    print(">>> pca\n手动pca，与sklearn库算出来的总体差别：{:.3f}".format((finalData - pca).sum()))
    print("手动pca还原出的data，与原始data的总体差别：{:.3f}\n".format((reconData - data).sum()))
    # 参考 https://www.cnblogs.com/clnchanpin/p/7199713.html

    # NMF clustering
    nmf = NMF(n_components=3, init='random', random_state=0)
    W = nmf.fit_transform(data.T)  # basis matrix: m features * k groups; input matrix: m features * n samples
    H = nmf.components_  # coefficient matrix: k groups * n samples
    scaleH = H / H.sum(axis=0)  # 用列和将H矩阵scale到0-1之间，即每个样本，在k个groups中的weights，合起来等于1
    cluster = scaleH.argmax(axis=0)  # 将weight最大的那个group，设为sample对应的group
    plot_three_group(pca, 2-cluster, 'output/3.1.nmf.jpg')
    print(">>> nmf\nNMF分组统计：\n{}\n".format(pd.Series(cluster).value_counts()))
    # 参考 https://blog.csdn.net/acdreamers/article/details/44663421



if __name__ == '__main__':
    data, target = load_iris(return_X_y=True)
    clustering_task(data, target)
    dimension_reduction_task(data, target)


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
# NMF, 矩阵中所有元素均为非负数约束条件之下，
