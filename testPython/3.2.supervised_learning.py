#! /usr/bin/env python
# -*- coding UTF-8 -*-
# @ author: Yang Wu
# Email: wuyang.bnu@139.com

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def use_kNN(data, target):
    # test_size，指定测试集比例；stratify，保证划分后，两个集合的各类别样本比例一致；random_state，保证可重复性
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, stratify=target, test_size=.5, random_state=42)
    # k nearest neighbor model
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, Y_train)  # 拿训练集
    Y_predict = knn.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, Y_predict)

    # plot
    pca = PCA(n_components=2)
    X_2dim = pca.fit_transform(X_test)
    plt.subplot(2, 2, 1)
    plt.title("kNN acc={:.2f}, by prediction".format(accuracy))
    plt.scatter(X_2dim[:,0], X_2dim[:,1], c=Y_predict)
    plt.subplot(2, 2, 2)
    plt.title("kNN acc={:.2f}, by label".format(accuracy))
    plt.scatter(X_2dim[:,0], X_2dim[:,1], c=Y_test)
    plt.show()



if __name__ == '__main__':
    plt.figure()
    data, target = load_iris(return_X_y=True)
    use_kNN(data, target)




# 监督学习
# 利用一组带有标签的数据，学习从输入到输出的映射，然后将这种映射关系，应用到未知数据上
# 分类，输出是离散的，训练数据是观察和评估，标签表明类别，根据训练数据学习出分类器，新数据输入给学好的分类器进行判断
#       训练集（已标注数据，用于建模），例如，从已标注数据中，随机选70%，做为训练集
#       测试集（已标注数据，标注隐藏，模型预测后进行对比，评估模型的学习能力），例如，已标注数据中，训练集外，为测试集
#       验证集（？）
# 回归，输出是连续的，了解两个或多个变数间是否相关、研究其相关方向及强度，可由给出的自变量，估计因变量的条件期望

# 评价标准
# 准确率，accuracy，(TP + TN) / (TP + FP + TN + FN)，针对正负两类
# 精确率，precision，TP / (TP + FP)，针对正类，在预测集里的比例
# 召回率，recall，TP / (TP + FN)，针对正类，在正样本集的比例

# 分类模型
# kNN，lazy learning，没有训练过程，把数据集加载到内存中就可以分类了，对未知点，先k个最近点投票
# locally weighted regression (LWR)，对未知样本，在X轴上以它为中心，左右各找几个点，合起来进行线性回归，据此预测
# naiveBayes
# SVM
# decision tree
# neural network

# 回归模型，适合时序数据
# sklearn.linear_model，线性函数
#       Linear Regression
#       Ridge
#       Lasso
# sklearn.preprocessing
#       多项式回归



