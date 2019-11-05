#! /usr/bin/env python
# -*- coding UTF-8 -*-
# @ author: Yang Wu
# Email: wuyang.bnu@139.com

import numpy as np
import pandas as pd
from os import listdir
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import Imputer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def load_databset(feature_paths, label_paths):
    feature = np.ndarray(shape=(0, 41))
    label = np.ndarray(shape=(0, 1))
    for file in feature_paths:
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))
    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))
    label = np.ravel(label)
    return(feature, label)


def img2vector(filename):
    retMat = np.zeros([1024], int)
    with open(filename) as fo:
        lines = fo.readlines()
        for i in range(32):
            for j in range(32):
                retMat[i*32+j] = lines[i][j]
    return(retMat)


def readDataset(path):
    filelist = listdir(path)
    numFiles = len(filelist)
    data = np.zeros([numFiles, 1024], int)
    target = np.zeros([numFiles, 10])
    for i in range(numFiles):
        filePath = filelist[i]
        digit = int(filePath.split('_')[0])
        data[i] = img2vector(path + '/' + filePath)
        target[i][digit] = 1.0
    return(data, target)


def classification_plot(X_test, Y, model, accuracy, number):
    # PCA降维
    X_2dim = PCA(n_components=2).fit_transform(X_test)  # 到二维，用于可视化
    # Plot
    plt.subplot(3, 2, number)
    plt.scatter(X_2dim[:,0], X_2dim[:,1], c=Y, label="{} {:.2f}".format(model, accuracy), s=5)  # 按预测值区分不同的颜色
    plt.legend(loc='upper right', fontsize='xx-small')


def regression_plot(X_test, Y_test, Y_predict, model, score, number):
    # PCA降维
    X_1dim = PCA(n_components=1).fit_transform(X_test)  # 到二维，用于可视化
    # Plot
    plt.subplot(3, 3, number)
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.axis((Y_test.min(), Y_test.max(), Y_test.min(), Y_test.max()))  # X-Y坐标值范围
    plt.scatter(Y_test, Y_predict, label="{} {:.2f}".format(model, score), s=5)
    plt.plot(Y_test, Y_test, linestyle='-', c='black')  # diagonal line
    plt.legend(loc='upper right', fontsize='xx-small')  # legend


def use_kNN(X_train, X_test, Y_train, Y_test):
    # k nearest neighbor model
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')  # n可以通过交叉验证来选，distance, 越近的点权重越高
    knn.fit(X_train, Y_train)  # 拿训练集来train
    Y_predict = knn.predict(X_test)  # 为测试集进行predict
    accuracy = accuracy_score(Y_test, Y_predict)  # 与答案做比较，计算accuracy
    print(">>> kNN\n", classification_report(Y_test, Y_predict))
    classification_plot(X_test, Y_test, 'True', 1, 1)
    classification_plot(X_test, Y_predict, 'kNN', accuracy, 5)


def use_decision_tree(X_train, X_test, Y_train, Y_test):
    # decision tree
    print('Start training DT')
    tree = DecisionTreeClassifier(criterion='gini', max_depth=3).fit(X_train, Y_train)
    print('Training done!')
    Y_predict = tree.predict(X_test)
    print('Prediction done!')
    accuracy = accuracy_score(Y_test, Y_predict)
    print(">>> Decision tree\n", classification_report(Y_test, Y_predict))
    classification_plot(X_test, Y_predict, 'DTree', accuracy, 6)


def use_decision_tree_cv(data, target):
    # decision tree, cross-validation
    tree = DecisionTreeClassifier(criterion='gini', max_depth=3)
    cv = cross_val_score(tree, data, target, cv=10)
    print(">>> decision tree\n10-fold cross validation: {}\n".format(cv.mean()))


def use_naive_bayes(X_train, X_test, Y_train, Y_test):
    # Gussian NB
    nb = GaussianNB(priors=None).fit(X_train, Y_train)
    Y_predict = nb.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_predict)
    print(">>> Gaussian NB\n", classification_report(Y_test, Y_predict))
    classification_plot(X_test, Y_predict, 'NB-Gaussian', accuracy, 4)
    # Multinomial NB
    nb2 = MultinomialNB().fit(X_train, Y_train)
    Y_predict = nb2.predict(X_test)
    accuracy= accuracy_score(Y_test, Y_predict)
    print(">>> Multinomial NB\n", classification_report(Y_test, Y_predict))
    classification_plot(X_test, Y_predict, 'NB-Multinomial', accuracy, 2)
    # Bernolli NB
    nb3 = BernoulliNB().fit(X_train, Y_train)
    Y_predict = nb3.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_predict)
    print(">>> Bernolli NB\n", classification_report(Y_test, Y_predict))
    # classification_plot(X_test, Y_predict, 'NB-Bernolli', accuracy, 6)


def use_svm(X_train, X_test, Y_train, Y_test):
    model = svm.SVC(kernel='rbf', gamma='scale').fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    print(">>> SVM\n", classification_report(Y_test, Y_predict))
    accuracy = accuracy_score(Y_test, Y_predict)
    classification_plot(X_test, Y_predict, 'SVM', accuracy, 3)


def use_linear_regression(X_train, X_test, Y_train, Y_test, number):
    linear = LinearRegression().fit(X_train, Y_train)
    score = linear.score(X_test, Y_test)
    print(">>> linear regression\nscore = {:.2f}\n".format(score))
    Y_predict = linear.predict(X_test)
    regression_plot(X_test, Y_test, Y_predict, 'LR', score, number)


def use_poly_linear_regression(X_train, X_test, Y_train, Y_test, degree, number):
    X_train_poly = PolynomialFeatures(degree=degree).fit_transform(X_train)# 先将变量X处理成多项式特征，再用线性模型学习多项式特征的参数
    X_test_poly = PolynomialFeatures(degree=degree).fit_transform(X_test)
    poly = LinearRegression().fit(X_train_poly, Y_train)
    score = poly.score(X_test_poly, Y_test)
    print(">>> polynomial linear regression (degree = {})\nscore = {:.2f}\n".format(degree, score))
    Y_predict = poly.predict(X_test_poly)
    regression_plot(X_test, Y_test, Y_predict, "poly LR {}".format(degree), score, number)


def use_ridge_regression(X_train, X_test, Y_train, Y_test, number):
    model = Ridge(alpha=1, fit_intercept=True, solver='lsqr').fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    print(">>> ridge regression\nscore = {:.2f}\n".format(score))
    Y_predict = model.predict(X_test)
    regression_plot(X_test, Y_test, Y_predict, 'Ridge LR', score, number)


def use_poly_ridge_regression(X_train, X_test, Y_train, Y_test, degree, number):
    X_train_poly = PolynomialFeatures(degree=degree).fit_transform(X_train)# 先将变量X处理成多项式特征，再用线性模型学习多项式特征的参数
    X_test_poly = PolynomialFeatures(degree=degree).fit_transform(X_test)
    model = Ridge(alpha=1, fit_intercept=True, solver='lsqr').fit(X_train_poly, Y_train)
    score = model.score(X_test_poly, Y_test)
    print(">>> polynomial ridge regression (degree = {})\nscore = {:.2f}\n".format(degree, score))
    Y_predict = model.predict(X_test_poly)
    regression_plot(X_test, Y_test, Y_predict, "poly Ridge LR {}".format(degree), score, number)


def use_lasso_regression(X_train, X_test, Y_train, Y_test, number):
    model = Lasso(alpha=1, fit_intercept=True).fit(X_train, Y_train)  # over-fitting and feature selection
    score = model.score(X_test, Y_test)
    print(">>> lasso regression\nscore = {:.2f}\n".format(score))
    Y_predict = model.predict(X_test)
    regression_plot(X_test, Y_test, Y_predict, 'Lasso LR', score, number)


def use_poly_lasso_regression(X_train, X_test, Y_train, Y_test, degree, number):
    X_train_poly = PolynomialFeatures(degree=degree).fit_transform(X_train)# 先将变量X处理成多项式特征，再用线性模型学习多项式特征的参数
    X_test_poly = PolynomialFeatures(degree=degree).fit_transform(X_test)
    model = Lasso(alpha=3, fit_intercept=True, tol=0.001, max_iter=10000).fit(X_train_poly, Y_train)
    score = model.score(X_test_poly, Y_test)
    print(">>> polynomial lasso regression (degree = {})\nscore = {:.2f}\n".format(degree, score))
    Y_predict = model.predict(X_test_poly)
    regression_plot(X_test, Y_test, Y_predict, "poly Lasso LR {}".format(degree), score, number)


def use_multilayer_perceptron(X_train, X_test, Y_train, Y_test, layer_size, number):
    clf = MLPClassifier(hidden_layer_sizes=layer_size, activation='logistic', solver='adam',
                        learning_rate_init=0.1, max_iter=5000)
    clf.fit(X_train, Y_train)
    Y_predict = clf.predict(X_test)
    error_num = 0
    total_num = len(X_test)
    for i in range(total_num):
        if sum(Y_predict[i] == Y_test[i]) < 10:
            error_num += 1
    print(">>> multilayer perceptron, layer size = {}\nerror rate: {} / {} = {:.2f} %".format(
        layer_size, error_num, total_num, error_num * 100 / total_num))



if __name__ == '__main__':
    # 分类，鸢尾花数据集，test_size测试集比例；stratify保证划分后两集合各类别样本比例一致；random_state保证可重复性
    plt.figure(dpi=200)
    data, target = load_iris(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, stratify=target, test_size=.3, random_state=42)
    use_kNN(X_train, X_test, Y_train, Y_test)
    use_decision_tree_cv(data, target)
    use_decision_tree(X_train, X_test, Y_train, Y_test)
    use_naive_bayes(X_train, X_test, Y_train, Y_test)
    use_svm(X_train, X_test, Y_train, Y_test)
    plt.savefig('output/3.2.classification.jpg')

    # 回归，波士顿数据集
    plt.figure(dpi=200)
    data, target = load_boston(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=.3, random_state=42)
    use_linear_regression(X_train, X_test, Y_train, Y_test, 1)
    use_poly_linear_regression(X_train, X_test, Y_train, Y_test, 2, 2)
    use_poly_linear_regression(X_train, X_test, Y_train, Y_test, 3, 3)
    use_ridge_regression(X_train, X_test, Y_train, Y_test, 4)
    use_poly_ridge_regression(X_train, X_test, Y_train, Y_test, 2, 5)
    use_poly_ridge_regression(X_train, X_test, Y_train, Y_test, 3, 6)
    use_lasso_regression(X_train, X_test, Y_train, Y_test, 7)
    use_poly_lasso_regression(X_train, X_test, Y_train, Y_test, 2, 8)
    use_poly_lasso_regression(X_train, X_test, Y_train, Y_test, 3, 9)
    plt.savefig('output/3.2.regression.jpg')

    # 两个实例，开始，加载数据，预处理，创建分类器，训练分类器，测试集上做预测，计算准确率和召回率，结束
    X_train, Y_train = readDataset('data/digits/trainingDigits/')
    X_test, Y_test = readDataset('data/digits/testDigits/')
    use_multilayer_perceptron(X_train, X_test, Y_train, Y_test, (100,), 1)
    use_multilayer_perceptron(X_train, X_test, Y_train, Y_test, (200,), 1)
    use_multilayer_perceptron(X_train, X_test, Y_train, Y_test, (300,), 1)
    use_multilayer_perceptron(X_train, X_test, Y_train, Y_test, (500,), 1)
    use_multilayer_perceptron(X_train, X_test, Y_train, Y_test, (200,500,), 1)

    # show plots
    # plt.show()

    # l手写数字测试数据集，准备figure
    # feature_paths = ['A/A.feature', 'B/B.feature', 'C/C.feature', 'D/D.feature', 'E/E.feature']
    # label_paths = ['A/A.label', 'B/B.label', 'C/C.label', 'D/D.label', 'E/E.label']
    # x_train, y_train = load_databset(feature_paths[:4], label_paths[:4])
    # x_test, y_test = load_databset(feature_paths[4:], label_paths[4:])
    # x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.0)


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
# 线性回归，基于最小二乘法求解时，计算(XTX)-1，如果某些列线性相关较大，X的转置乘以X，值接近0，求逆时出现不稳定性
# 岭回归，解决最小二乘法的不稳定性，平方误差的基础上，增加正则项，ridge regression，用于共线性数据有偏估计


# 数据缺失值
# 数据冗余特征
# 数据可视化

# 强化学习（RL）
# model-based, 马尔可夫决策过程， MDP算法
# model free, 蒙特卡洛强化学习

# 深度强化学习（DRL）
# 将深度学习和强化学习结合在一起，通过深度神经网络直接学习环境（或观察）与状态动作值函数Q(s, a)之间的映射关系
# Deep Q Network, DQN




