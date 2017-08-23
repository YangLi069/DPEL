# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 15:35:54 2016
Revised on Sun Dec 11 20:48 2016
Revised on Thu Mar 30 2017
@author: ly
"""

'''
公用函数：
1.计算gini（分类）和方差（回归）
2.加载数据和划分数据
3.计算差分值
4.计算准确度
5.计算Fp
'''

import numpy as np
import math
import random

#####多数投票###################


def midPoints(x):
    '''
    eg.x=array([1,2,3,4])
    return array([1.5,2.5,3.5])
    '''
    return (x[1:] + x[:-1]) / 2.0


def majority(y, classes):
    '''
    多数投票原则
    '''
    if classes is None:
        classes = np.unique(y)  # 取出所有唯一类别
    votes = np.zeros(len(classes))  # 初始权值为0
    for i, c in enumerate(classes):
        votes[i] = np.sum(y == c)  # 统计各个类别的票数
    majority_idx = np.argmax(votes)  # 票数最多的下标
#    print int(votes[majority_idx])
    return classes[majority_idx]  # 多数投票对应的类标签

#####计算Gini（分类）########


def Gini(classes, y, sample_weight):
    '''
    gini指数，划分结点指标
    一种数据不纯度的度量方法
    Gini(D)=1-\sum(|Dc|/|D|)^2
    '''
    sum_squares = 0.0
    n = len(y)
    if n == 0:
        return 0.0
    else:
        n2 = float(n*n)
        if sample_weight is not None:
            for c in classes:  # gini计算
                count = np.sum(y == c)
                getindex = np.where(y == c)[0]
                w = np.sum(sample_weight[getindex])
                # count -= w * count
                c2 = ((count * w) ** 2) / n2
                sum_squares += c2
        else:
            for c in classes:  # gini计算
                count = np.sum(y == c)
                c2 = (count ** 2) / n2
                sum_squares += c2
        return 1 - sum_squares


# def findMinVarSplit(feature_vector, thresholds, y):
#    '''
#    回归，找到最小方差分割点, 参考C++代码实现
#    '''
#    best_score = 999999999
#    best_thresh = None
#
#    for t_index in thresholds:
#        index  = feature_vector < t_index
#        left = y[index]
#        right = y[~index]
#        left_size = left.shape[0]
#        right_size = right.shape[0]
#
#        if left_size > 0 and right_size > 0:
#            totalSize = float(left_size + right_size)
#            score = (left_size / totalSize) * np.var(left) + \
#                (right_size / totalSize) * np.var(right)
#            if score < best_score:
#                best_score = score
#                best_thresh = t_index
#    return best_thresh, best_score


def findBestGiniSplit(classes, col_vector, thresholds, y, sample_weight):
    '''
    分类，找到最小gini分割点, 参考C++代码实现
    '''
    best_score = 999999999
    best_thresh = None
    n = len(y)

    # 遍历属性的每一个取值
    for t_index in thresholds:
        index = col_vector < t_index  # 二分
        left_labels = y[index]  # D1
        right_labels = y[~index]  # D2
        left_score = Gini(classes, left_labels, sample_weight)  # Gini(D1)
        right_score = Gini(classes, right_labels, sample_weight)  # Gini(D2)
        left_n = len(left_labels)  # D1样本个数|D1|
        right_n = len(right_labels)  # D2样本个数|D2|
        # Gini(D,A) = |D1|/|D|*Gini(D1) + |D2|/|D|*Gini(D2)
        totalScore = (left_n / n) * left_score + (right_n / n) * right_score
        if totalScore < best_score:
            best_score = totalScore
            best_thresh = t_index
    return best_thresh, best_score

#############加载和划分数据###########


def loadDataSet(fileName):
    '''
    加载数据，默认是.csv文件，逗号分隔
    '''
    n_features = len(open(fileName).readline().split(','))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArray = []
        curLine = line.strip().split(',')
        for i in range(n_features - 1):
            lineArray.append(float(curLine[i]))
        dataMat.append(lineArray)
        labelMat.append(float(curLine[-1]))
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
#    map(lambda label: 0 if label==0 else 1, labelMat)
    return dataMat, labelMat


def splitData(X, y, trainPre=0.8):
    '''
    划分训练集0.8和测试集0.2
    '''
    n_rows, n_cols = X.shape
    ind = np.arange(n_rows)
#    np.random.shuffle(ind)
    n_train = int(n_rows * trainPre)
    train_ind = ind[:n_train]
    test_ind = ind[n_train:]
    X_train = X[train_ind, :]
    X_test = X[test_ind, :]
    y_train = y[train_ind]
    y_test = y[test_ind]
    return X_train, y_train, X_test, y_test


def horizontallySplitData(X, y, section=5):
    '''
    水平划分数据集，每部分有且仅属于一个参与者
    每个子数据集属性相同，但记录不同
    section：划分个数，实验默认取5
    '''
    if section == 1:
        return X, y
    else:
        n_rows = X.shape[0]
        rangelist = range(1, n_rows)
        random.seed(12345)  # 随机划分结果重现
        valuelist = random.sample(rangelist, section - 1)
        valuelist.sort()
        valuelist.append(n_rows)
#       print "valuelist", valuelist
        k = 0
        j = 0
        # 字典结构，key是用户id，value是用户的数据
        h_X = {}
        h_y = {}
        for i in valuelist:
            assert j <= section - 1
            h_X[j] = X[k:i]
            h_y[j] = y[k:i]
            k = i
            j += 1
        # print h_X, h_y
        return h_X, h_y

###########计算RMSE和准确度###################


def RMSE(y, yhat):
    '''
    计算均方根误差
    '''
    return np.sqrt(np.mean((yhat - y)**2))


def scoreAcc(y, yhat):
    '''
    计算分类准确度
    输出: (0,1)
    accuracy = TP+TN / n
    '''
    n = len(y)
    tp = np.sum(yhat * y)
    tn = np.sum((1 - yhat) * (1 - y))
    acc = float((tp + tn) / n)
    return acc
    # return float(np.sum(yhat == y)) / len(y)


import scipy as sp


def logLoss(y, yhat):
    '''
    计算Logarithmic Loss
    '''
    epsilon = 1e-15
    yhat = sp.maximum(epsilon, yhat)
    yhat = sp.minimum(1 - epsilon, yhat)
    ll = sum(y * sp.log(yhat) + sp.subtract(1, y)
             * sp.log(sp.subtract(1, yhat)))
    ll = -1.0 / len(y) * ll
    return ll

#from sklearn.metrics import classification_report
# def printClassifierResults(y, yhat):
#     '''
#     打印分类结果报告
#     '''
#     target_names = ["neg", "pos"]
#     print classification_report(y, yhat, target_names=target_names, digits=3)


#####计算Fp###################
def weightScore(w):
    '''
    标准化权重值
    '''
    F = []
    for w_p in w:
        F_p = w_p / np.sum(w)
        F.append(F_p)
    return F


#####计算差分###################
def sgn(x):
    '''
    符号函数
    '''
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def laplaceMechanism(privacy):
    '''
    拉普拉斯噪声, privacy:ε
    p(x|λ) = 1/2λ*exp(−|x|/λ)
    '''
#    if privacy is not None:
#        value = (privacy / 2.0) * math.exp(-1 * privacy * math.fabs(x))
#        return value
#    else:
#        return 0.0

#     mu = 0
#     b = 1.0 / privacy
#     a = random.uniform(-0.5, 0.5)
#     return mu - b * sgn(a) * math.log(1 - 2 * math.fabs(a))

    return 0.0


#####论文13方案###################
def AdaboostPL(num_learners, section, Ada_set_p, alpha_p, lamb, X_test):
    #    print "Ada_set_p", Ada_set_p
    #    print "alpha_p", alpha_p
    #    print "lamb", lamb
    predTotal = np.zeros(X_test.shape[0], dtype=float)
    for t in range(num_learners[0]):
        pred_t = np.zeros(X_test.shape[0], dtype=float)
        alpha_t = 0.0
        for id in range(section):
            #            print "Ada_set_p[id]", Ada_set_p[id]
            #            print "Ada_set_p[id][t]", Ada_set_p[id][t]
            temp = Ada_set_p[id][t].predict(X_test)
            temp[temp == 0] = -1
            pred_t += temp
            alpha_t += alpha_p[id][t] * lamb[id]
#        print "alpha_t", alpha_t
#        print "pred_t", pred_t
#        pred_t = [ 0 if i==0 else 1 for i in pred_t]
        predTotal += pred_t * alpha_t

    predTotal = [0 if j <= 0 else 1 for j in predTotal]
#    print "predTotal", predTotal
    return predTotal
