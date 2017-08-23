# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 15:53:54 2016
Revised on Sun Dec 14 10:20 2016
Revised on Thu Mar 30 2017
@author: ly
"""
import math
import numpy as np
from sklearn.base import clone

from DecisionTree import DecisionTree

class AdaboostDTBinaryClassifier(object):
    '''
    Adaboost-DT类
    num_learners:决策树个数
    max_depth:树的高度，默认1
    privacy_p:参与者隐私预算
    '''
    def __init__(self,
                 privacy_p,
                 num_learners=100,
                 max_depth=1,
                 min_samples_leaf=1):
        '''
        初始化参数，决策树为CART
        '''
        self.base_learner = DecisionTree(gt_privacy_p=privacy_p / num_learners,
                                         min_samples_leaf=min_samples_leaf,
                                         max_depth=max_depth)

        self.num_learners = num_learners
        #used in fit()
        self.learners_ = []  # 所有分类器集合
        # 每个分类器权重
        self.learner_weight_ = np.zeros(self.num_learners, dtype=np.float)

    def _boost(self, X, y, sample_weight):
        '''
        一次迭代构建一个决策树
        输入：X,y 待分类的数据集
        输出：learner 决策树分类器
            sample_weight 更新后数据权重wt
            learner_weight 分类器权重ηt
        '''
        learner = clone(self.base_learner)
        dtc = learner
        dtc.fit(X, y, sample_weight)  # gt
        pred_y = dtc.predict(X)
        pred_y = [0.0 if i <= 0.0 else 1.0 for i in pred_y]
        # print 'pred_y', pred_y
        indicator = np.ones(X.shape[0]) * [pred_y != y][0]  # 错分的样本
        tau = np.dot(sample_weight, indicator) / np.sum(sample_weight)  # τt
        # 结果的权重，max函数是防止除0异常，ηt = 1/2*ln((1− τt)/τt)
        eta = 0.5 * np.log((1 - tau) / max(tau, 1e-16))
#        print "eta:", eta
        # wt = wt · exp(ηt[yi!=gt(xi)])
        new_sample_weight = sample_weight * np.exp(eta * indicator)
        return learner, new_sample_weight, eta

    def fit(self, X, y):
        '''
        构建分类器，遵循fit/predict模式
        '''
        # 权重初始化为1/N
        sample_weight = np.ones(X.shape[0])/X.shape[0]
        # 执行T轮
        for i in range(self.num_learners):
            learner, sample_weight, eta = self._boost(X, y, sample_weight)
            self.learners_.append(learner)
            self.learner_weight_[i] = eta

    def predict(self, X):
        '''
        预测
        '''
        predicts = []
        for learner in self.learners_:
            pred = learner.predict(X)
            # pred = [0.0 if i <= 0.0 else 1.0 for i in pred]
            pred[pred == 0] = -1  # 避免乘0计算
            predicts.append(pred)

        predicts = np.array(predicts)
        pr = np.sign(np.dot(self.learner_weight_, predicts))
        pr[pr == -1] = 0
        return pr
