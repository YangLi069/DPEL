# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:15:59 2016
Revised on Sun Dec 11 20:48 2016
@author: ly
"""
import numpy as np
from sklearn.base import clone, BaseEstimator

class BaseEnsemble(BaseEstimator):
    '''
    基础集成类
    '''
    def __init__(self,
                 baseModel,
                 num_learners,
                 featureSubsetPer):
        '''
        初始化
        baseModel 基础分类器,CART
        num_learners 分类器个数
        featureSubsetPer 特征子集比例，取0.8
        '''
        self.baseModel = baseModel
        self.num_learners = num_learners
        self.featureSubsetPer = featureSubsetPer
        self.learners = None  # G
        self.weights = None  # w

    def fit(self, X, y):
        '''
        训练分类器，gt
        '''
        self.learners = []  # 所有分类器gt集合G
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        n_rows, totalFeatures = X.shape
        # 权重初始化1/T，在RFs中所有gt权重相同
        self.weights = np.ones(
            self.num_learners, dtype="float") / self.num_learners

        self.init_fit(X, y)  # 预处理X和y，在分类和回归类里定义

        if self.featureSubsetPer < 1:
            sub_features = int(self.featureSubsetPer * totalFeatures)
            self.feature_subset = []
        else:
            sub_features = totalFeatures
            self.feature_subset = None

        # 执行T轮
        for i in xrange(self.num_learners):
            # 有放回的抽样
            # np.random.seed(1234)
            ind = np.random.random_integers(0, n_rows - 1, n_rows)

            X_subset = X[ind, :]  # 抽取大小为|D|的子集X
            if sub_features < totalFeatures:  # 选出特征
                features_ind = np.random.permutation(totalFeatures)[:sub_features]
                self.feature_subset.append(features_ind)
                X_subset = X_subset[:, features_ind]

            y_subset = y[ind]  # y

            learner = clone(self.baseModel)
            learner.fit(X_subset, y_subset)  # gt

            self.learners.append(learner)  # 添加到G
