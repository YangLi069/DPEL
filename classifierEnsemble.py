# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:57:27 2016
Revised on Sun Dec 11 20:48 2016
Revised on Thu Mar 30 2017
@author: ly
"""
import numpy as np

from baseEnsemble import BaseEnsemble

class ClassifierEnsemble(BaseEnsemble):
    '''
    BaseEnsemble类派生
    '''
    def __init__(self,
                 baseModel,
                 num_learners,
                 featureSubsetPer=0.8):
        '''
        初始化参数，含义与BaseEnsemble一致
        '''
        BaseEnsemble.__init__(self,
                              baseModel,
                              num_learners,
                              featureSubsetPer)  # 初始化基类

        self.classes = None  # 不同的类标签，如{-1, +1}
        self.classlist = None  # 类标签列表， 如[-1, +1]

    def init_fit(self, X, y):
        '''
        初始化X和y
        '''
        self.classes = np.unique(y)
        self.classlist = list(self.classes)  # array->list

    def predict_votes(self, X):
        '''
        给出多个分类器投票结果
        '''
        n_rows, n_features = X.shape
        n_classes = len(self.classes)
        votes = np.zeros([n_rows, n_classes])

        assert self.learners is not None
        for i, learner in enumerate(self.learners):  # G->gt
            w = self.weights[i]
            if self.feature_subset is not None:
                features_ind = self.feature_subset[i]
                X_subset = X[:, features_ind]
                pred_y = learner.predict(X_subset)
            else:
                pred_y = learner.predict(X)

#            pred_y = diffPrivTree(pred_y, self.privacy_p)
            for c in self.classes:
                c_ind = self.classlist.index(c)
                votes[pred_y == c, c_ind] += w
        return votes

    def predict(self, X):
        '''
        预测
        '''
        X = np.atleast_2d(X)
        majority_ind = np.argmax(self.predict_votes(X), axis=1)  # 按行逐条预测
        pr = np.array([self.classlist[i] for i in majority_ind])  # 给出对应的类标签
#        print "pr", np.unique(pr)
        return pr
