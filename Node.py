# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 17:32:23 2016
Revised on Sun Dec 11 20:48 2016
@author: ly
"""
from sklearn.base import BaseEstimator

class Node(BaseEstimator):
    '''
    决策树root结点
    '''
    def __init__(self, feature_idx, threshold, left_tree, right_tree):
        '''
        初始化参数
        '''
        self.feature_idx = feature_idx
        self.threshold = threshold #阈值
        self.left_tree = left_tree #左子树
        self.right_tree = right_tree #右子树
        
        
    def fillPredict(self, X, outputs, index):
        '''
        左右结点值预测
        '''
#        print self.feature_idx
        split = X[:, self.feature_idx] < self.threshold
        left_index = index & split
        right_index = index & ~split
        
        #递归
        if self.left_tree is not None: 
            self.left_tree.fillPredict(X, outputs, left_index)
        if self.right_tree is not None:
            self.right_tree.fillPredict(X, outputs, right_index)