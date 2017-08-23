# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 16:06:07 2016
Revised on Sun Dec 11 20:48 2016
@author: ly
"""
import numpy as np

class Leaf(object):
    '''
    决策树叶结点
    '''
    def __init__(self, v):
        '''
        初始化结点值
        '''
        self.v = v
        
    def toStr(self, split="#"):
        '''
        转换为字符串，便于打印
        '''
        return split + "<" + str(self.v) + ">"
    
    def __str__(self):
        '''
        重写该函数
        '''
        return self.toStr
    
    def predict(self, X):
        '''
        设定目标变量
        '''
        X = np.atleast_2d(X)
        output = np.zeros(X.shape[0])
        output[:] = self.v
        return output
        
    def fillPredict(self, X, output, index):
        '''
        指定index的赋值
        '''
        output[index] = self.v