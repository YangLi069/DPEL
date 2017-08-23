# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 17:04:45 2016
Revised on Sun Dec 11 20:48 2016
@author: ly
"""
import math
import numpy as np
from sklearn.base import BaseEstimator

from tool import midPoints, majority, findBestGiniSplit, laplaceMechanism
from Leaf import Leaf
from Node import Node


class DecisionTree(BaseEstimator):
    '''
    决策树，使用Gini划分结点
#    num_features_per_node: 属性个数f
    min_samples_leaf: 叶结点的最小样本个数
    max_depth: 树的最大高度
    gt_privacy_p:参与者分配给每棵决策树的隐私预算
    depth_gt_privacy_p:分配到每一层的隐私预算
    classes：类别属性，默认取所有类标签的唯一值
    '''

    def __init__(self,\
                 gt_privacy_p,\
                 min_samples_leaf=1,\
                 max_depth=6):
        '''
        初始化参数
        '''
        self.root = None  # 树的根节点
#        self.num_features_per_node = num_features_per_node
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.gt_privacy_p = gt_privacy_p
        self.classes = None
        self.getThresholds = self.thresholds
#        if max_thresholds is None:
#            self.getThresholds = self.thresholds
#        else:
#            self.getThresholds = self.randomThresholdsSubset

    def thresholds(self, x):
        '''
        求集合的相邻元素的中间值
            x: 一个属性对应的列向量
        '''
        if len(x) > 1:
            # print np.unique(x)
            return midPoints(np.unique(x))
        else:
            return x

#    def randomThresholdsSubset(self, x):
#        '''
#        随机生成阈值子集
#        '''
#        n_rows = len(x)
#        k = self.max_thresholds
#        n_samples = min(n_rows, k)
#        randSubset = random.sample(x, n_samples) #从x中随机抽取n_samples子集
#        return self.thresholds(randSubset)

    def splitData(self, X, y, h, sample_weight, epsilon):
        '''
        划分数据
        #d是分裂每个节点包含的样本的属性个数
        h是树的高度
        epsilon是隐私预算量
        '''
        n_rows, n_features = X.shape
        if n_rows <= self.min_samples_leaf or h > self.max_depth:
            self.n_leaf += 1  # 叶结点个数增1
            leaf = majority(y, self.classes)  # 返回类标签和记录条数
            if epsilon > 0.0:
                leaf += math.floor(laplaceMechanism(epsilon))
            return Leaf(leaf)
        
        elif np.all(y == y[0]):  # 检查所有元素是否与第一个元素相同
            self.n_leaf += 1  # 叶结点个数增1
#            y[0] += math.floor(laplaceMechanism(epsilon))
            return Leaf(y[0])  # 所有类标签都一样
            
        else:
            #random_feature_indices = random.sample(xrange(n_features), d) #随机采d个特征
            # 初始化最佳分裂属性和分裂点
            best_split_score = 999999999
            best_feature_idx = None
            best_threshold = None
            # 遍历所有的特征
            for feature_t in range(n_features):
                col_vector = X[:, feature_t]  # 列向量
                feature_vector = self.getThresholds(col_vector)
                # 分类，找到Gini最小的分割点
                thresh, totalScore = findBestGiniSplit(
                    self.classes, col_vector, feature_vector, y, sample_weight)
                if thresh is not None:
                    if totalScore < best_split_score:  # 找到更好的分裂属性，更新
                        best_split_score = totalScore  # 更新Gini值
                        best_feature_idx = feature_t  # 更新最佳分裂属性
                        best_threshold = thresh  # 更新分裂阈值 
                else:
                    break

            if best_feature_idx is not None:
#                print "best_feature_idx", best_feature_idx
                 #如果当前结点不是叶子结点，那么取一半的隐私量分给当前结点，另一半用于子结点
                if epsilon > 0.0:
                    best_threshold += math.floor(laplaceMechanism(0.5*epsilon))
                
                # 左右分支
                left_branch = X[:, best_feature_idx] < best_threshold
                right_branch = ~left_branch
#                print "lb.len", len(left_branch)
#                print "rb.len", len(right_branch)

                left_data = X[left_branch, :]  # 左子树X
                right_data = X[right_branch, :]  # 右子树X
#                print "ld.size", left_data.shape
#                print "rd.size", right_data.shape
                left_labels = y[left_branch]  # 左子树y
                right_labels = y[right_branch]  # 右子树y

                # 分配子结点隐私预算
                epsilon_left = epsilon_right = 0
                if left_data.shape[0] == 0 and right_data.shape[0] != 0:
                    epsilon_right = 0.5 * epsilon
                elif left_data.shape[0] != 0 and right_data.shape[0] == 0:
                    epsilon_left = 0.5 * epsilon
                else:
                    epsilon_left = 0.25 * epsilon
                    epsilon_right = 0.25 * epsilon

                # 避免数据重复引用
                del y
                del X
                del left_branch
                del right_branch

                # 递归该过程，建立左右子树
                left_tree = self.splitData(
                           left_data, left_labels, h+1, sample_weight, epsilon_left)
                right_tree = self.splitData(
                           right_data, right_labels, h+1, sample_weight, epsilon_right)
 
               # 构建出一个树结构root+left_tree+right_tree
                node = Node(best_feature_idx, best_threshold,\
                            left_tree, right_tree)
                return node

    def fit(self, X, y, sample_weight=None):
        '''
        训练模型，遵循fit/predict模式
        '''
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        self.classes = np.unique(y)  # eg.{0,+1}
        self.n_classes = len(self.classes)  # 类别个数
        self.n_leaf = 0  # 叶结点个数初始化为0

#        n_features = X.shape[1]
        # 每个节点属性个数
#        if self.num_features_per_node is None:
#            d = int(math.sqrt(n_features)) #默认d是原值的平方根
#        else:
#            d = self.num_features_per_node
#        print "d:", d

        self.root = self.splitData(X, y, 1, sample_weight, self.gt_privacy_p)

    def predict(self, X):
        '''
        预测
        '''
        X = np.atleast_2d(X)
        n_rows = X.shape[0]

        # 输出数组，树结点递归填充
        outputs = np.zeros(n_rows)
        index = np.ones(n_rows, dtype='bool')
#        print "self.root", self.root
        if self.root is not None:
            self.root.fillPredict(X, outputs, index)
        return outputs
