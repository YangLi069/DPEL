# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 16:30:29 2016
Revised on Sun Dec 11 20:48 2016
Revised on Thu Mar 30 2017
@author: ly
"""
from DecisionTree import DecisionTree
from classifierEnsemble import ClassifierEnsemble

def MyRandomForest(X,
                   y,
                   privacy_p,
                   num_learners = 100,
                   max_depth = 1,
                   min_samples_leaf = 1,
                   ):
    '''
    创建和训练随机森林
    num_learners: 树的个数
    max_height: 树的最大高度
    min_samples_leaf：叶结点的最小样本个数
    privacy_p 参与者的隐私预算
    '''

#    if max_depth is None:
#        max_depth = int(np.log2(X.shape[0]))+1 #树高设为log2(n)+1
#
#    if min_samples_leaf is None:
#        min_samples_leaf = 1

    #CART
    decisionTree = DecisionTree(gt_privacy_p = privacy_p/num_learners,
                                min_samples_leaf = min_samples_leaf,
                                max_depth = max_depth)

#    if regression:
#        rf = RegressionEnsemble(baseModel = decisionTree, num_learners = num_learners, \
#                                bagging = bagging, horizontalOrVertical = horizontalOrVertical, privacy_p = privacy_p)
#    else:
    RFs = ClassifierEnsemble(baseModel = decisionTree,
                             num_learners = num_learners)

    RFs.fit(X, y)
    return RFs