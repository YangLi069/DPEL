# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:18:11 2016
Revised on Sat Mar 18 2017
Revised on Thu Mar 30 2017
@author: ly
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from tool import loadDataSet

'''
测试decisionTree模块
'''
import DecisionTree as dt

def test_DT1():
    X = np.array([[0, 0, 0], [0.1, 0.1, 0.1], [
                 1.0, 1.0, 1.0], [.99, .99, .99]])
    y = np.array([0, 0, 1, 1])
    tree = dt.DecisionTree(gt_privacy_p=float(1.0 / 100))
    tree.fit(X, y)
    pred1 = tree.predict(np.array([0.05, 0.05, 0.05]))
    print "pred1 value-0:", pred1
    pred2 = tree.predict(np.array([0.995, 0.995, 0.995]))
    print "pred2 value-1:", pred2


def test_DT2():
    X, y = loadDataSet("HCTrain.csv")
    X_test, y_test = loadDataSet("HCTest.csv")
    tree = dt.DecisionTree(gt_privacy_p=float(1.0 / 100))
    tree.fit(X, y)
    pred1 = tree.predict(X_test)
    print "AUC value", roc_auc_score(y_test, pred1)


'''
测试RFs模块
'''
from randomForest import MyRandomForest


def test_RF():
    X, y = loadDataSet("HCTrain.csv")
    X_test, y_test = loadDataSet("HCTest.csv")

    rfs = MyRandomForest(X, 
                         y, 
                         privacy_p=10000, 
                         num_learners=100,
                         max_depth = 6)
    pred = rfs.predict(X_test)
    print "accuracy", scoreAcc(y_test, pred)
    print "F1-score", f1_score(y_test, pred, average='micro')


'''
测试Adaboost-DT模块
'''
from AdaBoost import AdaboostDTBinaryClassifier
from tool import scoreAcc

def test_Ada():
    '''
    测试算法
    '''
#    X, y = loadSimpDataTrain()
#    X_test, y_test = loadSimpDataTest()
    X, y = loadDataSet("HCTrain.csv")
    X_test, y_test = loadDataSet("HCTest.csv")
    ada = AdaboostDTBinaryClassifier(privacy_p=50.0,
                                     num_learners=100,
                                     max_depth=3)
    ada.fit(X, y)
    pred = ada.predict(X_test)
    print "accuracy", scoreAcc(y_test, pred)
    print "F1-score", f1_score(y_test, pred, average='micro')


'''
测试水平分隔和融合模块
'''
from CRFsDP import CRFsDP
from CAdaBoostDP import CAdaBoostDP

def test_hI():
    privacy = np.array([75.0, 75.0, 75.0, 75.0, 75.0])
    num_learners = [100]  # gt个数
    max_heights = [5]  # gt高度
    Rfs = CRFsDP(privacy,
            num_learners,
            max_heights,
            "HCTrain.csv",
            "HCTest.csv",
            section=5)
    Rfs.parallelRF()

    AdaDP = CAdaBoostDP(privacy,
            num_learners,
            max_heights,
            "HCTrain.csv",
            "HCTest.csv",
            section=5)
    AdaDP.parallelAda()

'''
实现批量计算
'''
def test_batch():
    #group1
    privacy1 = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    privacy2 = np.array([25.0, 25.0, 25.0, 25.0, 25.0])
    privacy3 = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
    privacy4 = np.array([75.0, 75.0, 75.0, 75.0, 75.0])
    privacy5 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

    #group3
#    privacy1 = np.array([80.0, 80.0, 80.0])
#    privacy2 = np.array([80.0, 80.0, 80.0, 80.0])
#    privacy3 = np.array([80.0, 80.0, 80.0, 80.0, 80.0])
#    privacy4 = np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0])
#    privacy5 = np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0])
#    
#     privacy1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
#     privacy2 = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
#     privacy3 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
#     privacy4 = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
#     privacy5 = np.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0])
    privacy = []
    privacy.append(privacy1)
    privacy.append(privacy2)
    privacy.append(privacy3)
    privacy.append(privacy4)
    privacy.append(privacy5)

    num_learners = [100]  # gt个数
    max_heights = [3, 4, 5, 6, 7] #gt高度
    # max_heights = [7]

   #HC
    for i in range(len(privacy)):
       Rfs = CRFsDP(privacy[i],
                  num_learners,
                  max_heights,
                  "HCTrain.csv",
                  "HCTest.csv",
                  section=5)
       Rfs.parallelRF()
       AdaDP = CAdaBoostDP(privacy[i],
                       num_learners,
                       max_heights,
                       "HCTrain.csv",
                       "HCTest.csv",
                       section=5)
       AdaDP.parallelAda()
#
#    #BC
#    for i in range(len(privacy)):
#        Rfs = CRFsDP(privacy[i],
#                   num_learners,
#                   max_heights,
#                   "breastCancerTrain.csv",
#                   "breastCancerTest.csv",
#                   section=3+i)
#        Rfs.parallelRF()
#        AdaDP = CAdaBoostDP(privacy[i],
#                        num_learners,
#                        max_heights,
#                        "breastCancerTrain.csv",
#                        "breastCancerTest.csv",
#                        section=3+i)
#        AdaDP.parallelAda()
#    
#    #SS
#    for i in range(len(privacy)):
#        Rfs = CRFsDP(privacy[i],
#                   num_learners,
#                   max_heights,
#                   "skinSegTrain.csv",
#                   "skinSegTest.csv",
#                   section=3+i)
#        Rfs.parallelRF()
#        AdaDP = CAdaBoostDP(privacy[i],
#                        num_learners,
#                        max_heights,
#                        "skinSegTrain.csv",
#                        "skinSegTest.csv",
#                        section=3+i)
#        AdaDP.parallelAda()

    # #CC
    # for i in range(len(privacy)):
    #     Rfs = CRFsDP(privacy[i],
    #                num_learners,
    #                max_heights,
    #                "creditCardTrain.csv",
    #                "creditCardTest.csv",
    #                section=5)
    #     Rfs.parallelRF()
    #     AdaDP = CAdaBoostDP(privacy[i],
    #                     num_learners,
    #                     max_heights,
    #                     "creditCardTrain.csv",
    #                     "creditCardTest.csv",
    #                     section=5)
    #     AdaDP.parallelAda()
    
    #group4 kdd12
    # for i in range(len(privacy)):
    #     Rfs = CRFsDP(privacy[i],
    #               num_learners,
    #               max_heights,
    #               "kdd12Train2_30000.csv",
    #               "kdd12Test2_30000.csv",
    #               section=5)
    #     Rfs.parallelRF()
    #     AdaDP = CAdaBoostDP(privacy[i],
    #                    num_learners,
    #                    max_heights,
    #                    "kdd12Train2_30000.csv",
    #                    "kdd12Test2_30000.csv",
    #                    section=5)
    #     AdaDP.parallelAda()
    
    #group2
#    privacy11 = np.array([10.0, 1.0, 2.0, 1.0, 12.0])
#    privacy12 = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
#    privacy13 = np.array([12.0, 23.0, 24.0, 39.0, 100.0])
#    privacy14 = np.array([10.0, 25.0, 50.0, 75.0, 100.0])
#    privacy15 = np.array([3.0, 69.0, 72.0, 73.0, 86.0])
#    privacy16 = np.array([5.0, 12.0, 78.0, 98.0, 98.0])
#    privacy17 = np.array([70.0, 75.0, 80.0, 90.0, 100.0])
#    privacy18 = np.array([90.0, 85.0, 81.0, 85.0, 97.0])
#
#    privacy_1 = []
#    privacy_1.append(privacy11)
#    privacy_1.append(privacy12)
#    privacy_1.append(privacy13)
#    privacy_1.append(privacy14)
#    privacy_1.append(privacy15)
#    privacy_1.append(privacy16)
#    privacy_1.append(privacy17)
#    privacy_1.append(privacy18)
#
#    max_heights_1 = [6]
#    #HC
#    for i in range(len(privacy_1)):
#        Rfs = CRFsDP(privacy_1[i],
#                   num_learners,
#                   max_heights_1,
#                   "HCTrain.csv",
#                   "HCTest.csv",
#                   section=5)
#        Rfs.parallelRF()
#        AdaDP = CAdaBoostDP(privacy_1[i],
#                        num_learners,
#                        max_heights_1,
#                        "HCTrain.csv",
#                        "HCTest.csv",
#                        section=5)
#        AdaDP.parallelAda()
#
#    #BC
#    for i in range(len(privacy_1)):
#        Rfs = CRFsDP(privacy_1[i],
#                   num_learners,
#                   max_heights_1,
#                   "breastCancerTrain.csv",
#                   "breastCancerTest.csv",
#                   section=5)
#        Rfs.parallelRF()
#        AdaDP = CAdaBoostDP(privacy_1[i],
#                        num_learners,
#                        max_heights_1,
#                        "breastCancerTrain.csv",
#                        "breastCancerTest.csv",
#                        section=5)
#        AdaDP.parallelAda()
#    
#    #SS
#    for i in range(len(privacy_1)):
#        Rfs = CRFsDP(privacy_1[i],
#                   num_learners,
#                   max_heights_1,
#                   "skinSegTrain.csv",
#                   "skinSegTest.csv",
#                   section=5)
#        Rfs.parallelRF()
#        AdaDP = CAdaBoostDP(privacy_1[i],
#                        num_learners,
#                        max_heights_1,
#                        "skinSegTrain.csv",
#                        "skinSegTest.csv",
#                        section=5)
#        AdaDP.parallelAda()
#    
#    #CC
#    for i in range(len(privacy_1)):
#        Rfs = CRFsDP(privacy_1[i],
#                   num_learners,
#                   max_heights_1,
#                   "creditCardTrain.csv",
#                   "creditCardTest.csv",
#                   section=5)
#        Rfs.parallelRF()
#        AdaDP = CAdaBoostDP(privacy_1[i],
#                        num_learners,
#                        max_heights_1,
#                        "creditCardTrain.csv",
#                        "creditCardTest.csv",
#                        section=5)
#        AdaDP.parallelAda()
        
    
    #group5
#    privacy6 = np.array([1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0])
#    Rfs = CRFsDP(privacy6,
#                num_learners,
#                max_heights,
#                "kdd12Train_k_2.csv",
#                "kdd12Test2_k_2.csv",
#                section=5)
#    Rfs.parallelRF()
#    AdaDP = CAdaBoostDP(privacy6,
#                    num_learners,
#                    max_heights,
#                    "kdd12Train_k_2.csv",
#                    "kdd12Test2_k_2.csv",
#                    section=5)
#    AdaDP.parallelAda()

#    Rfs = CRFsDP(privacy6,
#                num_learners,
#                max_heights,
#                "kdd12Train_k_3.csv",
#                "kdd12Test2_k_3.csv",
#                section=5)
#    Rfs.parallelRF()
#    AdaDP = CAdaBoostDP(privacy6,
#                    num_learners,
#                    max_heights,
#                    "kdd12Train_k_3.csv",
#                    "kdd12Test2_k_3.csv",
#                    section=5)
#    AdaDP.parallelAda()
#    
#    Rfs = CRFsDP(privacy6,
#                num_learners,
#                max_heights,
#                "kdd12Train_k_4.csv",
#                "kdd12Test2_k_4.csv",
#                section=5)
#    Rfs.parallelRF()
#    AdaDP = CAdaBoostDP(privacy6,
#                    num_learners,
#                    max_heights,
#                    "kdd12Train_k_4.csv",
#                    "kdd12Test2_k_4.csv",
#                    section=5)
#    AdaDP.parallelAda()
#    
#    Rfs = CRFsDP(privacy6,
#                num_learners,
#                max_heights,
#                "kdd12Train_k_5.csv",
#                "kdd12Test2_k_5.csv",
#                section=5)
#    Rfs.parallelRF()
#    AdaDP = CAdaBoostDP(privacy6,
#                    num_learners,
#                    max_heights,
#                    "kdd12Train_k_5.csv",
#                    "kdd12Test2_k_5.csv",
#                    section=5)
#    AdaDP.parallelAda()
#    
#    Rfs = CRFsDP(privacy6,
#                num_learners,
#                max_heights,
#                "kdd12Train_k_6.csv",
#                "kdd12Test2_k_6.csv",
#                section=5)
#    Rfs.parallelRF()
#    AdaDP = CAdaBoostDP(privacy6,
#                    num_learners,
#                    max_heights,
#                    "kdd12Train_k_6.csv",
#                    "kdd12Test2_k_6.csv",
#                    section=5)
#    AdaDP.parallelAda()