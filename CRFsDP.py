# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 2017
@author: ly
"""
import math
import sys
import numpy as np

from randomForest import MyRandomForest
from tool import loadDataSet, splitData, logLoss
from tool import weightScore, horizontallySplitData
# from sysProf import fn_timer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


class CRFsDP(object):
    def __init__(self,
                 privacy,
                 num_learners,
                 max_height,
                 fileTrain,
                 fileTest,
                 section=5):
        '''
        初始化参数
        privacy 各个参与者指定的隐私预算
        max_height 决策树的高度
        num_learners 决策树的个数
        fileTrain 训练集
        fileTest 测试集
        section: 参与者个数，至少为1
        '''
        assert section >= 1
        self.privacy = privacy
        self.num_learners = num_learners
        self.max_height = max_height
        self.section = section
        self.fileTrain = fileTrain
        self.fileTest = fileTest

    def RFsDP(self, hX, hy, privacy, num_learner, max_height, n_rows):
        '''
        每个用户本地RFsDP
        '''
        # 第p个用户lambda_p=Np/N
        lamb = float(hX.shape[0] / n_rows)
        # 第p个用户的训练集和验证集
        hX_train, hy_train, hX_val, hy_val = splitData(hX, hy)

        rfs_p = MyRandomForest(X=hX_train,
                               y=hy_train,
                               privacy_p=privacy,
                               num_learners=num_learner,
                               max_depth=max_height)
        predRF_p = rfs_p.predict(hX_val)

        # 第p个用户分类器准确度
        acc = accuracy_score(hy_val, predRF_p, normalize=True)
#        print "acc", acc
        # 第p个用户w函数值
        lamb_acc = math.exp(lamb) * acc
        return lamb_acc, rfs_p

    # @fn_timer
    def parallelRF(self):
        '''
        CRFsDP
        '''
        # 加载训练数据
        X, y = loadDataSet(self.fileTrain)
        n_rows = X.shape[0]  # 行数
        # 水平划分数据集
        hX, hy = horizontallySplitData(X, y, self.section)
        # 加载测试数据
        X_test, y_test = loadDataSet(self.fileTest)

        # 融合函数值
        lamb_acc = np.zeros(self.section, dtype=np.float)
        # 各个分类器在融合分类器的权重
        F = np.zeros(self.section, dtype=np.float)

        RFs_set = []
        for i in self.num_learners:  # gt个数
            for j in self.max_height:  # gt高度
                predTest = np.zeros(y_test.shape[0], dtype=float)

                for p in range(self.section):
                    lamb_acc[p], rfs_p = self.RFsDP(hX[p], hy[p], self.privacy[p], i, j, n_rows)
                    RFs_set.append(rfs_p)

                # 融合后模型性能
                # 标准化权重
                F = weightScore(lamb_acc)

                for p in range(self.section):
                    predTest += RFs_set[p].predict(X_test) * F[p]

                predTest = [0 if pred <= 0.5 else 1 for pred in predTest]
#                print "rfs_F1", f1_score(y_test, predTest, average="micro")
#                print "rfs_auc", roc_auc_score(y_test, predTest)

                output = sys.stdout
                outputFile = open("output_plain.txt", 'a')
                sys.stdout = outputFile
                # print "RFsDP-h", j
                # print "rfs_logloss", logLoss(y_test, predTest)
                print "rfs_F1",  f1_score(y_test, predTest, average="micro")
                print "rfs_auc", roc_auc_score(y_test, predTest)
                outputFile.close()
                sys.stdout = output