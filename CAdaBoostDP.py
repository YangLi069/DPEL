# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 2017
@author: ly
"""
import math
import sys
import numpy as np

from AdaBoost import AdaboostDTBinaryClassifier
from tool import loadDataSet, splitData, logLoss
from tool import weightScore, AdaboostPL, horizontallySplitData
# from sysProf import fn_timer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

class CAdaBoostDP(object):
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

    def AdaBoostDP(self, hX, hy, privacy, num_learner, height, n_rows):
        '''
        每个用户本地AdaboostDP
        '''
        # 第p个用户lambda_p=Np/N
        lamb = float(hX.shape[0] / n_rows)
        # 第p个用户的训练集和验证集
        hX_train, hy_train, hX_val, hy_val = splitData(hX, hy)

        ada_p = AdaboostDTBinaryClassifier(privacy_p=privacy,
                                           num_learners=num_learner,
                                           max_depth=height)
        ada_p.fit(hX_train, hy_train)
        # 对比论文13 2016年
#       alpha_p.append(ada_p.learner_weight_)
#       Ada_set_p.append(ada_p.learners_)

        # 预测
        predAda_p = ada_p.predict(hX_val)
        # 第p个用户分类器准确度
        acc = accuracy_score(hy_val, predAda_p, normalize=True)
        lamb_acc = math.exp(lamb) * acc
        return lamb_acc, ada_p

    # @fn_timer
    def parallelAda(self):
        '''
        CAdaBoostDP
        '''
        X, y = loadDataSet(self.fileTrain)
        n_rows = X.shape[0]
        hX, hy = horizontallySplitData(X, y, self.section)
        # 加载测试数据
        X_test, y_test = loadDataSet(self.fileTest)

        lamb_acc = np.zeros(self.section, dtype=np.float)

        # 各个分类器在融合分类器的权重
        F = np.zeros(self.section, dtype=np.float)

        for i in self.num_learners:  # gt个数
            for j in self.max_height:  # gt高度
                predTest = np.zeros(y_test.shape[0], dtype=float)
#                predTest1 = np.zeros(y_test.shape[0], dtype=float)
                Ada_set = []
#                alpha_p = []
#                Ada_set_p = []

                for p in hX:
                    lamb_acc[p], ada_p = self.AdaBoostDP(hX[p], hy[p], self.privacy[p], i, j, n_rows)
                    Ada_set.append(ada_p)
                    
                # 融合后模型性能
                # 计算F
                F = weightScore(lamb_acc)
                for p in range(self.section):
                    predTest += Ada_set[p].predict(X_test) * F[p]

                predTest = [0.0 if pred <= 0.5 else 1.0 for pred in predTest]
#                print "ada_F1",  f1_score(y_test, predTest, average="micro")
#                print "ada_auc", roc_auc_score(y_test, predTest)

                #对比论文13
#                predTest1 = AdaboostPL(self.num_learners, self.section, Ada_set_p, alpha_p, lamb, X_test)
#                print "ada_F1_13",  f1_score(y_test, predTest1, average="micro")

                output = sys.stdout
                outputFile = open("output_plain.txt", 'a')
                sys.stdout = outputFile
                # print "AdaBoostDP-h", j
                # print "ada_logloss", logLoss(y_test, predTest)
                print "ada_F1",  f1_score(y_test, predTest, average="micro")
                print "ada_auc", roc_auc_score(y_test, predTest)
                outputFile.close()
                sys.stdout = output