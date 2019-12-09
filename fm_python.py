# coding:UTF-8

from __future__ import division
from math import exp
import numpy as np
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score

trainData = 'diabetes_train.txt'
testData = 'diabetes_test.txt'
featureNum = 8
max_list = []
min_list = []


def normalize(x_list, max_list, min_list):
    index = 0
    scalar_list = []
    for x in x_list:
        x_max = max_list[index]
        x_min = min_list[index]
        if x_max == x_min:
            x = 1.0
        else:
            x = round((x - x_min) / (x_max - x_min), 4)
        scalar_list.append(x)
        index += 1
    return scalar_list


def loadTrainDataSet(data):
    global max_list
    global min_list
    dataMat = []
    labelMat = []

    fr = open(data)  # 打开文件

    for line in fr.readlines():
        currLine = line.strip().split(',')
        # lineArr = [1.0]
        lineArr = []

        for i in range(featureNum):
            lineArr.append(float(currLine[i]))

        dataMat.append(lineArr)

        labelMat.append(float(currLine[-1]) * 2 - 1)

    data_array = np.array(dataMat)
    max_list = np.max(data_array, axis=0)
    min_list = np.min(data_array, axis=0)

    scalar_dataMat = []
    for row in dataMat:
        scalar_row = normalize(row, max_list, min_list)
        scalar_dataMat.append(scalar_row)
    return scalar_dataMat, labelMat


def loadTestDataSet(data):
    global max_list
    global min_list
    dataMat = []
    labelMat = []

    fr = open(data)  # 打开文件

    for line in fr.readlines():
        currLine = line.strip().split(',')
        lineArr = []

        for i in range(featureNum):
            lineArr.append(float(currLine[i]))

        dataMat.append(lineArr)

        labelMat.append(float(currLine[-1]) * 2 - 1)

    data_array = np.array(dataMat)

    scalar_dataMat = []
    for row in dataMat:
        scalar_row = normalize(row, max_list, min_list)
        scalar_dataMat.append(scalar_row)
    return scalar_dataMat, labelMat


def sigmoid(inx):
    return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
    # return 1.0 / (1 + exp(-inx))


def stocGradAscent(dataMatrix, classLabels, k, iter):
    # dataMatrix用的是mat, classLabels是列表
    m, n = shape(dataMatrix)
    alpha = 0.1
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = zeros((n, 1))
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))

    r = 0.0001
    e = 1e-8

    g_0 = 0
    gw = zeros((n, 1))
    gv = zeros((n, k))
    decay_rate = 0.99

    def adagrad(_d, _c, _e):
        return _d / (np.sqrt(_c) + _e)

    for it in range(iter):
        print(it)
        for x in range(m):  # 随机优化，对每一个样本而言的
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # multiply对应元素相乘
            # 完成交叉项
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.

            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出
            # print "y: ",p
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            # print "loss: ",loss

            dw = loss * classLabels[x]
            g_0 = decay_rate * g_0 + (1 - decay_rate) * dw ** 2
            w_0 = w_0 - alpha * adagrad(dw, cache, e) - alpha * r * w_0

            # w_0 = w_0 - alpha * loss * classLabels[x] - alpha * r * w_0

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    dw = loss * classLabels[x] * dataMatrix[x, i]
                    gw[i] = decay_rate * gw[i] + (1 - decay_rate) * dw ** 2
                    w[i, 0] = w[i, 0] - alpha * adagrad(dw, gw[i], e) - alpha * r * w[i, 0]

                    # w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i] - alpha * r * w[i, 0]
                    for j in range(k):
                        # v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (
                        #         dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[
                        #     x, i]) - alpha * r * v[i, j]
                        dw = loss * classLabels[x] * (
                                dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[
                            x, i])
                        gv[i, j] = decay_rate * gv[i, j] + (1 - decay_rate) * dw ** 2
                        v[i, j] = v[i, j] - alpha * adagrad(dw, gv[i, j], e) - alpha * r * v[i, j]

    return w_0, w, v


def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    print('dataMatrix.shape=', dataMatrix.shape)
    print('w shape=', w.shape)
    print('v shape=', v.shape)
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出

        pre = sigmoid(p[0, 0])

        result.append(pre)

        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue

    print(result)
    print("AUC=", round(roc_auc_score(classLabels, result), 4))

    return float(error) / allItem


if __name__ == '__main__':
    dataTrain, labelTrain = loadTrainDataSet(trainData)
    dataTest, labelTest = loadTestDataSet(testData)
    date_startTrain = datetime.now()
    print("开始训练")
    w_0, w, v = stocGradAscent(mat(dataTrain), labelTrain, k=30, iter=1000)
    print("训练准确性为：%f" % (1 - getAccuracy(mat(dataTrain), labelTrain, w_0, w, v)))
    date_endTrain = datetime.now()
    print("训练时间为：%s" % (date_endTrain - date_startTrain))
    print("开始测试")
    print("测试准确性为：%f" % (1 - getAccuracy(mat(dataTest), labelTest, w_0, w, v)))
