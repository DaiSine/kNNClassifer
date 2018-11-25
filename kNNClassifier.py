# -*- coding:utf-8 -*-
# -*- python3.6 -*-

import os
from os import listdir
from numpy import *
import numpy as np
import operator
import matplotlib.pyplot as plt

# kNN分类器
def kNNclassifier(test_data, train_data, train_label, k):
    dataSetSize = train_data.shape[0]
    # 计算测试样本与每一个训练样本的距离
    distances = np.sqrt(np.sum(np.square(tile(test_data,(dataSetSize,1))-train_data),axis=1))
    sort_distance = distances.argsort()
    # 选择距离最小的k个点
    class_Count = {}
    for i in range(k):
        min_label = train_label[sort_distance[i]]
        class_Count[min_label] = class_Count.get(min_label,0)+1
    result = sorted( class_Count.items(), key = operator.itemgetter(1), reverse = True)
    return result[0][0]

# 将图像数据转换为（1，1024）向量
def getData(filename):
    data =[]
    fr = open(filename)
    for i in range(32):
        line_Str = fr.readline()
        for j in range(32):
            data.append(int(line_Str[j]))
    return data

# 从文件名中解析分类标签的数字
def getLabel(fileName):
    label = int(fileName.split('_')[0])
    return label

# 构建训练集数据向量，及对应分类标签向量
def trainingDataSet():
    train_label = []
    training_File_List = listdir('trainingDigits')
    m = len(training_File_List)
    train_data = zeros((m,1024))
    # 获取训练集的标签
    for i in range(m):
        file_Name = training_File_List[i]   # fileName:训练集文件名
        train_label.append(getLabel(file_Name))  #获取训练集分类
        train_data[i,:] = getData('trainingDigits/%s' % file_Name)
    return train_label,train_data

#测试手写数字识别代码
def handWritingRec(k):
   # k = int(input('选取最邻近的K个值，K='))
    train_label,train_data = trainingDataSet()
    testFileList = listdir('testDigits')
    error_count = 0
    test_count = len(testFileList)
    for i in range(test_count):
        file_Name = testFileList[i]
        # 得到测试集标签
        classNumStr = getLabel(file_Name)
        test_data = getData('testDigits/%s' % file_Name)
        # 调用kNN分类进行预测
        result = kNNclassifier(test_data, train_data, train_label, k)
        print("Testing No.",i+1,"the classifier came back with:",result,"the real answer is:",classNumStr)
        if (result != classNumStr):
            error_count += 1.0
    print("\ntotal number of tests is:",test_count)
    print("the total number of errors is:",error_count)
    print("\nthe total error rate is:",error_count/float(test_count)*100,'%')
    return error_count

def selectKValue():
    x = list()
    y = list()
    for i in range(1, 10):
        x.append(int(i))
        y.append(int(handWritingRec(i)))
    plt.plot(x, y,marker='*')
    plt.xlabel("the value of k")
    plt.ylabel("the number of errors")
    # 程序执行时间比较长
    plt.show()

# 开始测试，绘制k值与误分类数折线图
selectKValue()
