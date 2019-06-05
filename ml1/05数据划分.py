#coding=utf-8
#Version:python3.7.0
#Tools:Pycharm 2019
# Author:LIKUNHONG
__date__ = ''
__author__ = 'lkh'

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

li = load_iris()
print('获取特征值')
print(li.data)
print('目标值')
print(li.target)

# 第一个，数据集的特征值，第二个是目标值，第三个是指定测试集的大小，
# 注意返回值是包含训练集（train x_train表示特征值 y_train表示目标值）和测试集(test x_test y_test)，
x_train, y_train, x_test, y_test = train_test_split()   # 顺序不能错
'''这里其实就是拿样本序列，乱序一部分分成了测试集一部分训练集'''


# 用于分类的大数据集
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
print(news.data)
print(news.target)


# 回归数据集
from sklearn.datasets import load_boston
lb = load_boston()
print('获取特征值')
print(lb.data)
print('目标值')
print(lb.target)