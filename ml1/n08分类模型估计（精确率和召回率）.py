#coding=utf-8
#Version:python3.7.0
#Tools:Pycharm 2019
# Author:LIKUNHONG
__date__ = ''
__author__ = 'lkh'

from sklearn.metrics import classification_report

print('每个类别的精确率和召回率', classification_report(y_test, y_predict, target_names=news.target_names))
