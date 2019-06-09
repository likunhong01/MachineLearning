#coding=utf-8
#Version:python3.7.0
#Tools:Pycharm 2019
# Author:LIKUNHONG
__date__ = ''
__author__ = 'lkh'

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def naviebayes():
    '''
    朴素贝叶斯进行文本分类
    :return: None
    '''
    news = fetch_20newsgroups(subset='all')

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target,test_size=0.25)

    # 特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)

    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)

    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)

    print('文章类别为：', y_predict)
    # 得出准确率
    print('准确率为：', mlt.score(x_test, y_test))

if __name__ == '__main__':
    naviebayes()