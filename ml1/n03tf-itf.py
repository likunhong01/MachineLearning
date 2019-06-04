from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
from sklearn.preprocessing import MinMaxScaler  # 归一化
from sklearn.preprocessing import StandardScaler,Imputer  # 归一化
import numpy as np


def cutword():
    '''
    分词
    :return:
    '''
    c1 = jieba.cut('今天严鸿炜有病，明天严鸿炜也有病，后天可能会没病，但大部分人等不到明天晚上，所以不要放弃今天')
    c2 = jieba.cut('今天好热啊，明天也好热啊，李昆洪好帅啊，可是他为什么没有对象呢，为什么还不分手啊')
    c3 = jieba.cut('今天我想吃宫保鸡丁，明天我想吃鱼香肉丝，可是今天没有宫保鸡丁，明天也没有鱼香肉丝，所以不要吃了')
    # 转化为列表
    content1 = list(c1)
    content2 = list(c2)
    content3 = list(c3)
    # 转化为字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1,c2,c3

def tfidfvec():
    '''
    中文特征值化
    :return: None
    '''
    c1 ,c2,c3 = cutword()
    print(c1, c2,c3)
    tf = TfidfVectorizer()
    data = tf.fit_transform([c1,c2,c3])
    print(tf.get_feature_names())
    print(data.toarray())
    return None

def mm():
    '''
    归一化处理
    :return: None
    '''
    mm = MinMaxScaler()
    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
    print(data)
    return None

def stand():
    '''
    标准化缩放
    :return: None
    '''
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])
    print(data)


def im():
    '''
    缺失值处理
    :return: None
    '''
    # NaN, nan
    im = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data = im.fit_transform([[1,2],[np.nan, 3], [7,6]])
    print(data)
    return None

if __name__ == '__main__':
    # tfidfvec()
    # mm()
    # stand()
    im()