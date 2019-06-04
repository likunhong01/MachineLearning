from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# # 特征抽取
# from sklearn.feature_extraction.text import CountVectorizer
#
# # 实例化
# vector = CountVectorizer()
# # 调用fit_transform输入并转换数据
# res = vector.fit_transform(['life is short', 'i like python'])
# print(vector.get_feature_names())
# print(res.toarray())

def dictvec():
    '''
    字典数据抽取
    :return: None
    '''
    # 实例化
    dict = DictVectorizer()
    # 调用fit_transform
    data = dict.fit_transform()

    print(dict.get_feature_names())
    print(dict.inverse_transform(data))
    print(data)
    return None

def countvec():
    '''
    对文本进行特征值化
    :return: None
    '''
    cv = CountVectorizer()

    data = cv.fit_transform(['life is short', 'i like python'])

    # 统计所有文章当中的此，不记重复
    print(cv.get_feature_names())
    # 把每一篇文章对应的单词出现个数找到，单个字母不统计
    print(data.toarray())

    data2 = cv.fit_transform(['人生 苦短，我 喜欢 python', '人生 漫长，我 不喜欢 python'])
    print(cv.get_feature_names())
    print(data2.toarray())

    return None

if __name__ == '__main__':
    countvec()


