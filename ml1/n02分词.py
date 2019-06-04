from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba

def cutword():
    '''
    分词
    :return:
    '''
    c1 = jieba.cut('今天严鸿炜有病，明天严鸿炜也有病，后天可能会没病，但大部分人等不到明天晚上，所以不要放弃今天')
    # 转化为列表
    content1 = list(c1)
    # 转化为字符串
    c1 = ' '.join(content1)

    return c1

def hanzivec():
    '''
    中文特征值化
    :return: None
    '''
    c1 = cutword()
    print(c1)
    cv = CountVectorizer()
    data = cv.fit_transform([c1])
    print(cv.get_feature_names())
    print(data.toarray())
    return None


if __name__ == '__main__':
    hanzivec()


