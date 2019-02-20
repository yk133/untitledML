from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import sklearn.neighbors
import numpy as np


def func():
    """
    字典数据获取
    :return:None
    """

    dict = DictVectorizer()

    ans = dict.fit_transform([{'city': '西安', 'tp': 159}, {'city': '深圳', 'tp': 30}, {'city': '北京', 'tp': 100}])

    print(dict.get_feature_names())

    print(ans)

    return None

def start():
    """
    kaishi
    :return:  None
    """



    return None


if __name__ == '__main__':
    # func()
    start()


