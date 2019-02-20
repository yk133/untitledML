from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import sklearn.neighbors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

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

def KNNcls():
    """
    k近邻
    :return:  None
    """
    # 读取数据

    data = pd.read_csv("./data/train.csv")

    print(data.head(10))
    # 处理数据
    # 1 缩小数据
    d = data.query("x<1.4 & x> 1.2 & y > 1.0 & y < 1.3")
    # 2 事件处理
    time_values = pd.to_datetime(data["time"], unit='s')
    # 3 时间转换u
    time_values = pd.DatetimeIndex(time_values)
    # 4 构造特征
    data['day'] = time_values.day
    data['hour'] = time_values.hour
    data['year'] = time_values.year
    print(data.head(10))
    # 5 把原来的时间删除:
    data = data.drop(['time'], axis=1)
    # print(data.head(10))
    # 把签到数小鱼目标数的清理
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 0].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]
    #print(data)
    y= data['place_id']

    x= data.drop(['place_id'], axis=1)
    x= x.drop(['row_id'], axis=1)

    print("3333333\n",x)
    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 特征工程
    std = StandardScaler

    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    knn =KNeighborsClassifier(n_neighbors=5)

    knn.fit(x_train, y_train)

    y_predict = knn.predict(x_test)

    print("预测目标值情况:", y_predict)
    print("准确率",knn.score(x_test,y_test))

    return None


if __name__ == '__main__':
    # func()
    KNNcls()


