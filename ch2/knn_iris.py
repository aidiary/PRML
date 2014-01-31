#coding: utf-8
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

"""
irisデータをK近傍法で分類する
scikit-learnのチュートリアルから
http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
"""

if __name__ == "__main__":
    # irisデータをロード
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target

    # 訓練データとテストデータに分ける
    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    iris_X_train = iris_X[indices[:-10]]
    iris_y_train = iris_y[indices[:-10]]
    iris_X_test = iris_X[indices[-10:]]
    iris_y_test = iris_y[indices[-10:]]

    # K近傍法の学習器を作成
    knn = KNeighborsClassifier()

    # 訓練データを用いて学習器を訓練
    knn.fit(iris_X_train, iris_y_train)

    # テストデータを予測
    predict = knn.predict(iris_X_test)
    print predict
    print iris_y_test
