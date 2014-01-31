#coding: utf-8
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import datasets, neighbors

"""
PRML 2.5.2 再近傍法
irisデータのK近傍法結果をプロット
"""

if __name__ == "__main__":
    # 近傍のいくつの点を見るか？
    K = 15

    # irisデータをロード
    iris = datasets.load_iris()

    # irisの特徴量は3次元だが2次元まで用いる
    # データを平面にプロットできるようにするため
    X = iris.data[:, :2]
    y = iris.target

    # メッシュのステップサイズ
    h = 0.02

    # カラーマップを作成
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # 学習器を訓練
    knn = neighbors.KNeighborsClassifier(K)
    knn.fit(X, y)

    # 座標範囲を決定
    # 各特徴量次元の最小、最大の値の+-1の範囲で表示
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 平面をhxhのサイズのグリッドに分割して各場所での分類結果を求める
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # グリッドの各場所の分類結果をカラープロット
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 訓練データをプロット
    # クラスによって色を変える
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.title("3-class classification (K = %i)" % K)
    pl.show()
