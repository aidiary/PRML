#coding:utf-8
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

"""k-NNのカラーマップ表示"""

n_neighbors = 15

# irisデータをインポート
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# メッシュのステップサイズ
h = 0.02

# カラーマップ
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # 分類器を学習
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # データより少し広くなるように描画範囲を決定
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 範囲をメッシュに区切ってその座標での分類結果Zを求める
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 分類結果を元に色を割り当てて描画
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 訓練データをプロット
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.title("3-Class classification (k = %i, weights = '%s')"
             % (n_neighbors, weights))
pl.show()

