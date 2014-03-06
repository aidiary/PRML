#coding:utf-8
import numpy as np
import pylab as pl
from sklearn import linear_model, datasets

"""ロジスティック回帰によるirisデータの分類"""

# irisデータをロード
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

# メッシュサイズ
h = 0.02

# 分類器を学習
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)

# データより少し広くなるように描画範囲を決定
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 範囲をメッシュに区切ってその座標での分類結果Zを求める
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# メッシュの各点を描画
pl.figure(1)
pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)

# 訓練データをプロット
pl.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=pl.cm.Paired)
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())

pl.show()

