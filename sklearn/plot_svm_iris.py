#coding:utf-8
import numpy as np
import pylab as pl
from sklearn import svm, datasets

"""
線形SVMでirisデータを分類
"""

# データをロード
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

# 分類器を学習
clf = svm.SVC(C=1.0, kernel='linear')
clf.fit(X, Y)

pl.figure(1)

# メッシュの各点の分類結果を描画
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)

# 訓練データをプロット
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())

pl.show()
