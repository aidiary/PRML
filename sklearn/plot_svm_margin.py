#coding: utf-8
import numpy as np
import pylab as pl
from sklearn import svm

"""
線形SVM
正則化に係わるパラメータC（ペナルティ）の影響
Cが小さい <=> さらに多くのデータを使ってマージンを計算（正則化大）
Cが大きい <=> 分類境界に近いデータのみ用いてマージンを計算（正則化小）
"""

# データを作成
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

pl.figure(figsize=(13, 5))
for i, name, penalty in ((1, 'unregularized', 1), (2, 'regularized', 0.05)):
    # 分類器を学習
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)

    # 分類境界を求める
    w = clf.coef_[0]
    a = - w[0] / w[1]
    b = clf.intercept_[0]
    xx = np.linspace(-5, 5)
    yy = a * xx - (b / w[1])

    # マージンを通る分類境界との平行線を求める
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin

    # 分類境界と平行線をプロット
    pl.subplot(1, 2, i)
    pl.plot(xx, yy, 'k-')
    pl.plot(xx, yy_down, 'k--')
    pl.plot(xx, yy_up, 'k--')

    # 訓練データをプロット
    pl.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=pl.cm.Paired)
    # サポートベクタを強調
    pl.scatter(clf.support_vectors_[:, 0],
               clf.support_vectors_[:, 1],
               s=80, facecolors='none', zorder=10)

    # カラープロット
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    pl.pcolormesh(XX, YY, Z, cmap=pl.cm.Paired)

    pl.title("%s (C = %.2f)" % (name, penalty))
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)

pl.show()
