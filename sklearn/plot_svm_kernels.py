#coding:utf-8
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import svm

# データセットを作成
X = np.c_[(0.4, -0.7),
          (-1.5, -1),
          (-1.4, -0.9),
          (-1.3, -1.2),
          (-1.1, -0.2),
          (-1.2, -0.4),
          (-0.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          (1.3, 0.8),
          (1.2, 0.5),
          (0.2, -2.0),
          (0.5, -2.4),
          (0.2, -2.3),
          (0.0, -2.7),
          (1.3, 2.1)].T
Y = [0] * 8 + [1] * 8

fignum = 1

pl.figure(figsize=(18, 5))

for kernel in ('linear', 'poly', 'rbf'):
    # 分類器を訓練
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)

    pl.subplot(1, 3, fignum)

    cmap = ListedColormap(['red', 'blue'])

    # 訓練データをプロット
    pl.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=cmap)

    # サポートベクトルを強調
    pl.scatter(clf.support_vectors_[:, 0],
               clf.support_vectors_[:, 1],
               s=80, facecolors='none', zorder=10)

    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    # 識別境界をプロット
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    # decision_function()は識別境界までの距離を返す
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    pl.pcolormesh(XX, YY, Z > 0, cmap=cmap)
    pl.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-0.5, 0, 0.5])

    pl.title("kernel: %s" % kernel)
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)

    fignum += 1

pl.show()
