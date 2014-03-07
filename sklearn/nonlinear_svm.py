#coding: utf-8
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.metrics import classification_report

N = 100

# 訓練データを作成
mean1 = [-1, 2]
mean2 = [1, -1]
mean3 = [4, -4]
mean4 = [-4, 4]
cov = [[1.0, 0.8], [0.8, 1.0]]

X = np.r_[np.random.multivariate_normal(mean1, cov, N / 4),
          np.random.multivariate_normal(mean3, cov, N / 4),
          np.random.multivariate_normal(mean2, cov, N / 4),
          np.random.multivariate_normal(mean4, cov, N / 4)]
t = [-1] * (N / 2) + [1] * (N / 2)

pl.figure(figsize=(18, 5))
no = 1

# それぞれのカーネルでSVMを学習
for kernel in ('linear', 'poly', 'rbf'):
    # 分類器を訓練
    clf = svm.SVC(kernel=kernel)
    clf.fit(X, t)

    pl.subplot(1, 3, no)
    cmap1 = ListedColormap(['red', 'blue'])
    cmap2 = ListedColormap(['#FFAAAA', '#AAAAFF'])

    # 訓練データをプロット
    pl.scatter(X[:, 0], X[:, 1], c=t, zorder=10, cmap=cmap1)
    pl.scatter(clf.support_vectors_[:, 0],
               clf.support_vectors_[:, 1],
               s=80, facecolors='none', zorder=10)

    # 決定境界をプロット
    xmin = ymin = -6
    xmax = ymax = 6
    XX, YY = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    pl.pcolormesh(XX, YY, Z > 0, cmap=cmap2)
    pl.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-0.5, 0, 0.5])

    predict_svm = clf.predict(X)
    print classification_report(t, predict_svm)

    pl.title("kernel: %s" % kernel)
    pl.xlim(xmin, xmax)
    pl.ylim(ymin, ymax)

    no += 1

pl.show()
