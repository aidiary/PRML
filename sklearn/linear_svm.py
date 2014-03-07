#coding:utf-8
import numpy as np
import pylab as pl
from sklearn import svm
from matplotlib.colors import ListedColormap

N = 1000

# 訓練データを作成
mean1 = np.array([-1, 2])
mean2 = np.array([1, -1])
cov = np.array([[1.0, 0.8], [0.8, 1.0]])

np.random.seed(500)
X = np.r_[np.random.multivariate_normal(mean1, cov, N / 2),
          np.random.multivariate_normal(mean2, cov, N / 2)]
t = [0] * (N / 2) + [1] * (N / 2)

# 線形SVMを学習
clf = svm.SVC(kernel='linear', C=2)
clf.fit(X, t)

# 訓練データをプロット
cmap = ListedColormap(['red', 'blue'])
pl.scatter(X[:, 0], X[:, 1], c=t, cmap=cmap)

# サポートベクトルを強調
pl.scatter(clf.support_vectors_[:, 0],
           clf.support_vectors_[:, 1],
           s=80, facecolors='none')

# 識別境界と平行線をプロット
w = clf.coef_[0]
a = - w[0] / w[1]
b = clf.intercept_[0]
xx = np.linspace(-6, 6, 1000)
yy = a * xx - (b / w[1])

margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * margin
yy_up = yy - a * margin

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.xlim(-6, 6)
pl.ylim(-6, 6)

pl.show()


