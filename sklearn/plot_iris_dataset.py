#coding:utf-8
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# irisデータの主成分分析による可視化
# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

# irisデータの最初に二次元のみ平面座標にプロット
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5
y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

pl.figure(2, figsize=(8, 6))
pl.clf()

pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')
pl.xlim(x_min, x_max)
pl.ylim(y_min, y_max)

# 主成分分析により最初の第3主成分までプロット
fig = pl.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y, cmap=pl.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")

pl.show()
