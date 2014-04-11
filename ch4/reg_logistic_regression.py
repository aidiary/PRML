#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

"""
正則化ロジスティック回帰
高次元の特徴量を追加することで曲線でデータを分類する
共役勾配法で解く
"""

def plotData(X, y):
    # positiveクラスのデータのインデックス
    positive = [i for i in range(len(y)) if y[i] == 1]
    # negativeクラスのデータのインデックス
    negative = [i for i in range(len(y)) if y[i] == 0]

    plt.scatter(X[positive, 0], X[positive, 1], c='red', marker='o', label="positive")
    plt.scatter(X[negative, 0], X[negative, 1], c='blue', marker='o', label="negative")

def mapFeature(X1, X2):
    degree = 6

    # データ行列に1を追加
    m = X1.shape[0]
    X = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            newX = (X1 ** (i - j) * X2 ** j).reshape((m, 1))
            X = np.hstack((X, newX))
    return X

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def J(theta, *args):
    def safe_log(x, minval=0.0000000001):
        return np.log(x.clip(min=minval))
    X, y, lam = args
    # 二乗誤差関数ではなく、交差エントロピー誤差関数を使用
    h = sigmoid(np.dot(X, theta))
    return (1.0 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) + lam / (2 * m) * np.sum(theta[1:] ** 2)

def gradient(theta, *args):
    X, y, lam = args
    h = sigmoid(np.dot(X, theta))
    grad = np.zeros(theta.shape[0])
    grad[0] = (1.0 / m) * np.sum(h - y)
    grad[1:] = (1.0 / m) * np.dot(X[:,1:].T, h - y) + (lam / m) * theta[1:]
    return grad

if __name__ == "__main__":
    # 訓練データをロード
    data = np.genfromtxt("ex2data2.txt", delimiter=",")
    X = data[:, (0, 1)]
    y = data[:, 2]
    # 訓練データ数
    m = len(y)

    # 訓練データをプロット
    plt.figure(1)
    plotData(X, y)

    # 特徴量のマッピング
    # 元の特徴量の多項式項を追加
    # 1列目の1も追加する
    X = mapFeature(X[:, 0], X[:, 1])

    # パラメータを0で初期化
    initial_theta = np.zeros(X.shape[1])
    lam = 1.0

    # 初期状態のコストを計算
    print "initial cost:", J(initial_theta, X, y, lam)

    # Conjugate Gradientでコスト関数を最適化するパラメータを求める
    theta = optimize.fmin_cg(J, initial_theta, fprime=gradient, args=(X, y, lam), gtol=1e-10)
    print "theta:", theta
    print "final cost:", J(theta, X, y, lam)

    # 決定境界を描画
    plt.figure(1)
    gridsize = 100
    x1_vals = np.linspace(-1, 1.5, gridsize)
    x2_vals = np.linspace(-1, 1.5, gridsize)
    x1_vals, x2_vals = np.meshgrid(x1_vals, x2_vals)
    z = np.zeros((gridsize, gridsize))
    for i in range(gridsize):
        for j in range(gridsize):
            x1 = np.array([x1_vals[i, j]])
            x2 = np.array([x2_vals[i, j]])
            z[i, j] = np.dot(mapFeature(x1, x2), theta)
    # 決定境界はsigmoid(z)=0.5、すなわちz=0の場所
    plt.contour(x1_vals, x2_vals, z, levels=[0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim((-1, 1.5))
    plt.ylim((-1, 1.5))
    plt.legend()
    plt.show()
