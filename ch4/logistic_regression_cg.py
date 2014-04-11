#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

"""
ロジスティック回帰
共役勾配法（conjugate gradient method）で解く
"""

def plotData(X, y):
    # positiveクラスのデータのインデックス
    positive = [i for i in range(len(y)) if y[i] == 1]
    # negativeクラスのデータのインデックス
    negative = [i for i in range(len(y)) if y[i] == 0]

    plt.scatter(X[positive, 0], X[positive, 1], c='red', marker='o', label="positive")
    plt.scatter(X[negative, 0], X[negative, 1], c='blue', marker='o', label="negative")

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def J(theta, *args):
    """コスト関数"""
    def safe_log(x, minval=0.0000000001):
        return np.log(x.clip(min=minval))
    X, y = args
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    return (1.0 / m) * np.sum(-y * safe_log(h) - (1 - y) * safe_log(1 - h))

def gradient(theta, *args):
    """コスト関数Jの偏微分"""
    X, y = args
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = (1.0 / m) * np.dot(X.T, h - y)
    return grad

if __name__ == "__main__":
    # 訓練データをロード
    data = np.genfromtxt("ex2data1.txt", delimiter=",")
    X = data[:, (0, 1)]
    y = data[:, 2]
    # 訓練データ数
    m = len(y)

    # 訓練データをプロット
    plt.figure(1)
    plotData(X, y)

    # 訓練データの1列目に1を追加
    X = X.reshape((m, 2))
    X = np.hstack((np.ones((m, 1)), X))

    # パラメータを0で初期化;
    initial_theta = np.zeros(3)

    # 初期状態のコストを計算
    print "initial cost:", J(initial_theta, X, y)

    # Conjugate Gradientでパラメータ推定
    theta = optimize.fmin_cg(J, initial_theta, fprime=gradient, args=(X, y))
    print "theta:", theta
    print "final cost:", J(theta, X, y)

    # 決定境界を描画
    plt.figure(1)
    xmin, xmax = min(X[:,1]), max(X[:,1])
    xs = np.linspace(xmin, xmax, 100)
    ys = [- (theta[0] / theta[2]) - (theta[1] / theta[2]) * x for x in xs]
    plt.plot(xs, ys, 'b-', label="decision boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim((30, 100))
    plt.ylim((30, 100))
    plt.legend()
    plt.show()
