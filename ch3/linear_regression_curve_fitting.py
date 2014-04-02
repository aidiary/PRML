#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
勾配降下法で線形回帰による曲線フィッティング
PRML1章の多項式曲線フィッティングと同じだが、
正規方程式ではなく、勾配降下法で解く点が異なる
"""

def plotData(X, y):
    plt.scatter(X, y, c='red', marker='o', label="Training data")
    plt.xlabel("x")
    plt.ylabel("y")

def computeCost(X, y, theta):
    m = len(y)  # 訓練データ数
    tmp = np.dot(X, theta) - y
    J = 1.0 / (2 * m) * np.dot(tmp.T, tmp)
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)      # 訓練データ数
    J_history = []  # 各更新でのコスト
    for iter in range(iterations):
        theta = theta - alpha * (1.0 / m) * np.dot(X.T, np.dot(X, theta) - y)
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

if __name__ == "__main__":
    # 訓練データを作成
    # sin(2 * pi * x) の値にガウス分布に従う小さなランダムノイズを加える
    X = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, X.size)

    # 訓練データ数
    m = len(y)

    # 訓練データをプロット
    plt.figure(1)
    plotData(X, y)

    # 訓練データの1列目に1を追加
    # 元の変数の二乗、三乗の項も加える
    # 多項式回帰になり曲線フィッティングも可能
    X = X.reshape((m, 1))
    X = np.hstack((np.ones((m, 1)), X, X**2, X**3))

    # パラメータを0で初期化
    # yとthetaはわざわざ縦ベクトルにreshapeしなくてもOK
    # np.dot()では自動的に縦ベクトル扱いして計算してくれる
    theta = np.zeros(4)
    iterations = 100000
    alpha = 0.2

    # 初期状態のコストを計算
    initialCost = computeCost(X, y, theta)
    print "initial cost:", initialCost

    # 勾配降下法でパラメータ推定
    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
    print "theta:", theta
    print "final cost:", J_history[-1]

    # コストの履歴をプロット
    plt.figure(2)
    plt.plot(J_history)
    plt.xlabel("iteration")
    plt.ylabel("J(theta)")

    # 曲線をプロット
    plt.figure(1)
    plt.plot(X[:, 1], np.dot(X, theta), 'b-', label="Linear regression")
    plt.xlim((0, 1))
    plt.legend()
    plt.show()
