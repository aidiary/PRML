#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""3.1 勾配降下法で1変数線形回帰モデルのパラメータ推定"""

def plotData(X, y):
    plt.scatter(X, y, c='red', marker='o', label="Training data")
    plt.xlabel("Population of city in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.xlim(4, 24)
    plt.ylim(-5, 25)

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
    # 訓練データをロード
    data = np.genfromtxt("ex1data1.txt", delimiter=",")
    X = data[:, 0]
    y = data[:, 1]
    m = len(y)  # 訓練データ数

    # 訓練データをプロット
    plt.figure(1)
    plotData(X, y)

    # 訓練データの1列目に1を追加
    X = X.reshape((m, 1))
    X = np.hstack((np.ones((m, 1)), X))

    # パラメータを0で初期化
    # yとthetaはわざわざ縦ベクトルにreshapeしなくてもOK
    # np.dot()では自動的に縦ベクトル扱いして計算してくれる
    theta = np.zeros(2)
    iterations = 1500
    alpha = 0.01

    # 初期状態のコストを計算
    initialCost = computeCost(X, y, theta)
    print "initial cost:", initialCost

    # 勾配降下法でパラメータ推定
    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

    # コストの履歴をプロット
    plt.figure(2)
    plt.plot(J_history)
    plt.xlabel("iteration")
    plt.ylabel("J(theta)")

    # 直線をプロット
    plt.figure(1)
    plt.plot(X[:,1], np.dot(X, theta), 'b-', label="Linear regression")
    plt.legend()

    # 35,000と70,000のときの予測値を計算
    predict1 = np.dot(np.array([1, 3.5]), theta)
    predict2 = np.dot(np.array([1, 7.0]), theta)
    print "3.5 => %f" % predict1
    print "7.0 => %f" % predict2

    # J(theta)の三次元プロット
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    gridsize = 200
    theta0_vals = np.linspace(-10, 10, gridsize)
    theta1_vals = np.linspace(-1, 4, gridsize)
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    J_vals = np.zeros((gridsize, gridsize))
    for i in range(gridsize):
        for j in range(gridsize):
            t = np.array([theta0_vals[i, j], theta1_vals[i, j]])
            J_vals[i, j] = computeCost(X, y, t)
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    plt.xlabel("theta_0")
    plt.ylabel("theta_1")

    # 等高線を描画
    fig = plt.figure(4)
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
    plt.xlabel("theta_0")
    plt.ylabel("theta_1")
    plt.show()
