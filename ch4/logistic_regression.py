#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

"""
ロジスティック回帰
交差エントロピー誤差関数の勾配降下法で解く
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

def computeCost(X, y, theta):
    # 二乗誤差関数ではなく、交差エントロピー誤差関数を使用
    h = sigmoid(np.dot(X, theta))
    J = (1.0 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)      # 訓練データ数
    J_history = []  # 各更新でのコスト
    for iter in range(iterations):
        # sigmoid関数を適用する点が線形回帰と異なる
        theta = theta - alpha * (1.0 / m) * np.dot(X.T, sigmoid(np.dot(X, theta)) - y)
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

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

    # パラメータを0で初期化
    theta = np.zeros(3)
    iterations = 300000
    alpha = 0.001

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
