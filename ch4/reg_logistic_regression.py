#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

"""
正則化ロジスティック回帰
高次元の特徴量を追加することで曲線でデータを分類する
交差エントロピー誤差関数の勾配降下法で解く
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

def computeCost(X, y, theta, lam):
    # 二乗誤差関数ではなく、交差エントロピー誤差関数を使用
    h = sigmoid(np.dot(X, theta))
    J = (1.0 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    # 正則化項を加える
    temp = theta
    temp[0] = 0  # 正則化項はtheta[0]は使わない
    J += lam / (2 * m) * np.sum(temp * temp)
    return J

def costFunction(theta, X, y, lam):
    # コスト関数を計算
    h = sigmoid(np.dot(X, theta))
    J = (1.0 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) + lam / (2 * m) * np.sum(theta[1:] ** 2)
    
    m = X.shape[0]

    def J(x, *args):
        theta = x
        lam = args[0]

    # その微分を計算
    def gradJ(x, *args):
        
    grad = np.zeros(theta.shape[0])
    grad[0] = (1.0 / m) * np.sum(h - y)
    grad[1:] = (1.0 / m) * np.dot(X[:,1:].T, h - y) + (lam / m) * theta[1:]

    return J, grad

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
    mapFeature(X[:, 0], X[:, 1])

    # パラメータを0で初期化
    initial_theta = np.zeros(X.shape[1])
    lam = 1.0

    # 初期パラメータから求めたコスト関数とその微分を返す
    cost, grad = costFunction(initial_theta, X, y, lam)

    # Conjugate Gradientでコスト関数を最適化するパラメータを求める
    res = scipy.optimize.fmin_cg(cost, initial_theta, fprime=grad)
    print res
    exit()

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
