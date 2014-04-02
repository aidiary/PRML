#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
勾配降下法で多変数線形回帰モデル（重回帰モデル）の
パラメータ推定
"""

def featureNormalize(X):
    X_norm = X
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(len(mu)):
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]
    return X_norm, mu, sigma

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

def normalEqn(X, y):
    theta = np.linalg.inv(np.dot(X.T, X))
    theta = np.dot(theta, X.T)
    theta = np.dot(theta, y)
    return theta

if __name__ == "__main__":
    # 訓練データをロード
    # 0列目は部屋のサイズ
    # 1列目は寝室の数
    # 2列目は家の価格というデータ
    # 部屋のサイズと寝室の数から家の価格を推定する問題
    data = np.genfromtxt("ex1data2.txt", delimiter=",")
    X = data[:, (0,1)]
    y = data[:, 2]
    m = len(y)  # 訓練データ数

    # 部屋のサイズと寝室数ではスケールが違うため
    # 特徴量をスケーリング
    X, mu, sigma = featureNormalize(X)

    # 訓練データの1列目に1を追加
    X = np.hstack((np.ones((m, 1)), X))

    # パラメータを0で初期化
    # yとthetaはわざわざ縦ベクトルにreshapeしなくてもOK
    # np.dot()では自動的に縦ベクトル扱いして計算してくれる
    theta = np.zeros(3)
    iterations = 1500
    alpha = 0.01

    # 初期状態のコストを計算
    initialCost = computeCost(X, y, theta)
    print "initial cost:", initialCost

    # 勾配降下法でパラメータ推定
    thetaGD, J_history = gradientDescent(X, y, theta, alpha, iterations)
    print "theta (gradient descent):", thetaGD
    print "final cost:", J_history[-1]

    # 正規方程式による解法
    # 勾配降下法で求めたパラメータとほぼ同じ値が得られる
    thetaNE = normalEqn(X, y)
    print "theta (normal equation):", thetaNE

    # 1650sq-ftで3部屋寝室の価格を予測
    xs = np.array([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]])
    predict1 = np.dot(xs, thetaGD)
    predict2 = np.dot(xs, thetaNE)
    print "1650 sq-ft, 3 bedrooms => %f (gradient descent)" % predict1
    print "1650 sq-ft, 3 bedrooms => %f (normal equation)" % predict2

    # コストの履歴をプロット
    plt.figure(2)
    plt.plot(J_history)
    plt.xlabel("iteration")
    plt.ylabel("J(theta)")
    plt.show()
