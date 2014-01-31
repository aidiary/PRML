#coding:utf-8
import numpy as np
from scipy.linalg import norm
import cvxopt
import cvxopt.solvers
from pylab import *

"""
線形SVM
cvxoptのQuadratic Programmingを解く関数を使用
"""

N = 100  # データ数

def f(x1, w, b):
    return - (w[0] / w[1]) * x1 - (b / w[1])

def kernel(x, y):
    return np.dot(x, y)  # 線形カーネル

if __name__ == "__main__":
    # 訓練データを作成
    cls1 = []
    cls2 = []

    mean1 = [-1, 2]
    mean2 = [1, -1]
    cov = [[1.0,0.8], [0.8, 1.0]]

    cls1.extend(np.random.multivariate_normal(mean1, cov, N / 2))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N / 2))

    # データ行列Xを作成
    X = vstack((cls1, cls2))

    # ラベルtを作成
    t = []
    for i in range(N / 2):
        t.append(1.0)   # クラス1
    for i in range(N / 2):
        t.append(-1.0)  # クラス2
    t = array(t)

    # ラグランジュ乗数を二次計画法（Quadratic Programming）で求める
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = t[i] * t[j] * kernel(X[i], X[j])

    Q = cvxopt.matrix(K)
    p = cvxopt.matrix(-np.ones(N))             # -1がN個の列ベクトル
    G = cvxopt.matrix(np.diag([-1.0] * N))     # 対角成分が-1のNxN行列
    h = cvxopt.matrix(np.zeros(N))             # 0がN個の列ベクトル
    A = cvxopt.matrix(t, (1, N))               # N個の教師信号が要素の行ベクトル（1xN）
    b = cvxopt.matrix(0.0)                     # 定数0.0
    sol = cvxopt.solvers.qp(Q, p, G, h, A, b)  # 二次計画法でラグランジュ乗数aを求める
    a = array(sol['x']).reshape(N)             # 'x'がaに対応する
    print a

    # サポートベクトルのインデックスを抽出
    S = []
    for i in range(len(a)):
        if a[i] < 0.00001: continue
        S.append(i)

    # wを計算
    w = np.zeros(2)
    for n in S:
        w += a[n] * t[n] * X[n]

    # bを計算
    summation = 0
    for n in S:
        temp = 0
        for m in S:
            temp += a[m] * t[m] * kernel(X[n], X[m])
        summation += (t[n] - temp)
    b = summation / len(S)
    print S, b

    # 訓練データを描画
    x1, x2 = np.array(cls1).transpose()
    plot(x1, x2, 'rx')

    x1, x2 = np.array(cls2).transpose()
    plot(x1, x2, 'bx')

    # サポートベクトルを描画
    for n in S:
        scatter(X[n,0], X[n,1], s=80, c='c', marker='o')

    # 識別境界を描画
    x1 = np.linspace(-6, 6, 1000)
    x2 = [f(x, w, b) for x in x1]
    plot(x1, x2, 'g-')

    xlim(-6, 6)
    ylim(-6, 6)
    show()
