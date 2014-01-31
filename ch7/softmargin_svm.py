#coding:utf-8
import numpy as np
from scipy.linalg import norm
import cvxopt
import cvxopt.solvers
from pylab import *

"""
非線形SVM（ソフトマージン）
cvxoptのQuadratic Programmingを解く関数を使用
"""

P = 3           # 多項式カーネルのパラメータ
N = 200         # データ数
C = 1000        # スラック変数を用いて表されるペナルティとマージンのトレードオフパラメータ
SIGMA = 0.6     # ガウスカーネルのパラメータ

# 多項式カーネル
def polynomial_kernel(x, y):
    return (1 + np.dot(x, y)) ** P

# ガウスカーネル
def gaussian_kernel(x, y):
    return np.exp(-norm(x-y)**2 / (2 * (SIGMA ** 2)))

# どちらのカーネル関数を使うかここで指定
kernel = gaussian_kernel

def f(x, a, t, X, b):
    summation = 0.0
    for n in range(N):
        summation += a[n] * t[n] * kernel(x, X[n])
    return summation + b

if __name__ == "__main__":
    # 訓練データをロード
    data = np.genfromtxt("classification.txt")
    X = data[:,0:2]
    t = data[:,2] * 2 - 1.0  # 教師信号を-1 or 1に変換

    # ラグランジュ乗数を二次計画法（Quadratic Programming）で求める
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = t[i] * t[j] * kernel(X[i], X[j])

    Q = cvxopt.matrix(K)
    p = cvxopt.matrix(-np.ones(N))
    temp1 = np.diag([-1.0] * N)
    temp2 = np.identity(N)
    G = cvxopt.matrix(np.vstack((temp1, temp2)))
    temp1 = np.zeros(N)
    temp2 = np.ones(N) * C
    h = cvxopt.matrix(np.hstack((temp1, temp2)))
    A = cvxopt.matrix(t, (1, N))
    b = cvxopt.matrix(0.0)
    sol = cvxopt.solvers.qp(Q, p, G, h, A, b)
    a = array(sol['x']).reshape(N)
    print a

    # サポートベクトルのインデックスを抽出
    S = []
    M = []
    for n in range(len(a)):
        # 0.001より小さいのは0とみなしてサポートベクトルとみなさない
        if a[n] > 0.001:
            S.append(n)
        if 0.001 < a[n] < C:
            M.append(n)

    # bを計算
    summation = 0
    for n in M:
        temp = 0
        for m in S:
            temp += a[m] * t[m] * kernel(X[n], X[m])
        summation += (t[n] - temp)
    b = summation / len(M)

    print S, b

    # 訓練データを描画
    for n in range(N):
        if t[n] > 0:
            scatter(X[n,0], X[n,1], c='b', marker='o')
        else:
            scatter(X[n,0], X[n,1], c='r', marker='o')

    # サポートベクトルを描画
    for n in S:
        scatter(X[n,0], X[n,1], s=80, c='c', marker='o')

    # 識別境界を描画
    X1, X2 = meshgrid(linspace(-2,2,50), linspace(-2,2,50))
    w, h = X1.shape
    X1.resize(X1.size)
    X2.resize(X2.size)
    Z = array([f(array([x1, x2]), a, t, X, b) for (x1, x2) in zip(X1, X2)])
    X1.resize((w, h))
    X2.resize((w, h))
    Z.resize((w, h))
    CS = contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')

    for n in S:
        print f(X[n], a, t, X, b)

    xlim(-2, 2)
    ylim(-2, 2)
    show()
