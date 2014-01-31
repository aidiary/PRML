#coding:utf-8
import numpy as np
from pylab import *

"""3クラスに拡張した最小二乗による分類（図4.5）"""

K = 3    # 3クラス分類
N = 150  # データ数

def f1(x1, W_t):
    # クラス1とクラス2の決定境界の直線の方程式
    a = - ((W_t[0,1] - W_t[1,1]) / (W_t[0,2] - W_t[1,2]))
    b = - ((W_t[0,0] - W_t[1,0]) / (W_t[0,2] - W_t[1,2]))
    return a * x1 + b

def f2(x1, W_t):
    # クラス2とクラス3の決定境界の直線の方程式
    a = - ((W_t[1,1] - W_t[2,1]) / (W_t[1,2] - W_t[2,2]))
    b = - ((W_t[1,0] - W_t[2,0]) / (W_t[1,2] - W_t[2,2]))
    return a * x1 + b

if __name__ == "__main__":
    # 訓練データを作成
    cls1 = []
    cls2 = []
    cls3 = []

    mean1 = [-2, 2]  # クラス1の平均
    mean2 = [0, 0]   # クラス2の平均
    mean3 = [2, -2]  # クラス3の平均
    cov = [[1.0,0.8], [0.8,1.0]]  # 共分散行列（全クラス共通）

    # データを作成
    cls1.extend(np.random.multivariate_normal(mean1, cov, N / 3))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N / 3))
    cls3.extend(np.random.multivariate_normal(mean3, cov, N / 3))

    # データ行列Xを作成
    temp = vstack((cls1, cls2, cls3))
    temp2 = ones((N, 1))  # バイアスw_0用に1を追加
    X = hstack((temp2, temp))

    # ラベル行列Tを作成
    T = []
    for i in range(N / 3):
        T.append(array([1, 0, 0]))
    for i in range(N / 3):
        T.append(array([0, 1, 0]))
    for i in range(N / 3):
        T.append(array([0, 0, 1]))
    T = array(T)

    # パラメータ行列Wを最小二乗法で計算
    X_t = np.transpose(X)
    temp = np.linalg.inv(np.dot(X_t, X))  # 行列の積はnp.dot(A, B)
    W = np.dot(np.dot(temp, X_t), T)
    W_t = np.transpose(W)

    # 訓練データを描画
    x1, x2 = np.transpose(np.array(cls1))
    plot(x1, x2, 'rx')
    x1, x2 = np.transpose(np.array(cls2))
    plot(x1, x2, 'g+')
    x1, x2 = np.transpose(np.array(cls3))
    plot(x1, x2, 'bo')

    # クラス1とクラス2の間の識別境界を描画
    x1 = np.linspace(-6, 6, 1000)
    x2 = [f1(x, W_t) for x in x1]
    plot(x1, x2, 'r-')

    # クラス2とクラス3の間の識別境界を描画
    x2 = [f2(x, W_t) for x in x1]
    plot(x1, x2, 'b-')

    xlim(-6, 6)
    ylim(-6, 6)
    show()
