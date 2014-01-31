#coding:utf-8
import numpy as np
from pylab import *

"""4.3.2 ロジスティック回帰"""

N = 100  # データ数

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def f(x1, w):
    # 決定境界の直線の方程式
    # p(C1|X) > 0.5 -> XをC1に分類
    # p(C1|X) < 0.5 -> XをC2に分類
    # p(C1|X) = 0.5が決定境界 <-> シグモイド関数の引数が0のとき（図4.9）
    a = - (w[1] / w[2])
    b = - (w[0] / w[2])
    return a * x1 + b

if __name__ == "__main__":
    # 訓練データを作成
    cls1 = []
    cls2 = []

    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [8, -6]
    cov = [[1.0,0.8], [0.8,1.0]]

    # ノイズありデータ作成
    cls1.extend(np.random.multivariate_normal(mean1, cov, N / 2))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N / 2 - 20))
    cls2.extend(np.random.multivariate_normal(mean3, cov, 20))

    # データ行列Xを作成
    temp = vstack((cls1, cls2))
    temp2 = ones((N, 1))
    X = hstack((temp2, temp))

    # ラベルTを作成（1-of-K表現ではないので注意）
    t = []
    for i in range(N / 2):
        t.append(1.0)
    for i in range(N / 2):
        t.append(0.0)
    t = array(t)

    # パラメータwをIRLSで更新
    turn = 0
    w = array([0.0, 0.0, 0.0])  # 適当な初期値
    while True:
        # ファイを計算（恒等式とするのでデータ行列Xをそのまま使う）
        phi = X

        # Rとyを計算
        R = np.zeros((N, N))
        y = []
        for n in range(N):
            a = np.dot(w, phi[n,])
            y_n = sigmoid(a)
            R[n, n] = y_n * (1 - y_n)
            y.append(y_n)

        # ヘッセ行列Hを計算
        phi_T = phi.T
        H = np.dot(phi_T, np.dot(R, phi))

        # wを更新
        w_new = w - np.dot(np.linalg.inv(H), np.dot(phi_T, y-t))

        # wの収束判定
        diff = np.linalg.norm(w_new - w) / np.linalg.norm(w)
        print turn, diff
        if diff < 0.1: break

        w = w_new
        turn += 1

    # 訓練データを描画
    x1, x2 = np.array(cls1).transpose()
    plot(x1, x2, 'rx')

    x1, x2 = np.array(cls2).transpose()
    plot(x1, x2, 'bo')

    # 識別境界を描画
    x1 = np.linspace(-6, 10, 1000)
    x2 = [f(x, w) for x in x1]
    plot(x1, x2, 'g-')

    xlim(-6, 10)
    ylim(-10, 6)
    show()
