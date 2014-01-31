#coding:utf-8
import numpy as np
from pylab import *

# 4.3.4 多クラスロジスティック回帰

N = 300  # データ数
M = 3    # パラメータの次数
K = 3    # クラス数

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def f(x1, W_t, c1, c2):
    # クラスc1とクラスc2の決定境界の直線の方程式
    a = - ((W_t[c1, 1] - W_t[c2, 1]) / (W_t[c1, 2] - W_t[c2, 2]))
    b = - ((W_t[c1, 0] - W_t[c2, 0]) / (W_t[c1, 2] - W_t[c2, 2]))
    return a * x1 + b

if __name__ == "__main__":
    # 訓練データの作成
    cls1 = []
    cls2 = []
    cls3 = []

    mean1 = [-2, 2]  # クラス1の平均
    mean2 = [0, 0]   # クラス2の平均
    mean3 = [2, -2]   # クラス3の平均
    cov = [[1.0,0.8], [0.8,1.0]]  # 共分散行列（全クラス共通）

    # データを作成
    cls1.extend(np.random.multivariate_normal(mean1, cov, N / 3))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N / 3))
    cls3.extend(np.random.multivariate_normal(mean3, cov, N / 3))

    # データ行列Xを作成
    temp = vstack((cls1, cls2, cls3))
    temp2 = ones((N, 1))
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

    # パラメータ行列WをIRLSで更新
    turn = 0
    W = np.zeros((M, K))
    while True:
        # ファイを計算（恒等式とするのでデータ行列Xをそのまま使う）
        phi = X

        # 予測値の行列Yを計算
        Y = np.zeros((N, K))
        for n in range(N):
            denominator = 0.0
            for k in range(K):
                denominator += np.exp(np.dot(W[:,k], X[n,:]))
            for k in range(K):
                Y[n, k] = np.exp(np.dot(W[:,k], X[n,:])) / denominator
        print Y

        # ヘッセ行列Hを計算
        I = np.identity(K)
        H = np.zeros((K * K, M, M))
        for j in range(K):
            for k in range(K):
                # (4.110)に従った計算
                for n in range(N):
                    temp = Y[n, k] * (I[k, j] - Y[n,j])
                    H[k + j * K] += temp * matrix(phi)[n].reshape(M,1) * matrix(phi)[n].reshape(1,M)  # 縦ベクトルx横ベクトル
                # もしくは下のように行列計算でもできる
#                Ijk = 1 if j == k else 0
#                R = np.diag(Y[:,k] * (Ijk - Y[:,j]))
#                H[k + j * K] = np.dot(phi.T, np.dot(R, phi))

        # 重み行列を更新
        W_new = np.zeros((M, K))
        phi_T = phi.T
        for i in range(K):
            temp = np.dot(phi_T, Y[:,i] - T[:,i])
            W_new[:,i] = W[:,i] - np.dot(np.linalg.inv(H[i + i * K]), temp)

        # Wの収束判定
        diff = np.linalg.norm(W_new - W) / np.linalg.norm(W)
        print turn, diff
        if diff < 0.1: break

        W = W_new
        turn += 1

    W_t = np.transpose(W)
    print W_t

    # 訓練データを描画
    x1, x2 = np.transpose(np.array(cls1))
    plot(x1, x2, 'rx')
    x1, x2 = np.transpose(np.array(cls2))
    plot(x1, x2, 'g+')
    x1, x2 = np.transpose(np.array(cls3))
    plot(x1, x2, 'bo')

    # クラス1とクラス2の間の識別境界を描画
    x1 = np.linspace(-6, 6, 1000)
    x2 = [f(x, W_t, 0, 1) for x in x1]
    plot(x1, x2, 'r-')

    # クラス2とクラス3の間の識別境界を描画
    x1 = np.linspace(-6, 6, 1000)
    x2 = [f(x, W_t, 1, 2) for x in x1]
    plot(x1, x2, 'b-')

    xlim(-6, 6)
    ylim(-6, 6)
    show()
