#coding:utf-8
import numpy as np
from pylab import *

"""
カーネル回帰：カーネル法を利用した曲線フィッティング
"""

N = 10
BETA = 1

# カーネル関数
def kernel(x1, x2):
    return np.exp(-BETA * (x1-x2)**2)

def y(x, xlist, alpha):
    summation = 0
    for i in range(len(xlist)):
        summation += alpha[i] * kernel(xlist[i], x)
    return summation

if __name__ == "__main__":
    # 訓練データを作成
    xlist = np.linspace(0, 1, N)
    tlist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)

    # 行列Kを作成
    K = zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(xlist[j], xlist[i])

    # 重み係数αを計算
    alpha = np.dot(np.linalg.inv(K), tlist)
    print alpha

    # 連続関数のプロット用X値
    xs = np.linspace(0, 1, 1000)
    ideal = np.sin(2 * np.pi * xs)    # 理想曲線
    model = [y(x, xlist, alpha) for x in xs]

    # グラフを描画
    plot(xlist, tlist, 'bo')  # 訓練データ
    plot(xs, ideal, 'g-')     # 理想曲線
    plot(xs, model, 'r-')     # 推定モデル
    xlim(0.0, 1.0)
    ylim(-1.5, 1.5)
    show()
