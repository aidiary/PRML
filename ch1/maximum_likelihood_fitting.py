#coding:utf-8
import numpy as np
from pylab import *

# M次多項式近似
M = 3

def y(x, wlist):
    ret = wlist[0]
    for i in range(1, M + 1):
        ret += wlist[i] * (x ** i)
    return ret

def estimate(xlist, tlist):
    A = []
    for i in range(M + 1):
        for j in range(M + 1):
            temp = (xlist ** (i + j)).sum()
            A.append(temp)
    A = array(A).reshape(M + 1, M + 1)

    T = []
    for i in range(M + 1):
        T.append(((xlist ** i) * tlist).sum())
    T = array(T)

    # パラメータwは線形連立方程式の解
    wlist = np.linalg.solve(A, T)

    return wlist

def main():
    # 訓練データ
    xlist = np.linspace(0, 1, 10)
    tlist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)

    # 訓練データからパラメータw_mlを推定
    w_ml = estimate(xlist, tlist)
    print w_ml

    # 精度パラメータを推定
    N = len(xlist)
    sums = 0
    for n in range(N):
        sums += (y(xlist[n], w_ml) - tlist[n]) ** 2
    beta_inv = 1.0 / N * sums

    # 連続関数のプロット用X値
    xs = np.linspace(0, 1, 500)
    ideal = np.sin(2 * np.pi * xs)
    means = []
    uppers = []
    lowers = []
    for x in xs:
        m = y(x, w_ml)      # 予測分布の平均
        s = sqrt(beta_inv)  # 予測分布の標準偏差
        u = m + s           # 平均 + 標準偏差
        l = m - s           # 平均 - 標準偏差
        means.append(m)
        uppers.append(u)
        lowers.append(l)

    plot(xlist, tlist, 'bo')  # 訓練データ
    plot(xs, ideal, 'g-')     # 理想曲線
    plot(xs, means, 'r-')     # 予測モデルの平均
    plot(xs, uppers, 'r--')
    plot(xs, lowers, 'r--')
    xlim(0.0, 1.0)
    ylim(-1.5, 1.5)
    show()

if __name__ == "__main__":
    main()
