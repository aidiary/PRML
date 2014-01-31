#coding:utf-8
import numpy as np
import sys
from pylab import *

# M次多項式近似
M = 3

def y(x, wlist):
    ret = wlist[0]
    for i in range(1, M + 1):
        ret += wlist[i] * (x ** i)
    return ret

def estimate(xlist, tlist):
    """訓練データからパラメータを推定"""
    # M次多項式のときはM+1個のパラメータがある
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
    # sin(2 * pi * x)の関数値にガウス分布に従う小さなランダムノイズを加える
    xlist = np.linspace(0, 1, 10)
    tlist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)

    # 訓練データからパラメータを推定
    wlist = estimate(xlist, tlist)
    print wlist

    # 連続関数のプロット用X値
    xs = np.linspace(0, 1, 500)
    ideal = np.sin(2 * np.pi * xs)      # 理想曲線
    model = [y(x, wlist) for x in xs]  # 推定モデル

    plot(xlist, tlist, 'bo')  # 訓練データ
    plot(xs, ideal, 'g-')     # 理想曲線
    plot(xs, model, 'r-')     # 推定モデル
    xlim(0.0, 1.0)
    ylim(-1.5, 1.5)
    show()

if __name__ == "__main__":
    main()