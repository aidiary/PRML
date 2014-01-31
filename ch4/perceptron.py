#coding:utf-8
import numpy as np
from pylab import *

"""
パーセプトロン
確率分布からデータを生成
"""

N = 100
ETA = 0.1

if __name__ == "__main__":
    # 訓練データを作成
    cls1 = []
    cls2 = []
    t = []

    # データは正規分布に従って生成
    mean1 = [-2, 2]    # クラス1の平均
    mean2 = [2, -2]    # クラス2の平均
    cov = [[1.0, 0.0], [0.0, 1.0]]  # 共分散行列（全クラス共通）

    # データ作成
    cls1.extend(np.random.multivariate_normal(mean1, cov, N / 2))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N / 2))

    # 教師データ
    for i in range(N / 2):
        t.append(+1)  # クラス1
    for i in range(N/2):
        t.append(-1)  # クラス2

    # 訓練データを描画
    # クラス1
    x1, x2 = np.transpose(np.array(cls1))
    plot(x1, x2, 'bo')

    # クラス2
    x1, x2 = np.transpose(np.array(cls2))
    plot(x1, x2, 'ro')

    # クラス1とクラス2のデータをマージ
    x1, x2 = np.array(cls1+cls2).transpose()

    # 確率的最急降下法でパラメータwを更新
    w = array([1.0, 1.0, 1.0])  # 適当な初期値

    turn = 0
    correct = 0  # 分類が正解したデータ数
    while correct < N:  # 全部のデータが正しく分類できるまで続ける
        correct = 0
        for i in range(N):  # 全データについて検討
            if np.dot(w, [1, x1[i], x2[i]]) * t[i] > 0:  # 分類が正しいときは何もしない
                correct += 1
            else:  # 分類が間違っているときは重みを調整
                w += ETA * array([1, x1[i], x2[i]]) * t[i]
        turn += 1
        print turn, w

    # 決定境界を描画
    x = linspace(-6.0, 6.0, 1000)
    y = -w[1] / w[2] * x - w[0] / w[2]
    plot(x, y, 'g-')

    xlim(-6, 6)
    ylim(-6, 6)
    show()
