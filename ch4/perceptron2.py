#coding:utf-8
import numpy as np
from pylab import *

"""
パーセプトロン
データの生成を正解パラメータから作成（図4.7に近い）
"""

N = 100
ETA = 0.1

if __name__ == "__main__":
    # 正解パラメータから訓練データを作成
    true_w = array([-0.3, 1.3, -2.0])  # 正解パラメータ w
    x1 = -1 + np.random.random_sample(N) * 2  # (-1,+1)の乱数をN個
    x2 = -1 + np.random.random_sample(N) * 2  # (-1,+1)の乱数をN個
    t = (true_w[0] + x1 * true_w[1] + x2 * true_w[2] >= 0) * 2 - 1  # ラベル

    # 訓練データを描画
    for i in range(N):
        if t[i] == 1:
            plot([x1[i]], [x2[i]], 'ro')
        elif t[i] == -1:
            plot([x1[i]], [x2[i]], 'bo')

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
    x = linspace(-1.0, 1.0, 1000)
    y = - w[1] / w[2] * x - w[0] / w[2]
    plot(x, y, 'g-')

    # 正解の決定境界を描画
    y = - true_w[1] / true_w[2] * x - true_w[0] / true_w[2]
    plot(x, y, 'r--')

    xlim(-1.0, 1.0)
    ylim(-1.0, 1.0)
    show()