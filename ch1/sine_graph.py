#coding:utf-8
import numpy as np
from pylab import *

# 訓練データの数
N = 10

# N個の訓練データを生成
# 0から1の間から均等にN点を抽出
xlist = np.linspace(0, 1, N)
# sinにN(0, 0.2)の乱数を加える
tlist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)

# オリジナルのsin用データ
# sinは連続関数なのでxを細かくとる
xs = np.linspace(0, 1, 1000)
# xsに対応するsinを求める
ideal = np.sin(2 * np.pi * xs)

# 訓練データとオリジナルのsinデータをプロット
plot(xlist, tlist, 'bo')  # 訓練データを青い（b）の点（o）で描画
plot(xs, ideal, 'g-')     # sin曲線を緑（g）の線（-）で描画
xlim(0.0, 1.0)            # X軸の範囲
ylim(-1.5, 1.5)           # Y軸の範囲
show()                    # グラフを表示
