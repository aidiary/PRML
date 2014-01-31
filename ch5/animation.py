#coding:utf-8
import numpy as np
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt

"""
多層パーセプトロンによる関数近似のアニメーション
wxPythonが必要
"""

# 訓練データ数
N = 50

# 学習率
ETA = 0.1

# ループ回数
NUM_LOOP = 1000

# 入力層のユニット数（バイアス含む）
NUM_INPUT = 2
# 隠れ層のユニット数（バイアス含む）
NUM_HIDDEN = 4
# 出力層のユニット数
NUM_OUTPUT = 1

def heviside(x):
    """Heviside関数"""
    return 0.5 * (np.sign(x) + 1)

def sum_of_squares_error(xlist, tlist, w1, w2):
    """二乗誤差和を計算する"""
    error = 0.0
    for n in range(N):
        z = np.zeros(NUM_HIDDEN)
        y = np.zeros(NUM_OUTPUT)
        # バイアスの1を先頭に挿入
        x = np.insert(xlist[n], 0, 1)
        # 順伝播で出力を計算
        for j in range(NUM_HIDDEN):
            a = np.zeros(NUM_HIDDEN)
            a[j] = np.dot(w1[j, :], x)
            z[j] = np.tanh(a[j])
        for k in range(NUM_OUTPUT):
            y[k] = np.dot(w2[k, :], z)
        # 二乗誤差を計算
        for k in range(NUM_OUTPUT):
            error += 0.5 * (y[k] - tlist[n, k]) * (y[k] - tlist[n, k])
    return error

def output(x, w1, w2):
    """xを入力したときのニューラルネットワークの出力を計算
    隠れユニットの出力も一緒に返す"""
    # 配列に変換して先頭にバイアスの1を挿入
    x = np.insert(x, 0, 1)
    z = np.zeros(NUM_HIDDEN)
    y = np.zeros(NUM_OUTPUT)
    # 順伝播で出力を計算
    for j in range(NUM_HIDDEN):
        a = np.zeros(NUM_HIDDEN)
        a[j] = np.dot(w1[j, :], x)
        z[j] = np.tanh(a[j])
    for k in range(NUM_OUTPUT):
        y[k] = np.dot(w2[k, :], z)
    return y, z

fig = plt.figure()
fig.suptitle("Function approximation by multilayer perceptron", fontsize=18)
ax = fig.add_subplot(111)

# 訓練データ作成
xlist = np.linspace(-1, 1, N).reshape(N, 1)
tlist = xlist * xlist    # x^2
#tlist = np.sin(xlist)    # sin(x)
#tlist = np.abs(xlist)    # |x|
#tlist = heviside(xlist)  # H(x)

# 重みをランダムに初期化
w1 = np.random.random((NUM_HIDDEN, NUM_INPUT))
w2 = np.random.random((NUM_OUTPUT, NUM_HIDDEN))

print "学習前の二乗誤差:", sum_of_squares_error(xlist, tlist, w1, w2)

loop = 0

# 1回のループの処理内容をupdate()に記述
# 収束するまで十分なループを回す
# 二乗誤差の差がepsilon以下になったら終了でもOK
def update(idleevent):
    global loop

    # 訓練データすべてを使って重みを訓練する
    for n in range(N):
        # 隠れ層と出力層の出力配列を確保
        z = np.zeros(NUM_HIDDEN)
        y = np.zeros(NUM_OUTPUT)

        # 誤差（delta）の配列を確保
        d1 = np.zeros(NUM_HIDDEN)
        d2 = np.zeros(NUM_OUTPUT)

        # 訓練データにバイアスの1を先頭に挿入
        x = np.array([xlist[n]])
        x = np.insert(x, 0, 1)

        # (1) 順伝播により隠れ層の出力を計算
        for j in range(NUM_HIDDEN):
            # 入力層にはバイアスの1が先頭に入るので注意
            a = np.zeros(NUM_HIDDEN)
            a[j] = np.dot(w1[j, :], x)
            z[j] = np.tanh(a[j])

        # (1) 順伝播により出力層の出力を計算
        for k in range(NUM_OUTPUT):
            y[k] = np.dot(w2[k, :], z)

        # (2) 出力層の誤差を評価
        for k in range(NUM_OUTPUT):
            d2[k] = y[k] - tlist[n, k]

        # (3) 出力層の誤差を逆伝播させ、隠れ層の誤差を計算
        for j in range(NUM_HIDDEN):
            temp = np.dot(w2[:, j], d2)
            d1[j] = (1 - z[j] * z[j]) * temp

        # (4) + (5) 第1層の重みを更新
        for j in range(NUM_HIDDEN):
            for i in range(NUM_INPUT):
                w1[j, i] -= ETA * d1[j] * x[i]

        # (4) + (5) 第2層の重みを更新
        for k in range(NUM_OUTPUT):
            for j in range(NUM_HIDDEN):
                w2[k, j] -= ETA * d2[k] * z[j]

    error = sum_of_squares_error(xlist, tlist, w1, w2)
    print loop, error

    # このループでの途中経過のグラフを描画
    # グラフをクリア
    ax.cla()

    # 訓練データの入力に対する予測値を出力
    ylist = np.zeros((N, NUM_OUTPUT))
    zlist = np.zeros((N, NUM_HIDDEN))
    for n in range(N):
        ylist[n], zlist[n] = output(xlist[n], w1, w2)

    # 訓練データを青丸で表示
    # plot()には縦ベクトルを渡してもOK
        ax.plot(xlist, tlist, 'bo')

    # 訓練データの各xに対するニューラルネットの出力を赤線で表示
    ax.plot(xlist, ylist, 'r-')

    # 各重みの出力を点線で表示
    for i in range(NUM_HIDDEN):
        ax.plot(xlist, zlist[:,i], 'k--')

    ax.set_title("loop: %d  squared error: %f" % (loop, error))
    ax.set_ylim([0.0, 1.0])

    fig.canvas.draw_idle()
    loop += 1

import wx
wx.EVT_IDLE(wx.GetApp(), update)
plt.show()