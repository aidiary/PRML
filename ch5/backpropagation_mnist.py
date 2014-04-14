#coding: utf-8
import numpy as np
from scipy import optimize
from matplotlib import pyplot
from sklearn.datasets import load_digits, fetch_mldata

"""
多層パーセプトロンによる手書き文字認識
・バッチ処理
・共役勾配法（Conjugate Gradient Method）によるパラメータ最適化
"""

def displayData(X):
    """
    データからランダムに100サンプル選んで可視化
    画像データは8x8ピクセルを仮定
    """
    # ランダムに100サンプル選ぶ
    sel = np.random.permutation(X.shape[0])
    sel = sel[:100]
    X = X[sel, :]
    for index, data in enumerate(X):
        pyplot.subplot(10, 10, index + 1)
        pyplot.axis('off')
        image = data.reshape((8, 8))
        print image.shape
        pyplot.imshow(image, cmap=pyplot.cm.gray_r,
                     interpolation='nearest')
    pyplot.show()

def randInitializeWeights(L_in, L_out):
    """
    (-epsilon_init, +epsilon_init) の範囲で
    重みをランダムに初期化した重み行列を返す
    """
    # 入力となる層にはバイアス項が入るので+1が必要なので注意
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

def J(nn_params, *args):
    """ニューラルネットのコスト関数"""
    in_size, hid_size, num_labels, X, y, lam = args

    # ニューラルネットの全パラメータを行列形式に復元
    Theta1 = nn_params[0:(in_size + 1) * hid_size].reshape((hid_size, in_size + 1))
    Theta2 = nn_params[(in_size + 1) * hid_size:].reshape((num_labels, hid_size + 1))

if __name__ == "__main__":
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # 訓練データをロード
    digits = load_digits()
    X = digits.data
    X /= X.max()
    y = digits.target

    # データを可視化
#    displayData(X)

    # パラメータをランダムに初期化
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    # 行列ではなくベクトルに展開
    initial_nn_params = np.hstack((np.ravel(initial_Theta1), np.ravel(initial_Theta2)))
    print initial_Theta1.shape
    print initial_Theta2.shape
    print initial_nn_params.shape

    # 正則化係数
    lam = 1.0

    # 初期状態のコストを計算
    print "initial cost:", J(initial_nn_params, input_layer_size, hidden_layer_size,
                             num_labels, X, y, lam)


