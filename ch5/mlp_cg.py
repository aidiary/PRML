#coding: utf-8
import numpy as np
from scipy import optimize
from matplotlib import pyplot
from sklearn.datasets import load_digits, fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

"""
正則化多層パーセプトロンによる手書き文字認識
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

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def safe_log(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))

def nnCostFunction(nn_params, *args):
    """NNのコスト関数とその偏微分を求める"""
    in_size, hid_size, num_labels, X, y, lam = args

    # ニューラルネットの全パラメータを行列形式に復元
    Theta1 = nn_params[0:(in_size + 1) * hid_size].reshape((hid_size, in_size + 1))
    Theta2 = nn_params[(in_size + 1) * hid_size:].reshape((num_labels, hid_size + 1))

    # パラメータの偏微分
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # 訓練データ数
    m = X.shape[0]

    # 訓練データの1列目にバイアス項に対応する1を追加
    X = np.hstack((np.ones((m, 1)), X))

    # 教師ラベルを1-of-K表記に変換
    lb = LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    J = 0
    for i in range(m):
        xi = X[i, :]
        yi = y[i]
        # forward propagation
        a1 = xi
        z2 = np.dot(Theta1, a1)
        a2 = sigmoid(z2)
        a2 = np.hstack((1, a2))
        z3 = np.dot(Theta2, a2)
        a3 = sigmoid(z3)
        J += sum(-yi * safe_log(a3) - (1 - yi) * safe_log(1 - a3))
        # backpropagation
        delta3 = a3 - yi
        delta2 = np.dot(Theta2.T, delta3) * sigmoidGradient(np.hstack((1, z2)))
        delta2 = delta2[1:]  # バイアス項に対応する要素を除外
        # ベクトル x ベクトル = 行列の演算をしなければならないので
        # 縦ベクトルへのreshapeが必要
        # 行数に-1を指定すると自動的に入る
        delta2 = delta2.reshape((-1, 1))
        delta3 = delta3.reshape((-1, 1))
        a1 = a1.reshape((-1, 1))
        a2 = a2.reshape((-1, 1))
        # 正則化ありのときのデルタの演算
        Theta1_grad += np.dot(delta2, a1.T)
        Theta2_grad += np.dot(delta3, a2.T)
    J /= m

    # 正則化項
    temp = 0.0;
    for j in range(hid_size):
        for k in range(1, in_size + 1):  # バイアスに対応する重みは加えない
            temp += Theta1[j, k] ** 2
    for j in range(num_labels):
        for k in range(1, hid_size + 1): # バイアスに対応する重みは加えない
            temp += Theta2[j, k] ** 2
    J += lam / (2.0 * m) * temp;

    # 偏微分の正則化項
    Theta1_grad /= m
    Theta1_grad[:, 1:] += (lam / m) * Theta1_grad[:, 1:]
    Theta2_grad /= m
    Theta2_grad[:, 1:] += (lam / m) * Theta2_grad[:, 1:]

    # ベクトルに変換
    grad = np.hstack((np.ravel(Theta1_grad), np.ravel(Theta2_grad)))

    print J
    return J, grad

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    # forward propagation
    X = np.hstack((np.ones((m, 1)), X))
    h1 = sigmoid(np.dot(X, Theta1.T))

    h1 = np.hstack((np.ones((m, 1)), h1))
    h2 = sigmoid(np.dot(h1, Theta2.T))

    return np.argmax(h2, axis=1)

if __name__ == "__main__":
    in_size = 64
    hid_size = 25
    num_labels = 10

    # 訓練データをロード
    digits = load_digits()
    X = digits.data
    X /= X.max()
    y = digits.target

    # データを可視化
    displayData(X)

    # パラメータをランダムに初期化
    initial_Theta1 = randInitializeWeights(in_size, hid_size)
    initial_Theta2 = randInitializeWeights(hid_size, num_labels)

    # パラメータをベクトルにフラット化
    initial_nn_params = np.hstack((np.ravel(initial_Theta1), np.ravel(initial_Theta2)))

    # 正則化係数
    lam = 1.0

    # 初期状態のコストを計算
    J, grad = nnCostFunction(initial_nn_params, in_size, hid_size, num_labels, X, y, lam)
    print "initial cost:", J

    # Conjugate Gradientでパラメータ推定
    # NNはコスト関数と偏微分の計算が重複するため同じ関数（nnCostFunction）にまとめている
    # この場合、fmin_cgではなくminimizeを使用するとよい
    # minimize()はscipy 0.11.0以上が必要
    res = optimize.minimize(fun=nnCostFunction, x0=initial_nn_params, method="CG", jac=True,
                                  options={'maxiter':20, 'disp':True},
                                  args=(in_size, hid_size, num_labels, X, y, lam))
    nn_params = res.x

    # パラメータを分解
    Theta1 = nn_params[0:(in_size + 1) * hid_size].reshape((hid_size, in_size + 1))
    Theta2 = nn_params[(in_size + 1) * hid_size:].reshape((num_labels, hid_size + 1))

    # 隠れユニットを可視化
    displayData(Theta1[:, 1:])

    # 訓練データのラベルを予測して精度を求める
    pred = predict(Theta1, Theta2, X)
    print confusion_matrix(y, pred)
    print classification_report(y, pred)

