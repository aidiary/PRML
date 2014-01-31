#coding: utf-8
import numpy as np
import sys

"""
多層パーセプトロン
forループの代わりに行列演算にした高速化版

入力層 - 隠れ層 - 出力層の3層構造で固定（PRMLではこれを2層と呼んでいる）

隠れ層の活性化関数にはtanh関数またはsigmoid logistic関数が使える
出力層の活性化関数にはtanh関数、sigmoid logistic関数、恒等関数が使える
"""

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - x ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def identity(x):
    return x

def identity_deriv(x):
    return 1

class MultiLayerPerceptron:
    def __init__(self, numInput, numHidden, numOutput, act1="tanh", act2="sigmoid"):
        """多層パーセプトロンを初期化
        numInput    入力層のユニット数（バイアスユニットは除く）
        numHidden   隠れ層のユニット数（バイアスユニットは除く）
        numOutput   出力層のユニット数
        act1        隠れ層の活性化関数（tanh or sigmoid）
        act2        出力層の活性化関数（tanh or sigmoid or identity）
        """
        # 引数の指定に合わせて隠れ層の活性化関数とその微分関数を設定
        if act1 == "tanh":
            self.act1 = tanh
            self.act1_deriv = tanh_deriv
        elif act1 == "sigmoid":
            self.act1 = sigmoid
            self.act1_deriv = sigmoid_deriv
        else:
            print "ERROR: act1 is tanh or sigmoid"
            sys.exit()

        # 引数の指定に合わせて出力層の活性化関数とその微分関数を設定
        if act2 == "tanh":
            self.act2 = tanh
            self.act2_deriv = tanh_deriv
        elif act2 == "sigmoid":
            self.act2 = sigmoid
            self.act2_deriv = sigmoid_deriv
        elif act2 == "identity":
            self.act2 = identity
            self.act2_deriv = identity_deriv
        else:
            print "ERROR: act2 is tanh or sigmoid or identity"
            sys.exit()

        # バイアスユニットがあるので入力層と隠れ層は+1
        self.numInput = numInput + 1
        self.numHidden =numHidden + 1
        self.numOutput = numOutput

        # 重みを (-1.0, 1.0)の一様乱数で初期化
        self.weight1 = np.random.uniform(-1.0, 1.0, (self.numHidden, self.numInput))  # 入力層-隠れ層間
        self.weight2 = np.random.uniform(-1.0, 1.0, (self.numOutput, self.numHidden)) # 隠れ層-出力層間

    def fit(self, X, t, learning_rate=0.2, epochs=10000):
        """訓練データを用いてネットワークの重みを更新する"""
        # 入力データの最初の列にバイアスユニットの入力1を追加
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        t = np.array(t)

        # 逐次学習
        # 訓練データからランダムサンプリングして重みを更新をepochs回繰り返す
        for k in range(epochs):
            print k

            # 訓練データからランダムに選択する
            i = np.random.randint(X.shape[0])
            x = X[i]

            # 入力を順伝播させて中間層の出力を計算
            z = self.act1(np.dot(self.weight1, x))

            # 中間層の出力を順伝播させて出力層の出力を計算
            y = self.act2(np.dot(self.weight2, z))

            # 出力層の誤差を計算
            delta2 = self.act2_deriv(y) * (y - t[i])

            # 出力層の誤差を逆伝播させて隠れ層の誤差を計算
            delta1 = self.act1_deriv(z) * np.dot(self.weight2.T, delta2)

            # 隠れ層の誤差を用いて隠れ層の重みを更新
            # 行列演算になるので2次元ベクトルに変換する必要がある
            x = np.atleast_2d(x)
            delta1 = np.atleast_2d(delta1)
            self.weight1 -= learning_rate * np.dot(delta1.T, x)

            # 出力層の誤差を用いて出力層の重みを更新
            z = np.atleast_2d(z)
            delta2 = np.atleast_2d(delta2)
            self.weight2 -= learning_rate * np.dot(delta2.T, z)

    def predict(self, x):
        """テストデータの出力を予測"""
        x = np.array(x)
        # バイアスの1を追加
        x = np.insert(x, 0, 1)
        # 順伝播によりネットワークの出力を計算
        z = self.act1(np.dot(self.weight1, x))
        y = self.act2(np.dot(self.weight2, z))
        return y

if __name__ == "__main__":
    """XORの学習テスト"""
    mlp = MultiLayerPerceptron(2, 2, 1, "tanh", "sigmoid")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    mlp.fit(X, y)
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print i, mlp.predict(i)
