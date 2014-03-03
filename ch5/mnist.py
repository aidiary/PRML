#coding:utf-8
import numpy as np
from mlp import MultiLayerPerceptron
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

"""
MNISTの手書き数字データの認識
scikit-learnのインストールが必要
http://scikit-learn.org/
"""

if __name__ == "__main__":
    # MNISTの数字データ
    # 70000サンプル, 28x28ピクセル
    # カレントディレクトリ（.）にmnistデータがない場合は
    # Webから自動的にダウンロードされる（時間がかかる）
    mnist = fetch_mldata('MNIST original', data_home=".")

    # 訓練データを作成
    X = mnist.data
    y = mnist.target

    # ピクセルの値を0.0-1.0に正規化
    X = X.astype(np.float64)
    X /= X.max()

    # 多層パーセプトロンを構築
    mlp = MultiLayerPerceptron(28*28, 100, 10, act1="tanh", act2="softmax")

    # 訓練データとテストデータに分解
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # 教師信号の数字を1-of-K表記に変換
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    # 訓練データを用いてニューラルネットの重みを学習
    mlp.fit(X_train, labels_train, learning_rate=0.01, epochs=100000)

    # テストデータを用いて予測精度を計算
    predictions = []
    for i in range(X_test.shape[0]):
        o = mlp.predict(X_test[i])
        predictions.append(np.argmax(o))
    print confusion_matrix(y_test, predictions)
    print classification_report(y_test, predictions)
