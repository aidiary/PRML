#coding:utf-8
import numpy as np
from mlp import MultiLayerPerceptron
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

"""
MNISTの手書き数字データの認識
scikit-learnのインストールが必要
http://scikit-learn.org/
"""

def train_test_split(X, y, test_rate=0.1):
    """訓練データとテストデータに分割"""
    # 訓練データと教師信号をシャッフル
    num_samples= X.shape[0]
    p = np.arange(num_samples)
    np.random.seed(0)
    np.random.shuffle(p)
    X, y = X[p], y[p]

    # 訓練データとテストデータに分ける
    num_test = int(num_samples * test_rate)
    num_train = num_samples - num_test
    X_train = X[0:num_train, :]
    X_test = X[num_train:num_samples, :]
    y_train = y[0:num_train]
    y_test = y[num_train:num_samples]

    return X_train, X_test, y_train, y_test

def draw_digits(mnist):
    """数字データの10サンプルをランダムに描画"""
    import pylab
    np.random.seed(0)
    p = np.random.random_integers(0, len(mnist.data), 10)
    for index, (data, label) in enumerate(np.array(zip(mnist.data, mnist.target))[p]):
        pylab.subplot(2, 5, index + 1)
        pylab.axis('off')
        pylab.imshow(data.reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')
        pylab.title('%i' % label)
    pylab.show()

if __name__ == "__main__":
    # MNISTの数字データ
    # 70000サンプル, 28x28ピクセル
    # カレントディレクトリ（.）にmnistデータがない場合は
    # Webから自動的にダウンロードされる（時間がかかる）
    mnist = fetch_mldata('MNIST original', data_home=".")

    # 数字データのサンプルを描画
    draw_digits(mnist)

    # 訓練データを作成
    X = mnist.data
    y = mnist.target

    # ピクセルの値を0.0-1.0に正規化
    X = X.astype(np.float64)
    X /= X.max()

    # 多層パーセプトロンを構築
    mlp = MultiLayerPerceptron(28*28, 100, 10)

    # 訓練データとテストデータに分解
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_rate=0.1)

    # 教師信号の数字を1-of-K表記に変換
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    # 訓練データを用いてニューラルネットの重みを学習
    mlp.fit(X_train, labels_train, epochs=50000)

    # テストデータを用いて予測精度を計算
    predictions = []
    for i in range(X_test.shape[0]):
        o = mlp.predict(X_test[i])
        predictions.append(np.argmax(o))
    print confusion_matrix(y_test, predictions)
    print classification_report(y_test, predictions)
