#coding:utf-8
import numpy as np
from mlp import MultiLayerPerceptron
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

"""
簡易手書き数字データの認識
scikit-learnのインストールが必要
http://scikit-learn.org/
"""

if __name__ == "__main__":
    # scikit-learnの簡易数字データをロード
    # 1797サンプル, 8x8ピクセル
    digits = load_digits()

    # 訓練データを作成
    X = digits.data
    y = digits.target
    # ピクセルの値を0.0-1.0に正規化
    X /= X.max()

    # 多層パーセプトロン
    mlp = MultiLayerPerceptron(64, 100, 10, act1="tanh", act2="sigmoid")

    # 訓練データ（90%）とテストデータ（10%）に分解
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # 教師信号の数字を1-of-K表記に変換
    # 0 => [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # ...
    # 9 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    # 訓練データを用いてニューラルネットの重みを学習
    mlp.fit(X_train, labels_train, learning_rate=0.01, epochs=10000)

    # テストデータを用いて予測精度を計算
    predictions = []
    for i in range(X_test.shape[0]):
        o = mlp.predict(X_test[i])
        # 最大の出力を持つクラスに分類
        predictions.append(np.argmax(o))
    print confusion_matrix(y_test, predictions)
    print classification_report(y_test, predictions)

    # 誤認識したデータのみ描画
    # 誤認識データ数と誤っているテストデータのidxを収集
    cnt = 0
    error_idx = []
    for idx in range(len(y_test)):
        if y_test[idx] != predictions[idx]:
            print "error: %d : %d => %d" % (idx, y_test[idx], predictions[idx])
            error_idx.append(idx)
            cnt += 1

    # 描画
    import pylab
    for i, idx in enumerate(error_idx):
        pylab.subplot(cnt/5 + 1, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(X_test[idx].reshape((8, 8)), cmap=pylab.cm.gray_r)
        pylab.title('%d : %i => %i' % (idx, y_test[idx], predictions[idx]))
    pylab.show()

