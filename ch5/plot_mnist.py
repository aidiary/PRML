#coding: utf-8
import numpy as np
import pylab
from sklearn.datasets import fetch_mldata

# mnistの手書き数字データをロード
# カレントディレクトリ（.）にない場合は、
# Webから自動的にダウンロードされる（時間がかかるので注意！）
# 70000サンプル、28x28ピクセル
mnist = fetch_mldata('MNIST original', data_home=".")

# ランダムに25サンプルを描画
# digits.images[i] : i番目の画像データ（8x8ピクセル）
# digits.target[i] : i番目の画像データのクラス（数字なので0-9）
p = np.random.random_integers(0, len(mnist.data), 25)
for index, (data, label) in enumerate(np.array(zip(mnist.data, mnist.target))[p]):
    pylab.subplot(5, 5, index + 1)
    pylab.axis('off')
    pylab.imshow(data.reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')
    pylab.title('%i' % label)
pylab.show()
