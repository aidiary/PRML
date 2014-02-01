#coding: utf-8
import numpy as np
import pylab
from sklearn.datasets import load_digits

# scikit-learnの手書き数字データをロード
# 1797サンプル、8x8ピクセル
digits = load_digits()

# 最初の10サンプルを描画
# digits.images[i] : i番目の画像データ（8x8ピクセル）
# digits.target[i] : i番目の画像データのクラス（数字なので0-9）
for index, (image, label) in enumerate(zip(digits.images, digits.target)[:10]):
    pylab.subplot(2, 5, index + 1)
    pylab.axis('off')
    pylab.imshow(image, cmap=pylab.cm.gray_r, interpolation='nearest')
    pylab.title('%i' % label)
pylab.show()
