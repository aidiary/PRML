#coding: utf-8
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(1)
g = mixture.GMM(n_components=2)

# 一次元データ生成
# 平均が0のクラスタに属する100個のデータと平均が10のクラスタに属する300個のデータ
obs = np.concatenate((np.random.randn(100, 1), 10 + np.random.randn(300, 1)))

# GMMへフィッティング
g.fit(obs)

# 混合係数
print g.weights_

# 各ガウス分布の平均
print g.means_

# 各ガウス分布の分散
print g.covars_

# 新データに対する予測
newdata = [[0], [2], [4], [5], [9], [10]]

# どちらのクラスタに属するか返す
print g.predict(newdata)

# どちらのクラスタに属するかを表す確率
print np.round(g.predict_proba(newdata), 2)

# log probabilities
# データのモデルへの当てはまりやすさ
# 対数尤度か？
print np.round(g.score(newdata), 2)

plt.plot(obs)
plt.show()
