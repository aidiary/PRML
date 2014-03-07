#coding: utf-8
import numpy as np
from sklearn import datasets, svm

"""
K-fold Cross-validation
"""

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

K = 10

# データをK分割する
X_folds = np.array_split(X_digits, K)
y_folds = np.array_split(y_digits, K)

svc = svm.SVC(C=1, kernel='linear')

scores = list()
for k in range(K):
    # k番目をテストセットとして残り2つで学習
    # X_foldsは元のまま残しておくのでコピーする
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)

    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)

    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

print scores
print "mean:", np.mean(scores)
