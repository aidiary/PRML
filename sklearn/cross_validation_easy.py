#coding: utf-8
import numpy as np
from sklearn import datasets, svm, cross_validation

"""
K-fold Cross-validation
"""

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

K = 10

svc = svm.SVC(C=1, kernel='linear')

# KFoldは訓練集合とテスト集合の分割結果のインデックスを返す
kfold = cross_validation.KFold(len(X_digits), n_folds=K)
for i, (train_indices, test_indices) in enumerate(kfold):
    print ('%d : Train: %s | test: %s' % (i, len(train_indices), len(test_indices)))

# 分類精度を評価
scores = cross_validation.cross_val_score(svc, X_digits, y_digits, cv=kfold, n_jobs=-1)

print scores
print "mean:", np.mean(scores)
