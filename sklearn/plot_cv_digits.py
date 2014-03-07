#coding:utf-8
import numpy as np
from sklearn import cross_validation, datasets, svm

"""
SVMの最適なパラメータCを
Cross-validationで求める
"""

digits = datasets.load_digits()
X = digits.data
y = digits.target

svc = svm.SVC(kernel='linear')

# パラメータCの範囲
C_s = np.logspace(-10, 0, 10)

scores = list()
scores_std = list()

for C in C_s:
    svc.C = C
    # デフォルトでは3-fold CV
    this_scores = cross_validation.cross_val_score(svc, X, y, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

import pylab as pl
pl.figure()
pl.clf()
pl.semilogx(C_s, scores)
pl.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
pl.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
pl.xlabel('Parameter C')
pl.ylabel('CV score')
pl.ylim(0, 1.1)
pl.show()
