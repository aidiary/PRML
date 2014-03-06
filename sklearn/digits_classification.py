#coding:utf-8
from sklearn import datasets, neighbors, linear_model
from sklearn.metrics import confusion_matrix, classification_report
"""
digitsデータの分類
k-NN vs ロジスティック回帰
"""

# digitsデータをロード
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# 訓練データとテストデータに分割
n_samples = len(X_digits)
pivot = n_samples * 0.9
X_train = X_digits[:pivot]
y_train = y_digits[:pivot]
X_test = X_digits[pivot:]
y_test = y_digits[pivot:]

# k-NNとロジスティック回帰の分類器を学習
knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()
knn.fit(X_train, y_train)
logistic.fit(X_train, y_train)

# クラスを予測
predict_knn = knn.predict(X_test)
predict_logistic = logistic.predict(X_test)

# 結果を表示
print "KNN"
print "score: %f" % knn.score(X_test, y_test)
print confusion_matrix(y_test, predict_knn)
print classification_report(y_test, predict_knn)

print "Logistic Regression"
print "score: %f" % logistic.score(X_test, y_test)
print confusion_matrix(y_test, predict_logistic)
print classification_report(y_test, predict_logistic)

