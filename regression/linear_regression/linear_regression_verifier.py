import numpy as np

from solver.linear_regression import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
X_predict = np.array([[0], [-1], [10]])
y_predict = np.array([0, -2, 20])

lr = LinearRegression()
lr.fit(X, y)
y_ = lr.predict(X_predict)
print(y_ - y_predict)
print(y_)

X = np.array([[1, 3], [2, 5], [-5, 3], [-7, -13]])
w = np.array([-10, 20])
y = X.dot(w)
X_predict = np.array([[-20, 10], [10, -14]])
y_predict = X_predict.dot(w)
lr = LinearRegression()
lr.fit(X, y)
y_ = lr.predict(X_predict)
print(y_)
print(y_predict)
