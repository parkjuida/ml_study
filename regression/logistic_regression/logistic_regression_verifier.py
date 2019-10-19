import numpy as np

from regression.logistic_regression.solver.logistic_regression import LogisticRegression

X = np.array([[-2], [-1], [1], [2]])
y = np.array([0, 0, 1, 1])

x_predict = np.array([[-3], [3]])
y_predict = np.array([0, 1])

lr = LogisticRegression(iteration=1000)
lr.fit(X, y)
print(lr.predict(x_predict))
