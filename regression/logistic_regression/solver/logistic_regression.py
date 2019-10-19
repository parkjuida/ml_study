import numpy as np

from regression.base.base_regression import BaseRegression


class LogisticRegression(BaseRegression):
    def __init__(self, learning_rate=0.001, iteration=None):
        super(LogisticRegression, self).__init__(learning_rate=learning_rate, iteration=iteration)

    def _calculate_gradient(self, y_):
        x_ = self.x.transpose()
        return - x_.dot(self.y - y_) / y_.shape[0]

    def _estimate(self, x):
        return 1 / (1 + np.exp(-x.dot(self.w)))

    def _calculate_loss(self, y_):
        return np.mean(self.y * np.log(y_) - (1 - self.y) * np.log(1 - y_))
