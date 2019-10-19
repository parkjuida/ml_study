import numpy as np

from regression.base.base_regression import BaseRegression


class LinearRegression(BaseRegression):
    def __init__(self, learning_rate=0.001, iteration=None):
        super(LinearRegression, self).__init__(learning_rate, iteration)

    def _calculate_gradient(self, y_):
        x_ = self.x.transpose()
        return -2 * x_.dot(self.y - y_) / y_.shape[0]

    def _estimate(self, x):
        return x.dot(self.w)

    def _calculate_loss(self, y_):
        return np.mean(np.square(self.y - y_))
