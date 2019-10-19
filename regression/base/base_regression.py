from abc import abstractmethod

import numpy as np


class BaseRegression:
    def __init__(self, learning_rate=0.001, iteration=None):
        self.x = None
        self.y = None
        self.w = None
        self.rows = None
        self.features = None
        self.learning_rate = learning_rate
        self.iteration = iteration

    def _valid_check(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise Exception("x, y size difference x:{}, y:{}".format(x.shape, y.shape))

    @abstractmethod
    def _estimate(self, x):
        raise NotImplementedError()

    @abstractmethod
    def _calculate_loss(self, y_):
        raise NotImplementedError()

    @abstractmethod
    def _calculate_gradient(self, y_):
        raise NotImplementedError()

    def _fit_with_iteration(self, learning_rate: np.float, iteration: int):
        for _ in range(iteration):
            y_ = self._estimate(self.x)
            print(y_)
            grad = self._calculate_gradient(y_)
            self.w -= learning_rate * grad

    def _fit_with_learning_rate_decay(self, learning_rate: np.float):
        learning_rate_ = learning_rate
        learning_rate_limit = 1e-10
        loss_limit = 1e-20
        min_loss = np.inf
        while learning_rate > learning_rate_limit and np.abs(min_loss) > loss_limit:
            y_ = self._estimate(self.x)
            loss = self._calculate_loss(y_)
            grad = self._calculate_gradient(y_)
            self.w -= learning_rate_ * grad
            if min_loss > loss:
                min_loss = loss
            else:
                learning_rate_ = learning_rate_ / 10

    def _fit(self, learning_rate: np.float, iteration: int):
        if iteration:
            self._fit_with_iteration(learning_rate, iteration)
        else:
            self._fit_with_learning_rate_decay(learning_rate)

    def _append_one_to_x(self, x: np.array, data_type: np.core.numerictypes = np.float) -> np.array:
        ones = np.ones((x.shape[0], 1), dtype=data_type)
        return np.append(ones, x, axis=1)

    def fit_init(self, x, y):
        self.rows, self.features = x.shape

        self.x = self._append_one_to_x(x)
        self.y = y.reshape(-1, 1)
        self.w = np.random.rand(self.features + 1, 1)

    def fit(self, x: np.array, y: np.array):
        self._valid_check(x, y)
        self.fit_init(x, y)
        self._fit(self.learning_rate, self.iteration)

    def predict(self, x: np.array) -> np.array:
        x_ = self._append_one_to_x(x)
        return self._estimate(x_).squeeze()

