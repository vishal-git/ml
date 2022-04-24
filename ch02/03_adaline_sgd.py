# 03_adaline_sdg.py

from numpy import random, float_, dot, where, zeros, copy, mean
from pandas import read_csv
import sys

class AdalineSGD:
    def __init__(self, eta=.01, n_iter=10, shuffle=True, random_state=314):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for i in range(self.n_iter):
            if self.shuffle == True:
                X, y = self._shuffle(X, y)
            losses = []
            for x_i, target in zip(X, y):
                losses.append(self._update_weights(x_i, target))
            avg_loss = mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for x_i, target in zip(X, y):
                self._update_weights(x_i, target)
        else:
            self._update_weights(X, y)
    
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=.01, size=m)
        self.b_ = float(0.)
        self.w_initialized = True

    def _update_weights(self, x_i, target):
        output = self.activation(self.net_input(x_i))
        error = (target - output)
        self.w_ += self.eta * 2.0 * x_i * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error ** 2
        return loss
    
    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.activation(self.net_input(X)) >= .5, 1, 0)

    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.activation(self.net_input(X)) >= .5, 1, 0)

    
    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.activation(self.net_input(X)) >= .5, 1, 0)

    
    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.activation(self.net_input(X)) >= .5, 1, 0)

    
    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.activation(self.net_input(X)) >= .5, 1, 0)

    
    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.activation(self.net_input(X)) >= .5, 1, 0)

    
    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.activation(self.net_input(X)) >= .5, 1, 0)

    
    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.activation(self.net_input(X)) >= .5, 1, 0)

    
    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.activation(self.net_input(X)) >= .5, 1, 0)


if __name__ == "__main__":

    N_ITER = 15
    ETA = .01

    df = read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                  header=None, encoding='utf-8')

    # select setosa and versicolor
    y = df.iloc[:100, 4].values
    y = where(y == 'Iris-setosa', 0, 1)
    # extract sepal length and petal length (features)
    X = df.iloc[:100, [0, 2]].values

    ada = AdalineSGD(eta=ETA, n_iter=N_ITER)
    X_std = copy(X)
    X_std[:, 0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:, 1] = (X[:,1] - X[:,0].mean()) / X[:,0].std()

    ada.fit(X_std, y)

    print(f'The model was training using n_iter={N_ITER} and eta={ETA}')
    losses = ada.losses_
    for l in losses:
        print(f'{l:.4f}')
    if l <= .05:
        print("Yay! The algorith did converge!")
    else:
        print("Doh! The algorith did not converge! :(")
