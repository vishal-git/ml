# 02_adaline.py
# Optional command line arguments:
#   1. Learning rate (between 0.0 and 1.0)
#   2. Flag to denote standardizing the trainig data (1 or 0)

from numpy import random, float_, dot, where, zeros, copy
from pandas import read_csv
import sys

class AdalineGD:
    def __init__(self, eta=.01, n_iter=50, random_state=314):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=.0, scale=.01, size=X.shape[1])

        self.b_ = float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean() # mean squared error
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def activation(self, X):
        return X # linear activation

    def predict(self, X):
        return where(self.net_input(X) >= 0.5, 1, 0)

if __name__ == "__main__":

    N_ITER = 20
    ETA = .5

    df = read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                  header=None, encoding='utf-8')

    # select setosa and versicolor
    y = df.iloc[:100, 4].values
    y = where(y == 'Iris-setosa', 0, 1)
    # extract sepal length and petal length (features)
    X = df.iloc[:100, [0, 2]].values

    if len(sys.argv) > 1:
        ETA = float(sys.argv[1])
        if (ETA >= 0.0) & (ETA <= 1.0):
            ada = AdalineGD(eta=ETA, n_iter=N_ITER)
        else:
            raise ValueError('Please provide a valid value for the learning rate (between 0 and 1).')
    else:
        ada = AdalineGD(eta=ETA, n_iter=N_ITER)

    if len(sys.argv) > 2:
        std = sys.argv[2]
        if std == '1':
            print('The training data will be standardized.')
            X_std = copy(X)
            X_std[:, 0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
            X_std[:, 1] = (X[:,1] - X[:,0].mean()) / X[:,0].std()

            ada.fit(X_std, y)
        else:
            ada.fit(X, y)
    else:
        ada.fit(X, y)

    print(f'The model was training using n_iter={N_ITER} and eta={ETA}')
    losses = ada.losses_
    for l in losses:
        print(f'{l:.4f}')
    if l <= .01:
        print("Yay! The algorith did converge!")
    else:
        print("Doh! The algorith did not converge! :(")
