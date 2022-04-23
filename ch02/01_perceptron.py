from numpy import random, float_, dot, where, zeros
from pandas import read_csv
import sys

class Perceptron:
    def __init__(self, eta=.01, n_iter=50, weight_init='random', random_state=314):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.weight_init = weight_init

    def fit(self, X, y):
        rgen = random.RandomState(self.random_state)
        if self.weight_init == 'random':
            self.w_ = rgen.normal(loc=.0, scale=.01, size=X.shape[1])
            print(f'Initial weights: {self.w_}')
        else:
            self.w_ = zeros(X.shape[1])
            print(f'Initial weights: {self.w_}')

        self.b_ = float_(0.)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for x_i, target in zip(X, y):
                update = self.eta * (target - self.predict(x_i))
                self.w_ += update * x_i
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return dot(X, self.w_) + self.b_

    def predict(self, X):
        return where(self.net_input(X) >= 0.0, 1, 0)

if __name__ == "__main__":
    df = read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                  header=None, encoding='utf-8')

    # select setosa and versicolor
    y = df.iloc[:100, 4].values
    y = where(y == 'Iris-setosa', 0, 1)
    # extract sepal length and petal length (features)
    X = df.iloc[:100, [0, 2]].values
    if len(sys.argv) > 1:
        weight_init = sys.argv[1]
        if weight_init == 'uniform':
            print(f'Weight initialization: {weight_init}')
            ppn = Perceptron(eta=.1, n_iter=10, weight_init=weight_init)
        elif weight_init in ['random', 'normal']:
            print(f'Weight initialization: {weight_init}')
            ppn = Perceptron(eta=.1, n_iter=10)
        else:
            raise ValueError('Please provide a valid weight-initialization approach: normal, random, uniform')
    else:
        print('Weight initialization: normal (default)')
        ppn = Perceptron(eta=.1, n_iter=10)

    ppn.fit(X, y)
    print(ppn.errors_)
    if ppn.errors_[-1] == 0:
        print("Yay! The algorith did converge!")
    else:
        print("Doh! The algorith did not converge! :(")
