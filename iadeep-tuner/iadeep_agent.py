import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

def cubic_regression(a, b, c, d, X):
    return a * np.power(X, 3) + b * np.power(X, 2) + c * X + d

class GPLCB(object):
    def __init__(self, x, mu, sigma):
        self.X_grid = x
        self.X = []
        self.T = []
        self.mu = mu
        self.sigma = sigma
        self.cubic_models = []

    def argmin_lcb(self, times):
        beta = np.sqrt(2 * np.log(len(self.X)) / (times * times))
        arr = abs(self.mu - self.sigma * beta)
        return np.argmin(arr), min(arr)

    def learn(self, times):
        grid_idx, target = self.argmin_lcb(times)

        self.sample(self.X_grid[grid_idx], target)
        kernel = DotProduct() + WhiteKernel()
        gp = GaussianProcessRegressor(kernel, random_state=0)
        T = np.reshape(self.T, -1)
        gp.fit(self.X, T)
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)
        return grid_idx

    def sample(self, x, t):
        self.X.append(x)
        self.T.append(t)

