import sklearn.svm
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import numpy as np
class Agent:
    def __init__(self, x_grid, agent_type):
        self.x_grid = x_grid
        if agent_type == "rf":
            self.model = sklearn.ensemble.RandomForestRegressor(max_depth=3)
        if agent_type == "svr":
            self.model = sklearn.svm.SVR()
        if agent_type == "lr":
            self.model = sklearn.linear_model.LinearRegression()
        if agent_type == "mlp":
            self.model = sklearn.neural_network.MLPRegressor()
        if agent_type == "sgd":
            self.model = sklearn.linear_model.SGDRegressor()

    def train(self, X, Y):
        self.model.fit(X, Y)

    def select(self):
        x_index = np.argmin(self.model.predict(self.x_grid))
        return self.x_grid[x_index]
