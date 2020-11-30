import numpy as np
from Manan_Mukim_Assignment_10_1700146C202.Codes.Problem_4_Object_Oriented_Approach.Problem_4_Object_Oriented_Approach import Regression


class Linear_Regression(Regression):
    def __init__(self, epochs, lr, x, y, l1_ratio=None):
        super().__init__(epochs=epochs, lr=lr, l1_ratio=l1_ratio)
        self.x = x
        self.y = y
        np.random.seed(1)
        self.weights = np.random.randn(len(x[0] + 1))

    def fit(self):
        self.weights = super().gradient_descent(x=self.x, y=self.y, weights=self.weights)

    def predict(self, x_test):
        self.weights = list(self.weights)
        c = self.weights.pop(0)
        self.predictions = super().prediction(weights=self.weights, x=x_test)
        self.predictions += c
        return self.predictions
