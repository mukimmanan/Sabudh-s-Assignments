import numpy as np
from Manan_Mukim_Assignment_10_1700146C202.Codes.Problem_4_Object_Oriented_Approach.Problem_4_Object_Oriented_Approach import Regression


class Logistic_Regression(Regression):
    def __init__(self, epochs, lr, x, y, l1_ratio=None):
        super().__init__(epochs=epochs, lr=lr, l1_ratio=l1_ratio)
        self.x = x
        self.y = y
        np.random.seed(1)
        self.weights = np.zeros(len(x[0]))

    def fit(self):
        self.weights = super().gradient_descent(x=self.x, y=self.y, weights=self.weights, switch=1)

    def predict(self, x_test):
        self.predictions = super().prediction(weights=self.weights, x=x_test)
        self.predictions = [round(Regression.sigmoid_activation(value)) for value in self.predictions]
        return self.predictions
