import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Regression:
    def __init__(self, epochs, lr, l1_ratio=None):
        self.epochs = epochs
        self.learning_rate = lr
        self.costings = []
        self.predictions = []
        self.l1_ratio = l1_ratio

    @staticmethod
    def Mean_Squared_Error(actual, predicted):
        error = actual - predicted
        return 1 / (len(actual)) * np.dot(error.T, error)

    @staticmethod
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / len(actual) * 100.0

    def gradient_descent(self, x, y, weights, switch=0):
        for _ in range(self.epochs):
            prediction = np.dot(x, weights)
            if switch == 0:
                pass
            else:
                # print("Log")
                prediction = Regression.sigmoid_activation(prediction)
            error = prediction - y
            if switch == 0:
                # print("Linear")
                cost = 1 / (len(x)) * np.dot(error.T, error)
            else:
                # print("Log-1")
                cost = Regression.log_loss(y, prediction)
            self.costings.append(cost)
            if self.l1_ratio is None:
                weights = weights - (self.learning_rate * (1 / len(x)) * np.dot(x.T, error))
            else:
                weights = weights - (self.learning_rate * (x.T.dot(error) + self.l1_ratio * np.sign(weights) + (1 - self.l1_ratio) * 2 * weights) * 1 / len(x))
            # print(weights)
        return weights

    @staticmethod
    def log_loss(y, prediction):
        cost = (-y) * np.log(prediction) - \
               (1 - y) * np.log(1 - prediction)
        cost = sum(cost) / len(y)
        return cost

    @staticmethod
    def sigmoid_activation(prediction):
        return 1 / (1 + np.exp(-prediction))

    def cost_plot(self):
        plt.title('Cost Function')
        plt.xlabel('No. of iterations')
        plt.ylabel('Cost')
        plt.plot(self.costings)
        plt.show()

    def prediction(self, weights, x):
        print(len(x))
        for i in x:
            s = 0
            for j in range(0, len(i)):
                s = s + (weights[j] * i[j])
            self.predictions.append(s)
        return self.predictions
