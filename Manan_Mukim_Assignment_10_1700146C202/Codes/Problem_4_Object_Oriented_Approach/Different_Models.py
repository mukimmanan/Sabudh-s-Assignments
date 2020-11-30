from Manan_Mukim_Assignment_10_1700146C202.Codes.Problem_4_Object_Oriented_Approach.Linear_Regression import Linear_Regression
from Manan_Mukim_Assignment_10_1700146C202.Codes.Problem_4_Object_Oriented_Approach.Logistic_Regression import Logistic_Regression
import numpy as np
import pandas as pd

# Linear Regression Using Gradient Descent
print("=" * 40)
print("Linear Regression Using Gradient Descent")
data = pd.read_csv("E:\\Machine Learning  Project\\College\\Manan_Mukim_Assignment_10_1700146C202\\Codes\\Linear_Regression_Dataset.csv")

v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

model = Linear_Regression(1000, 0.021, X_train, Y_train)
model.fit()
model.cost_plot()
predictions = model.predict(X_test)
print("Predictions")
print(predictions)
print("Original Values")
print(Y_test)
print("Means Squared Error : ", Linear_Regression.Mean_Squared_Error(Y_test, predictions))

# Logistic Regression Using Gradient Descent
print("=" * 40)
print("Linear Regression Using Gradient Descent")
data = pd.read_csv("E:\\Machine Learning  Project\\College\\Manan_Mukim_Assignment_10_1700146C202\\Codes\\Logistic_Regression_Dataset.csv")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

model = Logistic_Regression(1000, 0.021, X_train, Y_train)
model.fit()
model.cost_plot()
predictions = model.predict(X_test)
print("Predictions")
print(predictions)
print("Original Values")
print(Y_test)
print("Accuracy Score : ", Logistic_Regression.accuracy_metric(Y_test, predictions))


# Linear Regression With L1_L2_Regularization
print("=" * 40)
print("Linear Regression With L1 And L2 Regularization")
data = pd.read_csv("E:\\Machine Learning  Project\\College\\Manan_Mukim_Assignment_10_1700146C202\\Codes\\Linear_Regression_Dataset.csv")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

model = Linear_Regression(1000, 0.021, X_train, Y_train, l1_ratio=0.5)
model.fit()
model.cost_plot()
predictions = model.predict(X_test)
print("Predictions")
print(predictions)
print("Original Values")
print(Y_test)
print("Means Squared Error : ", Linear_Regression.Mean_Squared_Error(Y_test, predictions))

# # Logistic Regression With L1_L2_Regularization
data = pd.read_csv("E:\\Machine Learning  Project\\College\\Manan_Mukim_Assignment_10_1700146C202\\Codes\\Logistic_Regression_Dataset.csv")
print("=" * 40)
print("Logistic Regression With L1 And L2 Regularization")
v = len(data.columns)
X = data.iloc[:, 1:v - 1].values
Y = data.iloc[:, v - 1].values

# Splitting Data_Sets
val = len(X) // 3
X_train = X[val:]
X_test = X[: val]
Y_train = Y[val:]
Y_test = Y[: val]

model = Logistic_Regression(1000, 0.021, X_train, Y_train, 0.5)
model.fit()
model.cost_plot()
predictions = model.predict(X_test)
print("Predictions")
print(predictions)
print("Original Values")
print(Y_test)
print("Accuracy Score : ", Logistic_Regression.accuracy_metric(Y_test, predictions))
