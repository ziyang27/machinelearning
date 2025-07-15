import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegresion

X, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], y_train, color='blue', label='Training data', marker='o', s=30)
plt.show()

model = LinearRegresion(lr=0.01)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_value = mse(y_test, y_pred)
print(f'Mean Squared Error: {mse_value}')   

y_pred_line = model.predict(X_train)
cmap = plt.get_cmap('viridis')
fig = plt
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=30)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=30)
plt.plot(X_train, y_pred_line, color='red', linewidth=2, label='Regression Line')
plt.show()