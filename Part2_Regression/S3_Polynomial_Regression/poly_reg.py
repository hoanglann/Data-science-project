# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fit Polynomial Regression to X
from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree = 4)
X_poly = poly_feature.fit_transform(X)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_poly, y)

#Predict for x = 6.5
y_pred = regressor.predict(poly_feature.fit_transform(6.5))
diff = 160000 - y_pred

#Visualize
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(poly_feature.fit_transform(X_grid)), color = 'blue')
plt.show()

