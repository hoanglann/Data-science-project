#Data preprocessing
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#upload dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

#fit simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict on test set
y_pred = regressor.predict(X_test)

#visualize the results on training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Exp (Training set)')
plt.xlabel('Exp (Years)')
plt.ylabel('Salary')
plt.show()

#visualize the results on test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Exp (Test set)')
plt.xlabel('Exp (Years)')
plt.ylabel('Salary')
plt.show()


