#Data preprocessing
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#upload dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])
ohe = OneHotEncoder(categorical_features= [3])
X = ohe.fit_transform(X).toarray()

np.set_printoptions(threshold = np.nan);

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

#train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

def backwardElimination(X, sl):
    numVars = len(X[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, X).fit()
        max_p = max(regressor_OLS.pvalues).astype(float)
        if max_p > sl :
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == max_p):
                    X = np.delete(X, j, 1)
    return X

X_modeled = backwardElimination(X, 0.05)


           












