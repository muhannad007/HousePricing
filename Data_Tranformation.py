## Regression using log prices
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

# Gather Data
boston_dataset = load_boston()

## Data exploration and cleaning
# Create a pandas dataFrame
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['price'] = boston_dataset.target # adding column with the price (which is the target)

prices = np.log(data['price'])
features = data.drop('price', axis=1)

## shuffling the data with 80% for test date and 20% for the training data
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

# regr = LinearRegression()
# regr.fit(X_train, y_train)

# print('Training data r-squared: ', regr.score(X_train, y_train))
# # print('Test data r-squared: ', regr.score(X_test, y_test))
# # print('Intercept', regr.intercept_)
# # print('The coefficients of our model are: ')
# # print(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']))


## Finding p valures
new_features = sm.add_constant(X_train)
model = sm.OLS(y_train, new_features)
results = model.fit()
# print(results.params)
# print(round(results.pvalues, 3))
pd.DataFrame({'coef': results.params, 'p-values': round(results.pvalues, 3)})
