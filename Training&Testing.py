from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Gather Data
boston_dataset = load_boston()

## Data exploration and cleaning
# Create a pandas dataFrame
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['price'] = boston_dataset.target # adding column with the price (which is the target)

prices = data['price']
features = data.drop('price', axis=1)

## shuffling the data with 80% for test date and 20% for the training data
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, y_train)

print('Intercept', regr.intercept_)
print('The coefficients of our model are: ')
print(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']))

print('Training data r_squared: ', regr.score(X_train, y_train))
print('Testing data r_squared: ', regr.score(X_test, y_test))

print(data['price'].skew()) # Skew is having more data points in one of the tails

## Data transformation
y_log = np.log(data['price'])
# sns.distplot(y_log)
# plt.title(f'Log price with skew {y_log.skew()}')
# sns.lmplot(x='LSTAT', y='price', data=data, scatter_kws={'alpha': 0.6}, line_kws={'color': 'darkred'})
# plt.show()

## Data transformed
transform_data = features
transform_data['LOG_PRICE'] = y_log
sns.lmplot(x='LSTAT', y='LOG_PRICE', data=transform_data, size=7, scatter_kws={'alpha': 0.6}, line_kws={'color': 'darkred'})
plt.show()
