import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

boston_dataset = load_boston()

data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['price'] = boston_dataset.target

## Modified model: transformed (log) and simplified (dropped 2 features)
prices = np.log(data['price'])
features = data.drop(['price', 'INDUS', 'AGE'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, train_size=0.2, random_state=10)

X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

## Finding residuals
# residuals = y_train - results.fittedvalues
# print(residuals)
# print(results.resid)

## Graph of actual vs predicted prices
# corr = round(y_train.corr(results.fittedvalues), 2)
# plt.scatter(x=y_train, y=results.fittedvalues, c='navy', alpha=0.6)
# plt.plot(y_train, y_train, color='cyan')
# plt.xlabel('Actual log prices $y _i$', fontsize=14)
# plt.ylabel('predicted log prices $\hat y _i$', fontsize=14)
# plt.title(f'Actual vs predicted log prices: $y _i$ vs $\hat y _i$ (corr {corr})', fontsize=17)
# plt.show()


## Graph of the Actual prices
# plt.scatter(x=np.e**y_train, y=np.e**results.fittedvalues, c='blue', alpha=0.6)
# plt.plot(np.e**y_train, np.e**y_train, color='cyan')
# plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
# plt.ylabel('Predicted prices 000s $\hat y _i$', fontsize=14)
# plt.title(f'Actual vs predicted prices: $y _i$ vs $\hat y _i$ (corr {corr})', fontsize=17)
# plt.show()


# plt.scatter(x=results.fittedvalues, y=np.e**results.resid, c='navy', alpha=0.6)
# plt.xlabel('Predicted log prices $\hat y _i$', fontsize=14)
# plt.ylabel('Residuals', fontsize=14)
# plt.title('Residuals vs Fitted Values', fontsize=17)
# plt.show()

## Distribution of Residuals (Log prices) - checking for normality
resid_mean = round(results.resid.mean(), 2)
resid_skew = round(results.resid.skew(), 2)
sns.distplot(results.resid, color='navy')
plt.title(f'Log price model: Residuals Skew ({resid_skew}) Mean ({resid_mean})')
plt.show()
