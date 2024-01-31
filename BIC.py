import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston_dataset = load_boston()

data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['price'] = boston_dataset.target

prices = np.log(data['price'])
features = data.drop(['price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, train_size=0.2, random_state=10)

X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

# org_coef = pd.DataFrame({'coef': results.params, 'p-values': round(results.pvalues, 3)})

## Finding the baysian information criterion (BIC)
# print('BIC is: ', results.bic)
# print('R_squared is: ', results.rsquared)

## Reduced model #1 excluding INDUS
# X_incl_const = X_incl_const.drop(['INDUS'], axis=1)
# model = sm.OLS(y_train, X_incl_const)
# results = model.fit()
#
# coef_minus_indus = pd.DataFrame({'coef': results.params, 'p-values': results.pvalues})
#
# print('This is the new BIS: ', results.bic)
# print('This is the new R-squared: ', results.rsquared)

## Reduced model #2 excluding INDUS and AGE
X_incl_const = X_incl_const.drop(['INDUS', 'AGE'], axis=1)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

coef_minus_indus_AGE = pd.DataFrame({'coef': results.params, 'p-values': round(results.pvalues, 3)})

print('This is the new BIS: ', results.bic)
print('This is the new R-squared: ', results.rsquared)
print(coef_minus_indus_AGE)
