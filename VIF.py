from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

boston_dataset = load_boston()

data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['price'] = boston_dataset.target

prices = np.log(data['price'])
features = data.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, train_size=0.2, random_state=10)

X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

# vif = []
for i in range(len(X_incl_const)):
    print(variance_inflation_factor(exog=X_incl_const.values, exog_idx=i))

# print(vif)
