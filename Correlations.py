# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Gather Data
# boston_dataset = load_boston()
boston_dataset = fetch_california_housing()
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# boston_dataset = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

## Data exploration and cleaning
# Create a pandas dataFrame
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['price'] = boston_dataset.target # adding column with the price (witch is the target)

## Finding correlations (-1<rho<1)
# print(data['price'].corr(data['RM']))       # Calculating the correlation between the property price and the number of rooms
# print(data['price'].corr(data['PTRATIO']))  # Calculating the correlation between the price and the pupil teacher ratio
# print(data.corr())                          # Calculating the correlations between all the features

# visualising our correlations
mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize=(16, 10))
sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={"size": 14})
sns.set_style('white')
plt.title('Correlations Table', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

## Correlation between the dinstance from industrial (DIS) areas and air pollution (NOX)
# nox_dis_corr = round(data['NOX'].corr(data['DIS']), 3)
# plt.figure(figsize=[16, 9])
# plt.title(f'DIS vs NOX Correlation {nox_dis_corr}', fontsize=20)
# plt.xlabel('DIS - Distance from employment centers', fontsize=16)
# plt.ylabel('NOX - Nitric Oxide Pollution', fontsize=16)
# plt.scatter(x=data['DIS'], y=data['NOX'], alpha=0.6, s=80, color='indigo')
# plt.show()

## Correlation between Taxes and the Accessibilty to highways
# sns.set_context('talk')
# sns.set_style('whitegrid')
# sns.jointplot(x=data['TAX'], y=data['RAD'], size=8, color='darkred', joint_kws={"alpha": 0.5})
# sns.lmplot(x='TAX', y='RAD', data=data, size=8) # This will show the correlations between TAX and RAD
# plt.show()

# sns.pairplot(data)  # This will create a scatter plot between all the features and the target
# plt.show()
