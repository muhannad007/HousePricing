from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Gather Data
boston_dataset = load_boston()

## Data exploration and cleaning
# Create a pandas dataFrame
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['price'] = boston_dataset.target # adding column with the price (witch is the target)

## Descriptive Statistics
# print(boston_dataset.DESCR)   # Returns the description of each feature
# print(pd.isnull(data).any())  # Returns the null values
# print(data.head())            # Returns the first five rows
# print(data.count())           # Returns the number of elements in each feature
# print(data.mean())            # Returns the mean value of each feature
# print(data.max())             # Returns the maximum value of each feature
# print(data.min())             # Returns the minimum value of each feature
# print(data.median())          # Returns the median value of each feature
# print(data.describe())        # Returns all the stuff above

## vusualising data
# freq = data['RAD'].value_counts()
# plt.figure(figsize=(10, 6))
# plt.grid()
# plt.title('Houses\' Accessibility to Highways in Boston', fontsize=30)
# plt.xlabel('Accessibility to Highways', fontsize=20)
# plt.xlim(0, 25)
# plt.ylabel('Number of Houses', fontsize=20)
# # plt.hist(data['RAD'], bins=50, ec='black', color='#2196f3')
# # sns.distplot(data['RM'], color='#2196f3')
# plt.bar(freq.index, height=freq)
# plt.show()

# print(boston_dataset.data)
# print(dir(boston_dataset))
# print(boston_dataset.data.shape)
# print(boston_dataset.feature_names)
# print(boston_dataset.target)

