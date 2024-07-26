import pandas as pd
import seaborn as sns
diamonds_df=sns.load_dataset('diamonds')
print(diamonds_df.tail)
'''This dataset contains the prices and other attributes of almost 54000 diamonds.
...the contents of the dataset are price,carat,cut.'''
print(diamonds_df.head())
print(f" DataFrame size :{diamonds_df.size}\n===========================")
print(f" DataFrame shape :{diamonds_df.shape}\n===========================")
'''there are 53940 records each being describes by 10 features'''
'''do all records have features?'''
print(f" DataFrame info :{diamonds_df.info}\n=================")
print("this is what we have:")
print(diamonds_df.info())
print("this is the description means,std,minimum,percentiles and the maximum:")
print(diamonds_df.describe())
print(diamonds_df.isnull())
print(diamonds_df.isnull().sum())
import matplotlib as plt





