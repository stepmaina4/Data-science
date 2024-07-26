#Time Series Analysis
'''import the necessary libraries'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''generate random time-series data'''
np.random.seed(42)
dates=pd.date_range(start='2022-01-01',periods=100,freq='D')
values=np.random.randn(100).cumsum()
'''create a dataframe from the generated data'''
data=pd.DataFrame({'date':dates, 'value':values})
'''set the date column as the index'''
data.set_index('date',inplace=True)
'''plot the time series'''
plt.plot(data.index,data['value'])
plt.xlabel('time')
plt.ylabel('value')
plt.xticks(rotation=45)
plt.title('Time series data')
plt.show()
'''testing for stationarity'''
from statsmodels.tsa.stattools import adfuller
'''Assuming data is the time series data'''
result=adfuller(data)
print('ADF statistics:',result[0])
print('''The more negative it is, the stronger the rejection of the hypothesis that there is a unit root at some level of confidence,
      Your ADF statistic is -1.3583317659818988, which is not very negative,
      This suggests that the time series might have a unit root, and therefore, it might be non-stationary.''')
print('p-value:',result[1])
print('''our p-value is 0.6020814791099101, which is greater than 0.05, suggesting that you fail to reject the null hypothesis.
This means that the time series might have a unit root, and therefore, it might be non-stationary.''')

