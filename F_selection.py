import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

'''load the breast cancer dataset'''

df=pd.read_csv('house.csv')
print(df.head(10))
'''leave one out cross validation'''
from sklearn.model_selection import train_test_split,LeaveOneOut,cross_val_score
'''Defining predictor variable'''
X=df.drop(['median_house_value'],axis=1)
y=df['median_house_value']

'''Initialize LOOCV'''
cv=LeaveOneOut()

'''Build a multiple logistic regression model'''
model=LogisticRegression()

'''Evaluate model using LOOCV and calculate mean absolute error(MAE)'''
scores=cross_val_score(model, X,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
rmse=np.sqrt(np.mean(np.abs(scores)))
print(f"Root mean squared(rmse):{rmse:.6f}")






