import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

'''load the dataset'''
data=pd.read_csv('heart.csv')
print(data.head())
'''printing the dimensions of the data'''
print(f"shape:{data.shape}")
'''viewing the columns headings'''

print(f"column: {data.columns}")

'''inspecting the target variable'''

#print(f":{data.HeartDisease.value_counts}")
#print(f":{data.dtypes}")

'''identifying the unique number of values in the dataset'''

#print(f":{data.nunique}")

'''checking the missing values in the dataset'''

#print(f":{data.isnull().sum()}")

'''plotting correlation between heartdisease and age'''

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(x="HeartDisease", y="Age", data=data)
plt.subplot(1,2,2)
sns.violinplot(x="HeartDisease", y="Age", data=data)
#plt.show()




'''  Distribution density plot KDE (kernel density estimate)'''
sns.FacetGrid(data, hue="HeartDisease", height=6).map(sns.kdeplot, "Age").add_legend()
#plt.show()

'''  Plotting the distribution of the Age'''

sns.stripplot(x="HeartDisease", y="Age", data=data, jitter=True, edgecolor="auto")
#plt.show()


'''Spliting target variable and independent variables'''

X = data.drop(['HeartDisease'], axis = 1)
y = data['HeartDisease']

''' Splitting the data into training set and testset'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
#print("Size of training set:", X_train.shape)
#print("Size of test set:", X_test.shape)

train_set = pd.concat([X_train,y_train],axis=1)
#print(train_set.head())

num = [i for i in train_set.columns if train_set[i].dtype != 'object']
print(num)

plt.figure(figsize=(16, 18))
sns.heatmap(train_set[num].corr(), cbar=True,square=True,annot=True, annot_kws={'size':15},cmap='Reds')
plt.show()











