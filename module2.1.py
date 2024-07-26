import pandas as pd
#loading dataset
df=pd.read_csv('retail_sales.csv')
print(df.head())
#check data set information
print(f"information:{df.info}")
#check missing values
print(f"missing values:{df.isna()}")
#data set description
print(df.describe())
#specify on the columns to work on
df_2=pd.read_csv('retail_sales.csv',usecols=['Gender','Age','Product Category','Quantity','Price per Unit','Total Amount'])
print(df_2.head())
#use histogram for visualization
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
'''gender distribution'''
df_2.hist()
plt.show()
sns.histplot(df_2['Gender'],kde=True)
#plt.show()
sns.histplot(df_2['Quantity'],kde=True)
#plt.show()
sns.boxplot(df_2,x='Age',y='Product Category')
#plt.show()
sns.boxplot(df_2,x='Product Category',y='Product Category')
plt.show()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df_2[['Age','Price per Unit']],df_2[['Total Amount']],test_size=0.33,random_state=0)
from sklearn import preprocessing
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)
from sklearn.cluster import KMeans
df_3=KMeans(n_clusters = 3,random_state =0,n_init='auto')
df_3.fit(x_train_norm)
sns.scatterplot(data=x_train,x='Age',y='Price per Unit',hue=df_3.labels_)
#plt.show()
from sklearn.metrics import silhouette_score
perf=(silhouette_score(x_train_norm,df_3.labels_,metric='euclidean'))
#print(perf)
'''Testing a number of clusters to determine how many to use'''
K=range(2,8)
fit=[]
score=[]
for k in K:
    '''Train the model for the current value of k on the training model'''
    model=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(x_train_norm)
    fit.append(model)
    score.append(silhouette_score(x_train_norm,model.labels_,metric='euclidean'))
print(fit)
print(score)
'''Visualize the models for k=2,k=4,k=7,k=5'''
sns.scatterplot(data=x_train,x='Age',y='Price per Product',hue=fit[0].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Age',y='Price per Product',hue=fit[2].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Age',y='Price per Product',hue=fit[5].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Age',y='Price per Product',hue=fit[0].labels_)
plt.show()
sns.lineplot(x=K,y=score)
plt.show()
sns.scatterplot(data=x_train,x='Price per Product',y='Age',hue=fit[3].labels_)
plt.show()
sns.boxplot(x=fit[3].labels_, y=y_train['Total Amount'])
plt.show()










