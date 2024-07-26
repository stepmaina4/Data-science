'''import the necessary libraries'''
import pandas as pd
'''read the csv file named Mall_Customers from the computer'''
mc_df=pd.read_csv('Mall_Customers.csv')
print(mc_df.head())
'''checking how many records are there in the file'''
print(f"Data frame: {mc_df.shape}")
'''checking the features it is composed of'''
print(f"DataFrame_info:{mc_df.info}")
'''checking whether there are any missing values in the file'''
print(f"missing values: {mc_df.isna()}")
''' if there are missing values they could be replaced with either,
mean,mode or the median values'''
x=mc_df["Age"].mean()
mc_df.fillna({"Age":x},inplace = True)
x_1=mc_df["Age"].mode()
mc_df.fillna({"Age":x_1},inplace = True)
x_2=mc_df["Age"].median()
mc_df.fillna({"Age":x_2},inplace = True)
'''print the values of the mean,
mode and the median that can be used to replace any missing values'''
print(f"mean:",x)
print(f"mode:",x_1)
print(f"median:",x_2)
''' Specifying on the columns to use in the description'''
mc2_df=pd.read_csv('Mall_Customers.csv',usecols=['Annual Income (k$)','Age','Spending Score (1-100)','Gender'])
print(mc2_df.head())
'''import the additional necessary libraries'''
from sklearn.model_selection import train_test_split
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
'''showing how anuall income, age and spending score vary with gender'''
plt.figure(figsize=(6,10))
sns.histplot(mc2_df['Gender'],kde=True)
plt.show()
sns.histplot(mc2_df['Annual Income (k$)'],kde=True)
plt.show()
sns.histplot(mc2_df['Spending Score (1-100)'],kde=True)
plt.show()

x_train, x_test, y_train, y_test=train_test_split(mc2_df[['Annual Income (k$)','Age']],mc2_df[['Spending Score (1-100)']],test_size=0.33,random_state=0)
'''Training the data and evaluation'''
from sklearn import preprocessing
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)
from sklearn.cluster import KMeans
mc3_df=KMeans(n_clusters = 3,random_state =0,n_init='auto')
mc3_df.fit(x_train_norm)
'''visualize the result using a scatterplot and a boxplot'''
sns.scatterplot(data=x_train,x='Annual Income (k$)',y='Age',hue=mc3_df.labels_)
plt.show()
'''look at the distribution of the spending score in the 3 groups using a boxplot'''
sns.boxplot(x=mc3_df.labels_, y=y_train['Spending Score (1-100)'])
plt.show()
'''evaluate perfomance of clustering algorithm using a silhouette score
a lower score represents a better fit'''
from sklearn.metrics import silhouette_score
perf=(silhouette_score(x_train_norm,mc3_df.labels_,metric='euclidean'))
print(perf)
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
sns.scatterplot(data=x_train,x='Annual Income (k$)',y='Age',hue=fit[0].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Annual Income (k$)',y='Age',hue=fit[2].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Annual Income (k$)',y='Age',hue=fit[5].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Annual Income (k$)',y='Age',hue=fit[0].labels_)
plt.show()
sns.lineplot(x=K,y=score)
plt.show()
sns.scatterplot(data=x_train,x='Annual Income (k$)',y='Age',hue=fit[3].labels_)
plt.show()
sns.boxplot(x=fit[3].labels_, y=y_train['Spending Score (1-100)'])
plt.show()












