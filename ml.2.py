import pandas as pd
house_df=pd.read_csv('house.csv',usecols=['longitude','latitude','median_house_value'])
print(house_df.head())
'''Normalize the Data'''
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
x_train, x_test, y_train, y_test=train_test_split(house_df[['latitude','longitude']],house_df[['median_house_value']],test_size=0.33,random_state=0)
'''then normalize'''
from sklearn import preprocessing
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)
'''Fitting/Training and Evaluation'''
from sklearn.cluster import KMeans
ex_3=KMeans(n_clusters = 3,random_state =0,n_init='auto')
ex_3.fit(x_train_norm)
'''Visualize the result'''
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=ex_3.labels_)
#plt.show()
'''Look at the distribution of the median house prices in the 3 groups.A boxplot can be useful'''
sns.boxplot(x=ex_3.labels_, y=y_train['median_house_value'])
#plt.show()
'''people in the 1st and 3rd cluster have similar distributions of median house value
and are higher than that of the 2nd cluster'''
from sklearn.metrics import silhouette_score
'''Evaluate perfomance of the clustering algorithm using a silhouette score which is part
of the sklearn.metrics.A lower score represents a better fit'''
perf=(silhouette_score(x_train_norm,ex_3.labels_,metric='euclidean'))
#print(perf)
'''How many clusters to use?
we neeed to test a range of them'''
K=range(2,8)
fit=[]
score=[]
for k in K:
    model=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(x_train_norm)
    fit.append(model)
    score.append(silhouette_score(x_train_norm,model.labels_,metric='euclidean'))
'''print(fit)
print(score)
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=fit[0].labels_)
plt.show()
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=fit[2].labels_)
plt.show()
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=fit[5].labels_)
plt.show()
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=fit[0].labels_)
sns.lineplot(x=K,y=score)
plt.show()
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=fit[3].labels_)
plt.show()'''
sns.boxplot(x=fit[3].labels_, y=y_train['medium_house_value'])
plt.show()


