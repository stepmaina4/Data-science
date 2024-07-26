import pandas as pd
from datetime import datetime,timedelta
'''loading dataset'''
df=pd.read_csv('retail.csv',encoding='iso-8859-1')
print(df.head())

'''viewing the columns headings'''

print(f"column: {df.columns}")


'''identifying the unique number of values in the dataset'''

#print(f":{df.nunique}")

'''checking the missing values in the dataset'''

#print(f":{df.isnull().sum()}")

'''see rows with missing values'''

#print(df[df.isnull().any(axis=1)])

'''viewing the data statistics'''

#print(df.describe())

'''print columns with missing values'''

#print(df.isnull().sum())

'''dropping the missing values'''

df_null=round(100*(df.isnull().sum())/len(df),2)
#print(df_null)

df=df.dropna()
'''displying the dataset without missing values'''

df['CustomerID']=df['CustomerID'].astype(str)
#print(df.isnull().sum())

'''calculate the 'Amount' column'''

df["Amount"]=df["Quantity"]*df["UnitPrice"]
print(df.head())

'''Calculate total amount spent by each customer'''

df2=df.groupby("CustomerID")["Amount"].sum()
#print(df2.head())

df2=df2.reset_index()
#print(df2.head())
''' Calculate the total quantity for each product description'''

df2=df.groupby("Description")["Quantity"].sum()
#print(df2.head())

'''Remove trailing spaces from column names'''

df.columns = df.columns.str.strip()
'''calculating the with country with highest total sales amount'''

df2 = df.groupby("Country")["Amount"].sum()
df2= df2.idxmax()
#print(df2)
'''Calculating the count of invoices for each product description,sorted in descending order'''

df2=df.groupby("Description")["InvoiceNo"].count().sort_values(ascending=False)
#print(df2.head)


'''convert invoiceDate to datetime '''

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
#print(df.info())

''' Calculating the date range in the dataset'''

max_date=max(df["InvoiceDate"])
#print(max_date)
min_date=min(df["InvoiceDate"])
#print(min_date)
days=max_date-min_date
#print(days)

'''calculate the  cutoff date for the last 30 days'''

Tsale=max_date - pd.Timedelta(days=30)
#print(Tsale)


'''total sales'''



total_sales=df[(df['InvoiceDate']>=Tsale)&(df['InvoiceDate']<=max_date)]['Amount'].sum()
#print("Total sales of the last month:",total_sales)


import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pandas.plotting import scatter_matrix

df2=df.groupby("StockCode").agg({"Quantity":"sum","UnitPrice":"sum"}).reset_index()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_3=scaler.fit_transform(df2[["Quantity","UnitPrice"]])

df.plot(kind='box',subplots=True,sharex=False,sharey=False)
plt.show()




from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters = 3,random_state =0,n_init='auto')
kmeans.fit(df_3)
df2["Clusters"]= kmeans.predict(df_3)

from sklearn.metrics import silhouette_score
perf=silhouette_score(df_3,kmeans.labels_,metric="euclidean")
print(perf)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df[['Quantity']],df[['UnitPrice']],test_size=0.3,random_state=0)

from sklearn import preprocessing
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)

'''Testing a number of clusters to determine how many to use'''
K=range(2,8)
fit=[]
score=[]
for k in K:
    '''Train the model for the current value of k on the training model'''
    model=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(df_3)
    fit.append(model)
    score.append(silhouette_score(df_3,model.labels_,metric='euclidean'))
print(fit)
print(score)
'''plotting the elbowplot for comparison'''

sns.lineplot(x=K,y=score)
plt.show()
'''sns.scatterplot(data=x_train,x='UnitPrice',y='Quantity',hue=fit[3].labels_)
#plt.show()
sns.boxplot(x=fit[3].labels_, y=y_train['Description'])
#plt.show()'''





