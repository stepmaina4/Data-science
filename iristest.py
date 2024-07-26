'''import the necessary libraries in this case pandas and seaborn'''
import pandas as pd
import seaborn as sns
'''read the data stored by the name"iris" in a file of the given website,
in our case"https://archive.ics.uci.edu/dataset/53/iris"using seaborn library'''
iris_df=sns.load_dataset('iris')
'''show the Dataframe using pandas library'''
print(iris_df.tail())
'''output;
    sepal_length  sepal_width  petal_length  petal_width    species
145           6.7          3.0           5.2          2.3  virginica
146           6.3          2.5           5.0          1.9  virginica
147           6.5          3.0           5.2          2.0  virginica
148           6.2          3.4           5.4          2.3  virginica
149           5.9          3.0           5.1          1.8  virginica'''
'''show the first 8 records'''
print(iris_df.head(8))
'''output;
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
5           5.4          3.9           1.7          0.4  setosa
6           4.6          3.4           1.4          0.3  setosa
7           5.0          3.4           1.5          0.2  setosa'''
'''show the number of records  and their features'''
print(f"DataFrame columns : {iris_df.info()}\n=================================")
'''output;
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object 
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
DataFrame info : None
================================='''
'''show the existence of  missing values'''
print(f" missing values: {iris_df.isna()}\n=================================")
'''output;
missing values:      sepal_length  sepal_width  petal_length  petal_width  species
0           False        False         False        False    False
1           False        False         False        False    False
2           False        False         False        False    False
3           False        False         False        False    False
4           False        False         False        False    False
..            ...          ...           ...          ...      ...
145         False        False         False        False    False
146         False        False         False        False    False
147         False        False         False        False    False
148         False        False         False        False    False
149         False        False         False        False    False

[150 rows x 5 columns]
================================='''
'''Import the necessary libraries to show relationships using histogram,
line grapgh and scatter plots'''
import matplotlib.pyplot as plt
import plotly.express as px
'''used a histogram to show the count of sepal_length'''
fig = px.histogram(iris_df,x='sepal_length')
fig.show()
'''used a scatterplot to show the relationship between species and petal width'''
fig =px.scatter(iris_df,x='species', y='petal_width')
fig.show()
fig =px.line(iris_df,x='species', y='petal_length')
fig.show()















