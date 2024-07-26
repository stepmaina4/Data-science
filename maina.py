import plotly.express as px
import matplotlib.pyplot as plt
x=[1,2,3,4]
y=[1,2,3,4]
data = {'x':x, 'y':y}
fig = px.line(data,x='x',y='y')
fig.show()
import seaborn as sns
diamonds_df=sns.load_dataset('diamonds')
print(diamonds_df.head())
fig = px.line(diamonds_df, x='carat', y='price')
fig.show()
fig = px.histogram(diamonds_df,x='cut')
fig.show()
fig =px.violin(diamonds_df,x='cut', y='price')
fig.show()
fig =px.violin(diamonds_df,x='cut', y='carat')
fig.show()
fig =px.scatter(diamonds_df,x='cut', y='price')
fig.show()
fig =px.violin(diamonds_df,x='carat', y='price')
fig.show()
fig =px.violin(diamonds_df,x='cut', y='carat')
fig.show()
