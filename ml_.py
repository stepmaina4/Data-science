import pandas as pd
house_df=pd.read_csv('house.csv')
print(house_df.tail())
import plotly.express as px
fig=px.line(house_df,x='housing_median_age',y='median_income')
fig.show()
