import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
x= [1,2,3,4]
y=[1,2,3,4]
#plt.plot(x,y)
plt.show()
import pandas as pd
diamonds_df=sns.load_dataset('diamonds')
print(diamonds_df.head())
plt.figure(figsize=(6,10))
sns.histplot(diamonds_df['price'],kde=True)
plt.show()
print("the histogram shows that the lower prices of the diamonds,are more frequent than the higher prices")
sns.histplot(diamonds_df['carat'],kde=True)
plt.show()
sns.histplot(diamonds_df['cut'],kde=True)
plt.show()
sns.histplot(diamonds_df['color'],kde=True)
plt.show()
sns.histplot(diamonds_df['depth'],kde=True)
plt.show()
sns.histplot(diamonds_df['clarity'],kde=True)
plt.show
sns.histplot(diamonds_df['depth'],kde=True)
plt.show()
sns.histplot(diamonds_df['table'],kde=True)
plt.show()
sns.histplot(diamonds_df['x'],kde=True)
plt.show
sns.histplot(diamonds_df['y'],kde=True)
plt.show()
sns.histplot(diamonds_df['z'],kde=True)
plt.show

           
           

