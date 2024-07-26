'''import the necessary library in this case pandas'''
import pandas as pd
'''read a csv file named 'sample.csv' from the computer,into a pandas DataFrame'''
df = pd.read_csv('sample.csv')
'''show the only 10 rows of the pandas DataFrame'''
print(df.to_string(index=False,header=True,max_rows=10))
'''import pandas as pd

df = pd.read_csv('data.csv')

df.dropna(inplace = True)

print(df.to_string())
print (df)
print(df.to_string())
import pandas as pd

df = pd.read_csv('data.csv')

df.fillna(130, inplace = True)
print(df.to_string())
import pandas as pd

df = pd.read_csv('data.csv')

df.fillna({"calories":130}, inplace = True)

df = pd.read_csv('data.csv')

x = df["Calories"].mean()

df.fillna({"calories":x}, inplace = True)
print(df.to_string())

df = pd.read_csv('data.csv')

x = df["Calories"].median()

df.fillna({"calories":x}, inplace = True)

df = pd.read_csv('data.csv')

x = df["Calories"].mode()[0]

df.fillna({"calories":x}, inplace = True)
import pandas as pd

df = pd.read_csv('data.csv')

x = df["Calories"].mean()

df["Calories"].fillna(x, inplace = True)

print(df.to_string())
import pandas as pd

df = pd.read_csv('data.csv')

x = df["Calories"].median()

df["Calories"].fillna(x, inplace = True)
print(df.to_string())
import pandas as pd

df = pd.read_csv('data.csv')

x = df["Calories"].mode()[0]

df["Calories"].fillna(x, inplace = True)
print(df.to_string())
df.loc[7, 'Duration'] = 45
print(df.to_string())
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120
    print(df.to_string())
    for x in df.index:
       if df.loc[x, "Duration"] > 120:
          df.drop(x, inplace = True)
          print(df.to_string())'''






