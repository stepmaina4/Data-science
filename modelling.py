#1.Linear Regression
'''import necessary libraries'''
import numpy as np
from sklearn.linear_model import LinearRegression
'''generate the data'''
x=np.array([5,15,25,35,45,55]).reshape((-1,1))
y=np.array([5,20,14,32,22,38])
print('These are values in x:', x)
print('These are values in y:',y)
'''Create a Linear Regression model'''
ex4_1 = LinearRegression()
'''Fit the model to the data'''
ex4_1.fit(x, y)
'''get the results'''
r_sq=ex4_1.score(x,y)
print(r_sq)
'''interpret the result'''
intercept=ex4_1.intercept_
gradient=ex4_1.coef_
print('The y-intercept is is:',intercept)
print('The gradient is:',gradient)
'''apply the results'''
y_pred = ex4_1.predict(x)
print(y_pred)
'''Forecast values'''
x_new=np.arange(5).reshape((-1,1))
print(x_new)
y_new=ex4_1.predict(x_new)
print('The new y values are:',y_new)


