import matplotlib.pyplot as plt
import numpy  as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
'''get the Data'''
x =np.arange(10).reshape((-1,1))
y=np.array([0,0,0,0,1,1,1,1,1,1])
'''View the Data'''
print('for x,we have:',x)
print('for y we have:',y)
'''create a model and train it'''
ex4_2=LogisticRegression(solver='liblinear',random_state=0)
'''fit the model, or train it'''
ex4_2.fit(x,y)
'''evaluate/validate/confirm the model'''
print('The probability that the output is 0 or 1 is:',ex4_2.predict_proba(x))
'''the actual prediction'''
print('these are the predictions:')
print(ex4_2.predict(x))
'''accuracy'''
print(ex4_2.score(x,y))
'''Confusion Matrix,it provides the actual and predicted outputs'''
print('the confusion matrix is:',confusion_matrix(y,ex4_2.predict(x)))
'''Visualize'''
cm = confusion_matrix(y,ex4_2.predict(x))
fig,ax = plt.subplots(figsize=(8,8))
ax.grid(False)
ax.imshow(cm)
ax.xaxis.set(ticks=(0,1),ticklabels=('predicted 0s','predicted 1s'))
ax.yaxis.set(ticks=(0,1),ticklabels=('actual 0s','actual 1s'))
ax.set_ylim(1.5,-0.5)
for i in range(2):
 for j in range(2):
    ax.text(j,i,cm[i,j],ha='center',va='center',color='blue')
'''plt.show()'''
print(classification_report(y,ex4_2.predict(x)))


