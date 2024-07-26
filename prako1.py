import pandas as pd
import seaborn as sns
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
'''load your data set'''
prako=pd.read_csv('heart.csv')
#prako=sns.load_dataset('diamonds')
#prako=pd.read_csv('Mall_Customers.csv')
print(prako.head(15))
print(prako.shape)
print(prako.describe())
prako.plot(kind='box',subplots=True,sharex=False,sharey=False)
plt.show()
prako.hist()
plt.show()
scatter_matrix(prako)
plt.show()
array= prako.values
X=array[:,0:1]
y=array[:,1]

X_train,X_validation,Y_train,Y_validation= train_test_split(X,y,test_size=0.2,random_state=0)
print("Size of training set:", X_train.shape)
print("Size of test set:", X_validation.shape)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
plt.boxplot(results,labels=names)
plt.title('Algorithm Comparison')
plt.show()

#make prediction on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


