#Model Optimization and Hyperparameter tuning
'''loading the necessary libraries'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

'''load your dataset'''
data=pd.read_csv('bc_data.csv')
print(data.head())

'''printing the dimensions of the data'''
print(f"shape:{data.shape}")

'''viewing the columns headings'''

#print(f"column: {data.columns}")

'''inspecting the target variable'''

#print(f":{data.diagnosis.value_counts}")
#print(f":{data.dtypes}")

'''identifying the unique number of values in the dataset'''

#print(f":{data.nunique}")

'''checking the missing values in the dataset'''

#print(f":{data.isnull().sum()}")

'''dropping some columns which do not provide any useful information for the model'''

data.drop(['Unnamed: 32','id'],axis=1,inplace=True)

'''see rows with missing values'''

#print(data[data.isnull().any(axis=1)])

'''viewing the data statistics'''

#print(data.describe())

'''     DATA VISUALIZATION    '''

'''finding the correlation between features'''

#corr=data.corr()
#print(f":{corr.shape}")

'''analyzing the target variable'''

plt.title('count of cancer type')
sns.countplot(data['diagnosis'])
plt.ylabel('cancer Lethality')
plt.xlabel('count')
plt.show()


'''plotting correlation between diagnosis and radius'''



plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(x="diagnosis", y="radius_mean", data=data)
plt.subplot(1,2,2)
sns.violinplot(x="diagnosis", y="radius_mean", data=data)
plt.show()


''' Plotting correlation between diagnosis and concativity'''

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(x="diagnosis", y="concavity_mean", data=data)
plt.subplot(1,2,2)
sns.violinplot(x="diagnosis", y="concavity_mean", data=data)
plt.show()

'''  Distribution density plot KDE (kernel density estimate)'''
sns.FacetGrid(data, hue="diagnosis", height=6).map(sns.kdeplot, "radius_mean").add_legend()
plt.show()

'''  Plotting the distribution of the mean radius'''

sns.stripplot(x="diagnosis", y="radius_mean", data=data, jitter=True, edgecolor="auto")
plt.show()

'''Plotting bivariate relations between each pair of features (4 features x4 so 16 graphs) with hue = "diagnosis"'''

sns.pairplot(data, hue="diagnosis", vars = ["radius_mean", "concavity_mean", "smoothness_mean", "texture_mean"])
plt.show()

'''Spliting target variable and independent variable'''

X = data.drop(['diagnosis'], axis = 1)
y = data['diagnosis']

''' Splitting the data into training set and testset'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
print("Size of training set:", X_train.shape)
print("Size of test set:", X_test.shape)


'''           LOGISTIC REGRESSION     '''

''' Import library for LogisticRegression'''

from sklearn.linear_model import LogisticRegression


'''Create a Logistic regression classifier'''
logreg = LogisticRegression()


'''Train the model using the training sets '''
logreg.fit(X_train, y_train)


''' Prediction on test data'''
y_pred = logreg.predict(X_test)


'''Calculating the accuracy'''
acc_logreg = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Logistic Regression model : ', acc_logreg )

'''      GAUSSIAN NAIVE BAYES '''

'''Import library of Gaussian Naive Bayes model'''

from sklearn.naive_bayes import GaussianNB

'''Create a Gaussian Classifier'''

model = GaussianNB()

'''Train the model using the training sets '''

model.fit(X_train,y_train)

'''Prediction on test set'''

y_pred = model.predict(X_test)

'''Calculating the accuracy'''

acc_nb = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Gaussian Naive Bayes model : ', acc_nb )

'''     DECISION TREE CLASSIFIER  '''

'''Import Decision tree classifier'''

from sklearn.tree import DecisionTreeClassifier

''' Create a Decision tree classifier model'''

clf = DecisionTreeClassifier()

'''   HYPERPARAMETER OPTIMIZATION '''

'''Hyperparameter Optimization'''

parameters = {'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 50], 
              'min_samples_split': [2, 3, 50, 100],
              'min_samples_leaf': [1, 5, 8, 10]
             }

'''Run the grid search'''

grid_obj = GridSearchCV(clf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

''' Set the clf to the best combination of parameters'''

clf = grid_obj.best_estimator_

'''Train the model using the training sets'''

clf.fit(X_train, y_train)

''' Prediction on test set'''

y_pred = clf.predict(X_test)

''' Calculating the accuracy'''
acc_dt = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Decision Tree model : ', acc_dt )


''' RANDOM FOREST CLASSIFIER '''

'''Import library of RandomForestClassifier model'''

from sklearn.ensemble import RandomForestClassifier

'''Create a Random Forest Classifier'''

rf = RandomForestClassifier()

''' Hyperparameter Optimization'''

parameters = {'n_estimators': [4, 6, 9, 10, 15], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
             }

''' Run the grid search '''

grid_obj = GridSearchCV(rf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

'''Set the rf to the best combination of parameters'''

rf = grid_obj.best_estimator_

'''Train the model using the training sets'''

rf.fit(X_train,y_train)

''' Prediction on test data '''

y_pred = rf.predict(X_test)

'''Calculating the accuracy'''

acc_rf = round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 )
print( 'Accuracy of Random Forest model : ', acc_rf )



'''      SUPPORT VECTOR MODEL   '''

''' Creating scaled set to be used in model to improve the results'''

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

''' Import Library of Support Vector Machine model'''

from sklearn import svm

'''Create a Support Vector Classifier'''

svc = svm.SVC()

''' Hyperparameter Optimization  '''

parameters = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

''' Run the grid search '''

grid_obj = GridSearchCV(svc, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

''' Set the svc to the best combination of parameters'''

svc = grid_obj.best_estimator_

'''Train the model using the training sets '''

svc.fit(X_train,y_train)

''' Prediction on test data'''

y_pred = svc.predict(X_test)

''' Calculating the accuracy'''

acc_svm = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of SVM model : ', acc_svm )


''' K - NEAREST NEIGHBORS '''

'''Import library of KNeighborsClassifier model'''

from sklearn.neighbors import KNeighborsClassifier

'''Create a KNN Classifier'''

knn = KNeighborsClassifier()

''' Hyperparameter Optimization'''

parameters = {'n_neighbors': [3, 4, 5, 10], 
              'weights': ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'leaf_size' : [10, 20, 30, 50]
             }

'''Run the grid search'''

grid_obj = GridSearchCV(knn, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

'''Set the knn to the best combination of parameters'''

knn = grid_obj.best_estimator_

''' Train the model using the training sets '''

knn.fit(X_train,y_train)

'''Prediction on test data'''

y_pred = knn.predict(X_test)

'''Calculating the accuracy'''

acc_knn = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of KNN model : ', acc_knn )


''' Evaluation and comparison of all the models'''

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machines', 
              'K - Nearest Neighbors'],
    'Score': [acc_logreg, acc_nb, acc_dt, acc_rf, acc_svm, acc_knn]})
models.sort_values(by='Score', ascending=False)
print(models)





























