import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve, auc
import plotly.express as px
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score, confusion_matrix,classification_report,accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier


# Load data
Heart_failure = pd.read_csv("heart.csv")
print(Heart_failure.head())
print(Heart_failure.to_string())
print(Heart_failure.info())
print(Heart_failure.isnull())




Heart_failure['HeartDisease'] = Heart_failure['HeartDisease'].astype(str)
columns_to_convert = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
Heart_failure_encoded = pd.get_dummies(Heart_failure, columns=columns_to_convert)

numeric_columns = Heart_failure_encoded.select_dtypes(include=['float64', 'int64']).columns
skewness = Heart_failure_encoded[numeric_columns].skew().sort_values(ascending=False)

# Create a DataFrame to visualize skewness
skewness_Heart_failure = pd.DataFrame({'Skewness': skewness})

# Visualize skewness using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(skewness_Heart_failure, annot=True, cmap='coolwarm', cbar=False)
plt.title('Skewness of Numerical Features')
plt.show()




# Define categorical features
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Split the data into input features (X) and target variable (y)
X = Heart_failure.drop("HeartDisease", axis=1)
y = Heart_failure["HeartDisease"]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the train_set DataFrame by merging x_train and y_train
train_set = x_train.copy()
train_set['HeartDisease'] = y_train

# Convert the 'HeartDisease' column to string in the training set
train_set['HeartDisease'] = train_set['HeartDisease'].astype(str)

# Convert categorical columns to string in the training set
for feature in categorical_features:
    train_set[feature] = train_set[feature].astype(str)

    # Visualize the distribution of categorical features by HeartDisease status
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_set, x=feature, hue='HeartDisease', palette='viridis')
    plt.title(f'{feature} Distribution by Heart Disease Status')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# Define a list of continuous features
continuous_features = [col for col in X.columns if col not in categorical_features]

# Visualize the distribution of continuous features by HeartDisease status
for feature in continuous_features:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=train_set, x=feature, hue='HeartDisease', fill=True, palette='viridis')
    plt.title(f'{feature} Distribution by Heart Disease Status')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()



for feature in continuous_features:
   plt.figure(figsize=(10, 6))
   sns.boxplot(x='HeartDisease', y=feature, data=train_set, hue='HeartDisease', palette='viridis')
   plt.title(f'{feature} Distribution by Heart Disease Status')
   plt.xlabel('Heart Disease Status')
   plt.ylabel(feature)
   plt.show()

''' Cholesterol Distribution
This dataset exhibits a right-skewed distribution for cholesterol.
The median cholesterol level is around 225 mg/dL, indicating that a higher proportion of people have cholesterol levels below the median.
There are a few outliers with cholesterol levels above 500 mg/dL, indicating a small number of individuals with very high cholesterol levels in this dataset.

Cholesterol and Heart Disease
people without heart disease have a median cholesterol level around 240.
people with heart disease have a median cholesterol level around 230.
Most people, regardless of heart disease status, have cholesterol levels between 200 and 300.
people with heart disease exhibit a wider range of cholesterol levels compared to those without heart disease.
extremely low cholesterol levels are more frequent in those with heart disease, which could warrant further investigation.

Age and Heart Disease¶
Age is a significant factor in the prevalence of heart disease, with older individuals being more likely to have heart disease.
For those with heart disease, the median age is higher compared to those without.
While heart disease is more common in older age groups, it also affects a notable number of younger individuals.

Resting Blood Pressure Insights
RestingBP is right-skewed.
Most individuals have a RestingBP between 120 and 140.
More outliers on the higher end (resting blood pressure > 175) than on the lower end (resting blood pressure < 50).
The median RestingBP is around 130, meaning half of the individuals have a resting blood pressure below this value.'''

# Define categorical features
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Split the data into input features (X) and target variable (y)
X = Heart_failure.drop("HeartDisease", axis=1)
y = Heart_failure["HeartDisease"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the categorical features with handle_unknown='ignore'
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

# StandardScaler for numerical features
scaler = StandardScaler()

# Define numerical features
numerical_features = [col for col in X.columns if col not in categorical_features]

# Column transformer to apply one-hot encoding to the categorical features and scaling to numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, categorical_features),
        ('num', scaler, numerical_features)
    ]
)

# Logistic regression model with increased max_iter
logreg = LogisticRegression(max_iter=2000)

# Create a pipeline that first transforms the data then fits the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', logreg)])

# Fit the model
pipeline.fit(X_train, y_train)

# Check the score of the model
score = pipeline.score(X_test, y_test)
print(f'Model accuracy: {score:.2f}')



numeric_features =['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
# Define the numeric pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier_remover', RobustScaler()),
    ('skewness_corrector', PowerTransformer(method='yeo-johnson', standardize=True))
])

# Define the categorical pipeline
categorical_transformer = OneHotEncoder(handle_unknown='ignore')


preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GradientBoostingClassifier()
voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)],
    voting='hard')
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])
pipeline.fit(X_train, y_train)
# Transform the test data
y_pred = pipeline.predict(X_test)
# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred))


# Encode categorical variables
le = LabelEncoder()
df1 = Heart_failure.copy(deep=True)
df1['Sex'] = le.fit_transform(df1['Sex'])
df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])

# Split data into features (X) and target (y)
features = df1[df1.columns.drop(['HeartDisease', 'RestingBP', 'RestingECG'])].values
# Assuming 'HeartDisease' is the target column
X = df1.drop('HeartDisease', axis=1)  
y = df1['HeartDisease'].values
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Define model function
def model(classifier):
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    
    

# Define model evaluation function
def model_evaluation(classifier):
    # Confusion Matrix
    cm = confusion_matrix(y_test, classifier.predict(x_test))
    print("Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, classifier.predict(x_test)))

    

#1] Logistic Regression 
classifier_lr = LogisticRegression(random_state=0, C=10, penalty='l2', solver='liblinear')
model(classifier_lr)
model_evaluation(classifier_lr)


#2] Support Vector Classifier :
from sklearn.svm import SVC
classifier_svc = SVC(kernel = 'linear',C = 0.1)
model(classifier_svc)
model_evaluation(classifier_svc)

#3] Decision Tree Classifier 
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)
model(classifier_dt)
model_evaluation(classifier_dt)

#4] Random Forest Classifier :
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)
model(classifier_rf)
model_evaluation(classifier_rf)

'''5] K-nearest Neighbors Classifier :
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(leaf_size = 1, n_neighbors = 3,p = 1)
model(classifier_knn)
model_evaluation(classifier_knn)'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


# Split the data into input features (X) and target variable (y)
X = Heart_failure.drop("HeartDisease", axis=1)
y = Heart_failure["HeartDisease"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('skewness_corrector', PowerTransformer(method='yeo-johnson', standardize=True))
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ]
)

# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier()

# Create a pipeline that first transforms the data then fits the model
pipeline_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', knn)
])

# Define hyperparameter grid for KNN
param_grid = {
    'classifier__leaf_size': [1, 2, 3, 4, 5],
    'classifier__n_neighbors': [1, 3, 5, 7, 9],
    'classifier__p': [1, 2]
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline_knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict and evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best parameters: {best_params}')
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))



