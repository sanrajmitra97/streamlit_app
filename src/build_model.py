import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import pickle
from utils import *
import os

# Get the current folder
current_folder = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory
os.chdir(current_folder)

# Assume we have a dataset X with numerical and categorical columns and target y
# First we clean the original data before train test split. 
data = pd.read_csv("../data/numbers.csv")
dc = Datacleaner(data)
dc.drop_na()
dc.unnecessary_cols(['date', 'device'])
data = dc.create_response('Country')

# Define X and y
X = data.drop(['Country'], axis=1)
y = data.Country.values

# Name numerical and categorical features
numerical_features = ['downloads', 'active_users', 'total_sessions', 'total_minutes', 'usage_penetration', 'open_rate', 'mb_per_user', 'mb_per_session']
categorical_features = ['App']

# define the transformers for numerical and categorical features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

# combine the transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# combine the preprocessor with the classifier using Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('custom_transformer', FunctionTransformer(custom_transformer)), ('classifier', SVC())])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search CV
parameteres = {'classifier__C':[0.001,0.1,10,100,10e5], 'classifier__gamma':[0.1,0.01]}
grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5)

# fit the pipeline on the training data
grid.fit(X_train, y_train)

# make predictions on the testing data
y_pred = grid.predict(X_test)

# evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

# Build full model if accuracy above 0.9
if accuracy > 0.90:
    grid.fit(X, y)

# Save model
filename = '../model/test_model.sav'
pickle.dump(grid, open(filename, 'wb'))

