import pickle
import pandas as pd
from utils import *
import os

# Get the current folder
current_folder = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory
os.chdir(current_folder)

# Load Model
loaded_model = pickle.load(open('../model/test_model.sav', 'rb'))

# Load Test Data
data = pd.read_csv("../data/numbers.csv")
data = data.sample(int(0.1*len(data)), axis=0) # Sample test set

# Clean Data
dc = Datacleaner(data)
dc.drop_na()
dc.unnecessary_cols(['date', 'device'])
data = dc.create_response('Country')


# Define X and y
X_test = data.drop(['Country'], axis=1)
y_test = data.Country.values

# Make Prediction
result = loaded_model.score(X_test, y_test)

# Print results
print(f"Score of model: {result}")
