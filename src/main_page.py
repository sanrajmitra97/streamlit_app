import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# Title 
st.title("Welcome to my Streamlit demo app!")

# Header
st.header("Prediction Page")

# Brief explanation of app
st.caption("This application focuses on building the streamlit app so I can learn how to deploy future ML models to users. The machine learning and design of the app is not the main focus, and hence the code used to build the model and application should not be taken seriously \:3. This is my first time building a streamlit app so it may look a bit lackluster \:(")


# Prediction Description
st.subheader("Prediction Description")
st.caption("In this task, decide the values of input features for mobile data and the machine learning model will classify which country these features correspond to, A or B. Let's get started!")

# The page we are at
st.sidebar.markdown("Prediction Page")

# Get the current folder
current_folder = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory
os.chdir(current_folder)

# Load Model
loaded_model = pickle.load(open('../model/test_model.sav', 'rb'))

# General Info
name = st.text_input("Enter Your Name: ", key="name")
if name:
    st.write(f"Hello {st.session_state.name}, I'm Sanraj!")

# Data
data = pd.read_csv("../data/numbers.csv")
data.drop(['date', 'device'], axis=1, inplace=True)
data['Country'] = data['Country'].apply(lambda x: 1 if x=='A' else 0)


# Dataframe
st.markdown("To learn more about our dataset and its features, visit the `dataframe page` on the left.")

# Choose the categorical variable
categories = data.App.unique()
App = st.radio('Choose application type', categories)
st.write(f'You have selected: {App}')

# Choose variables
numerical_columns = data.columns[2:]
res = []
for col in numerical_columns:
    val = st.number_input(f"Choose number of {col} from {data[col].min()} and {data[col].max()}", min_value = data[col].min(), max_value=data[col].max())
    st.write(f'You have selected: {val} for the {col}')
    res.append(val)

# Make prediction
test_row = np.append(App, res).reshape(1,9)
test = pd.DataFrame(test_row, columns=data.columns[1:])
prediction = loaded_model.predict(test)

if prediction:
    country = 'A'
else:
    country = 'B'

if st.button("Make Prediction!"):
    st.write(f"Model Predict's country: {country}")













