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
st.caption("This application focuses on building the streamlit app so I can learn how to deploy future ML models.\n The machine learning done here is not the main focus, and hence\n the code used to build the model should not be taken seriously. This is my firt time building a streamlit app so it may look a bit lackluster \:P")

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

# Dataframe
st.markdown("To learn more about our dataset, visit the `dataframe page`")

# Widgets
age = st.slider('Choose your age', min_value=1, max_value=100, key='age')  # 👈 this is a widget
st.write(f"{age + 10} is your real age.")

# Choose the categorical variable
categories = data.App.unique()
option = st.selectbox('Choose application type', categories)
st.write(f'You have selected: {option}')






