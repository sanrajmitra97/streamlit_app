import streamlit as st
import pandas as pd
import pickle

# Load Model
loaded_model = pickle.load(open('../model/test_model.sav', 'rb'))

# General Info
st.header("Streamlit ML Prediction App")
st.text_input("Enter Your Name: ", key="name")

# Data
data = pd.read_csv("../data/numbers.csv")

