import streamlit as st
import pandas as pd
import os

st.header("DataFrame Page")
st.sidebar.markdown("Dataframe Page")

# Read in the data
data = pd.read_csv("../data/numbers.csv")
data.dropna(how='any', axis=0, inplace=True)
data.drop(['date', 'device'], axis=1, inplace=True)

# Showcase Data
st.markdown("Click below to view 5 random rows of the data!")
st.checkbox('Show dataframe', key='show_df')
if st.session_state.show_df:
    st.dataframe(data.sample(5))

# Shape of the data 
r, c = data.shape
dim = st.checkbox('Show dimensions of the data')
if dim:
    st.write(f"Dimensions of the data: {r} rows and {c} columns.")

# Categorical columns
st.markdown("We only have 1 predictor variable that's categorical, and that's `App`. The 3 values are: ")
x,y,z= data.App.unique()


