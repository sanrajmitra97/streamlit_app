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
x,y,z= data.App.unique()
st.markdown("We only have 1 predictor variable that's categorical, and that's `App`.")
if st.button("Show categories"):
    st.markdown(f"{x}, {y}, {z}")


# Continuous columns
cont_data = data[data.columns[2:]]
st.markdown("Here are the descriptor statistics for the continuous variables.")
if st.button("Show descriptions of continuous variables"):
    st.write(cont_data.describe())