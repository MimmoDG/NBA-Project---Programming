import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skm
import time

#import the datasets

Adv_Stats_1_df = pd.read_csv('Adv_Stats_1.csv')
xG_Stats_1_df = pd.read_csv('xG_Stats_1.csv')
LeBron_Injuries_df = pd.read_csv('LeBron_Injuries.csv')

st.title('NBA Project: Streamlit page for the Programming Project')

sec = st.sidebar.radio('Sections:', ['Data cleaning', 'LeBron James exploration and analysis', 'Predictive model for LeBron James', 'Season 2020/2021 exploration and analysis', 'Predictive model for season 2020/2021'])

if sec == 'Data cleaning': 

