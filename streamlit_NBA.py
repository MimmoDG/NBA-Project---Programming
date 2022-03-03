import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skm
import time
import streamlit.components.v1 as components

#import the datasets

Adv_Stats_1_df = pd.read_csv('Adv_Stats_1.csv')
xG_Stats_1_df = pd.read_csv('xG_Stats_1.csv')
LeBron_Injuries_df = pd.read_csv('LeBron_Injuries.csv')

HtmlFile = open("https://www.basketball-reference.com/players/j/jamesle01.html", 'r', encoding='utf-8')
LBJ_xG_Stats = HtmlFile.read() 
print(LBJ_xG_Stats)
components.html(LBJ_xG_Stats)

st.title('NBA Project: Streamlit page for the Programming Project')

sec = st.sidebar.radio('Sections:', ['Data cleaning', 'LeBron James exploration and analysis', 'Predictive model for LeBron James', 'Season 2020/2021 exploration and analysis', 'Predictive model for Season 2020/2021'])

if sec == 'Data cleaning': 
     st.header('Data cleaning')

     st.write("For this project I choose to develop two different topics:")
     st.markdown('''
            - LeBron James Analysis: an analysis where I studied his career trends and predicted some record he could break until he will retire, like the current 'All-Time Point Leader';
            - Season 2020/2021 Analysis: an analysis where I studied stats about the season and provided interesting models to reach specific results.

            I used two different datasets for each topic 
            
            ''')

if sec == 'LeBron James exploration and analysis':
    st.header('LeBron James exploration and analysis')

if sec == 'Predictive model for LeBron James':
    st.header('Predictive model for LeBron James')

if sec == 'Season 2020/2021 exploration and analysis':
    st.header('Season 2020/2021 exploration and analysis')

if sec == 'Predictive model for Season 2020/2021':
    st.header('Predictive model for Season 2020/2021')