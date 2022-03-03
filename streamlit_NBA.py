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

original_xGStats_df = pd.read_csv('df_xGStats.csv')
original_AdvStats_df = pd.read_csv('df_AdvStats.csv')

Adv_Stats_1_df = pd.read_csv('Adv_Stats_1.csv')
xG_Stats_1_df = pd.read_csv('xG_Stats_1.csv')
LeBron_Injuries_df = pd.read_csv('LeBron_Injuries.csv')

LBJ_Stats_df = pd.read_html('https://www.basketball-reference.com/players/j/jamesle01.html')
LBJ_xGStats_df = pd.DataFrame(LBJ_Stats_df[0])
LBJ_TotStats_df = pd.DataFrame(LBJ_Stats_df[2])

st.title('NBA Project: Analysis and Prediction about LeBron James Career and Season 2020/2021')

sec = st.sidebar.radio('Sections:', ['Data cleaning', 'LeBron James exploration and analysis', 'Predictive model for LeBron James', 'Season 2020/2021 exploration and analysis', 'Predictive model for Season 2020/2021'])

if sec == 'Data cleaning': 
    st.header('Data cleaning')

    st.write("For this project I choose to develop two different topics:")
    st.markdown('''
            - LeBron James Analysis: an analysis where I studied his career trends and predicted some record he could break until he will retire, like the current 'All-Time Point Leader';
            - Season 2020/2021 Analysis: an analysis where I studied stats about the season and provided interesting models to reach specific results.

            I used two different datasets for each topic. For the LeBron James analysis, I evaluated:
            - LeBron James per Game stats;
            - LeBron James totals stats.
            For the Season 2020/2021 analysis, I evaluated:
            - Season 2020/2021 per Game stats;
            - Season 2020/2021 advanced stats. 
            ''')

    #aggiungere with con i due dataset di LeBron (xG e Tot)
    with st.expander('The LeBron James Career dataset based on Per Game stats for season'):
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/players/j/jamesle01.html. It is the first table in the web page and it is an html file because LeBron is still playing, then his stats are increasing and changing day by day.')
            st.download_button('Download CSV', LBJ_xGStats_df.to_csv(index=False))
            st.write('It contains all the stats during the whole career of LeBron James taken with the per game stats per season and it is still in update because he is still playing.')

    with st.expander('The LeBron James Career dataset based on Totals stats for season'):
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/players/j/jamesle01.html. It is the third table in the web page and it is an html file because LeBron is still playing, then his stats are increasing and changing day by day.')
            st.download_button('Download CSV', LBJ_TotStats_df.to_csv(index=False))
            st.write('It contains all the stats during the whole career of LeBron James taken with the totals per season and it is still in update because he is still playing.')

    with st.expander('The Per Game Stats dataset'):
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/leagues/NBA_2021_per_game.html. It is the first table in the web page and before turning the html dataframe into csv file I had to erase and clean some stats and some inaccuracies. You can download the raw data here.')
            st.download_button('Download CSV', original_xGStats_df.to_csv(index=False))
            st.write('It contains all the usual stats provided to analyse the whole season 2020/2021, like points, rebounds, assists and so on.')

            #spiegare variabili che ci sono nel dataset

    with st.expander('The Advanced Stats dataset'):
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/leagues/NBA_2021_advanced.html. It is the first table in the web page and before turning the html dataframe into csv file I had to erase and clean some stats and some inaccuracies. You can download the raw data here.')
            st.download_button('Download CSV', original_AdvStats_df.to_csv(index=False))
            st.write('It contains specific stats based on advanced analysis about the whole season 2020/2021, and unlike the previous stats these are established only when the season is finished. Some of these stats are win share, box plus minus and others.')

            #spiegare variabili che ci sono nel dataset

    

if sec == 'LeBron James exploration and analysis':
    st.header('LeBron James exploration and analysis')

if sec == 'Predictive model for LeBron James':
    st.header('Predictive model for LeBron James')

if sec == 'Season 2020/2021 exploration and analysis':
    st.header('Season 2020/2021 exploration and analysis')

if sec == 'Predictive model for Season 2020/2021':
    st.header('Predictive model for Season 2020/2021')