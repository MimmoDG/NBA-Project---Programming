import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skm
import time
import streamlit.components.v1 as components

#import the initials datasets

original_xGStats_df = pd.read_csv('df_xGStats.csv')
original_AdvStats_df = pd.read_csv('df_AdvStats.csv')

Adv_Stats_1_df = pd.read_csv('Adv_Stats_1.csv')
xG_Stats_1_df = pd.read_csv('xG_Stats_1.csv')
LeBron_Injuries_df = pd.read_csv('LeBron_Injuries.csv')

LBJ_Stats_df = pd.read_html('https://www.basketball-reference.com/players/j/jamesle01.html')
LBJ_xGStats_df = pd.DataFrame(LBJ_Stats_df[0])
LBJ_TotStats_df = pd.DataFrame(LBJ_Stats_df[2])

#import the finals datasets

st.title('NBA Project: Analysis and Prediction about LeBron James Career and Season 2020/2021')

sec = st.sidebar.radio('Sections:', ['Data cleaning', 'LeBron James exploration and analysis', 'Predictive model for LeBron James', 'Season 2020/2021 exploration and analysis', 'Predictive model for Season 2020/2021'])

if sec == 'Data cleaning': 
    st.header('Data cleaning')

    st.write("For this project I choose to develop two different topics:")
    st.markdown('''
            - LeBron James Analysis: an analysis where I studied his career trends and predicted some record he could break until he will retire, like the current 'All-Time Point Leader';
            - Season 2020/2021 Analysis: an analysis where I studied stats about the season and provided interesting models to reach specific results.

            In this way, I conducted different surveys about the same subject analysing two specific topics. In the same time, I examined the two topics focusing in particular datasets.
            To carry out these analysis, I therefore used two different datasets for each topic. 
            For the LeBron James analysis, I evaluated:
            - LeBron James per Game stats;
            - LeBron James totals stats.
            For the Season 2020/2021 analysis, I evaluated:
            - Season 2020/2021 per Game stats;
            - Season 2020/2021 advanced stats. 
            ''')

    with st.expander('The LeBron James Career dataset based on Per Game stats for season'):
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/players/j/jamesle01.html. It is the first table in the web page and it is an html file because LeBron is still playing, then his stats are increasing and changing day by day.')
            st.download_button('Download CSV', LBJ_xGStats_df.to_csv(index=False))
            st.write('It contains all the stats during the whole career of LeBron James taken with the per game stats per season and it is still in update because he is still playing.')

    with st.expander('The LeBron James Career dataset based on Totals stats for season'):
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/players/j/jamesle01.html. It is the third table in the web page and it is an html file because LeBron is still playing, then his stats are increasing and changing day by day.')
            st.download_button('Download CSV', LBJ_TotStats_df.to_csv(index=False))
            st.write('It contains all the stats during the whole career of LeBron James taken with the totals per season and it is still in update because he is still playing.')

    #explanation of variables in the datasets
    st.write('''
            In these datasets, there are the same stats explaining the LeBron James's style of play and these are also the usual stats used to analise different players.
            First of all, the two raws datasets contain 24 rows and 30 coloumns each.
            The rows represent the different seasons he has played through his career and the last 4 shows his career trends and his trends concerning the three different teams he has played for.
            The columns are:
            - Season: every season he has played;
            - Age: player's age on February 1 of every season;
            - Tm: the different teams he has played for;
            - Lg: the league where he played;
            - Pos: which position he played during the different seasons;
            - G, GS: these two variables show how many games he has played and he played starting the match;
            - MP: minutes played per game and totals for every season;
            - FG, FGA, FG%: these variables represent the field goals realized, attempted and the percentage;
            - 3P, 3PA, 3P%: these variables represent the 3-point field goals realized, attempted and the percentage;
            - 2P, 2PA, 2P%: these variables represent the 2-point field goals realized, attempted and the percentage;
            - FT, FTA, FT%: these variables represent the free throws realized, attempted and the percentage;
            - eFG%: This statistic adjusts for the fact that a 3-point field goal is worth one more point than a 2-point field goal;
            - ORB, DRB, TRB: these variables represent the offensive, defensive and total rebounds made;
            - AST: assists provided to his teammates;
            - STL: balls stolen to his opponents;
            - BLK: shots blocked to his opponents;
            - TOV: balls lost during a possession;
            - PF: personal fouls made;
            - PTS: sum of points realized with every basket.

            All these stats are analysed according to the Per Game and the Totals, so in the first case the playing statistics are normalized per Game. In the second case those statistics are just sumed up in order to find the totals in every season.
            ''')

    with st.expander('The Per Game Stats dataset'):
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/leagues/NBA_2021_per_game.html. It is the first table in the web page and before turning the html dataframe into csv file I had to erase and clean some stats and some inaccuracies. You can download the raw data here.')
            st.download_button('Download CSV', original_xGStats_df.to_csv(index=False))
            st.write('It contains all the usual stats provided to analyse the whole season 2020/2021, like points, rebounds, assists and so on.')

            #explanation of variables in the dataset
            st.write('''
            In this dataset are represented the same statistics as the LeBron James Per Game Dataset, so the variables won't be presented avoiding a repetition.
            The only difference between the two datasets is that in the first one, only one player is analysed, instead in the second case there are initially 731 rows instead of 24.
            In these rows are represented all the players in the NBA in the season 2020/2021 but also some repetition of the first row, which represents the name of the different variables.
            ''')

    with st.expander('The Advanced Stats dataset'):
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/leagues/NBA_2021_advanced.html. It is the first table in the web page and before turning the html dataframe into csv file I had to erase and clean some stats and some inaccuracies. You can download the raw data here.')
            st.download_button('Download CSV', original_AdvStats_df.to_csv(index=False))
            st.write('It contains specific stats based on advanced analysis about the whole season 2020/2021, and unlike the previous stats these are established only when the season is finished. Some of these stats are win share, box plus minus and others.')
            #spiegare variabili che ci sono nel dataset
    
    #aggiungere dataset finali o fare le modifiche al momento e spiegarle
    #scrivere questa parte con markdown
    #vanno aggiunti anche altri expander coi dataset finali
    #dopo aver inserito i dataset finali e il download button con i csv è finita la parte di data cleaning 
    #capire bene come fare le modifiche al momento e renderle visibili su streamlit


if sec == 'LeBron James exploration and analysis':
    st.header('LeBron James exploration and analysis')

    x = st.selectbox('Choose a Stat', LBJ_xGStats_df.columns.tolist())
    ms = LBJ_xGStats_df[x].max()
    a = LBJ_xGStats_df[LBJ_xGStats_df[x]==ms].Season
    st.write('The max has been registered in the: ', a, 'The max for this stat is: ', ms)

    Season = list(LBJ_xGStats_df['Season'])
    Peppino = list(LBJ_xGStats_df[x])
    fig = plt.figure(figsize=(10, 6))
    plt.plot(Season, Peppino, '-o')
    plt.title('LeBron ' + x + ' averages in every season')
    plt.xlabel('Seasons')
    plt.ylabel(x)
    plt.xticks(rotation=45)
    st.pyplot(fig)

if sec == 'Predictive model for LeBron James':
    st.header('Predictive model for LeBron James')

if sec == 'Season 2020/2021 exploration and analysis':
    st.header('Season 2020/2021 exploration and analysis')

if sec == 'Predictive model for Season 2020/2021':
    st.header('Predictive model for Season 2020/2021')