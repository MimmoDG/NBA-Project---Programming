import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as skm
import time
import streamlit.components.v1 as components
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn import cluster
from sklearn.cluster import KMeans
import streamlit_option_menu as som
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


#import the initials datasets

original_xGStats_df = pd.read_csv('df_xGStats.csv')
original_AdvStats_df = pd.read_csv('df_AdvStats.csv')

LBJ_Stats_df = pd.read_html('https://www.basketball-reference.com/players/j/jamesle01.html')
LBJ_xGStats_df = pd.DataFrame(LBJ_Stats_df[0])
LBJ_TotStats_df = pd.DataFrame(LBJ_Stats_df[2])

#copie dei datasets originali
or_xG_Stats_df = original_xGStats_df #first dataset season 2020/21
or_Adv_Stats_df = original_AdvStats_df #second dataset season 2020/21
copy_LBJ_xGStats_df = LBJ_xGStats_df #first dataset LeBron
copy_LBJ_TotStats_df = LBJ_TotStats_df #second dataset LeBron

#processi di cleaning dei datasets:

# --> first dataset LeBron

for el in copy_LBJ_xGStats_df:
  if(el != "Season" and el != "Pos"  and el != "Tm" and el != "Lg"):
    copy_LBJ_xGStats_df[el] = pd.to_numeric(copy_LBJ_xGStats_df[el])

LeB_C_PG_RS1 = copy_LBJ_xGStats_df.drop(labels=range(19, 24), axis=0)
LeB_C_PG_RS = LeB_C_PG_RS1.drop(labels=['Lg', 'Age', 'GS'], axis=1)

# --> second dataset LeBron

for el in copy_LBJ_TotStats_df:
  if(el != "Season" and el != "Pos"  and el != "Tm" and el != "Lg"):
    copy_LBJ_TotStats_df[el] = pd.to_numeric(copy_LBJ_TotStats_df[el])

LeB_C_Tot_RS1 = copy_LBJ_TotStats_df.drop(labels=range(19, 24), axis=0)
LeB_C_Tot_RS = LeB_C_Tot_RS1.drop(labels=['Lg', 'Age', 'GS', 'Unnamed: 30', 'Trp Dbl'], axis=1)

# --> first dataset season 2020/21

xG_Stats = or_xG_Stats_df.where(or_xG_Stats_df["Player"] != "Player").dropna(how='all', axis=0)
xG_Stats_1 = xG_Stats.where(xG_Stats["Tm"] != "TOT").dropna(how='all', axis=0)
xG_Stats_1.drop('Rk', axis=1, inplace=True)
for el in xG_Stats_1:
  if(el != "Player" and el != "Pos"  and el != "Tm"):
    xG_Stats_1[el] = pd.to_numeric(xG_Stats_1[el])
xG_Stats_1['FG%'].fillna(0, inplace=True)
xG_Stats_1['3P%'].fillna(0, inplace=True)
xG_Stats_1['2P%'].fillna(0, inplace=True)
xG_Stats_1['eFG%'].fillna(0, inplace=True)
xG_Stats_1['FT%'].fillna(0, inplace=True)

# --> second dataset season 2020/21

Adv_Stats = or_Adv_Stats_df.where(or_Adv_Stats_df["Player"] != "Player").dropna(how='all',axis=0)
Adv_Stats.drop(['Rk', 'Unnamed: 19', 'Unnamed: 24'], axis=1, inplace=True)
Adv_Stats_1 = Adv_Stats.where(xG_Stats["Tm"] != "TOT").dropna(how='all', axis=0)
for el in Adv_Stats_1:
  if(el != "Player" and el != "Pos"  and el != "Tm"):
    Adv_Stats_1[el] = pd.to_numeric(Adv_Stats_1[el])
Adv_Stats_1['TOV%'].fillna(0, inplace=True)
Adv_Stats_1['TS%'].fillna(0, inplace=True)
Adv_Stats_1['3PAr'].fillna(0, inplace=True)
Adv_Stats_1['FTr'].fillna(0, inplace=True)


#import the finals datasets

final_LBJ_xG_df = pd.read_csv('df_LBJ_xG_Stats_final.csv')
final_LBJ_Tot_df = pd.read_csv('df_LBJ_Tot_Stats_final.csv')

Adv_Stats_1_df = pd.read_csv('Adv_Stats_1.csv')
xG_Stats_1_df = pd.read_csv('xG_Stats_1.csv')
LeBron_Injuries_df = pd.read_csv('LeBron_Injuries.csv')

st.title('NBA Project: Analysis and Prediction about LeBron James Career and Season 2020/2021')

with st.sidebar:
  sec = som.option_menu('Sections:', ['Data cleaning', 'LeBron James exploration and analysis', 'Predictive model for LeBron James', 'Season 2020/2021 exploration and analysis', 'Predictive model for Season 2020/2021'], menu_icon="cast", default_index=0)

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
            st.header('The LeBron James Career dataset based on Per Game stats for season')
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/players/j/jamesle01.html. It is the first table in the web page and it is an html file because LeBron is still playing, then his stats are increasing and changing day by day.')
            st.download_button('Download CSV', LBJ_xGStats_df.to_csv(index=False))
            st.write('It contains all the stats during the whole career of LeBron James taken with the per game stats per season and it is still in update because he is still playing.')

    with st.expander('The LeBron James Career dataset based on Totals stats for season'):
            st.header('The LeBron James Career dataset based on Totals stats for season')
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
            There is only one difference between the two datasets which is represented by the triple double variable.
            ''')

    with st.expander('The Per Game Stats dataset'):
            st.header('The Per Game Stats dataset')
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
            st.header('The Advanced Stats dataset')
            st.write('The dataset is available on Basketball Reference at https://www.basketball-reference.com/leagues/NBA_2021_advanced.html. It is the first table in the web page and before turning the html dataframe into csv file I had to erase and clean some stats and some inaccuracies. You can download the raw data here.')
            st.download_button('Download CSV', original_AdvStats_df.to_csv(index=False))
            st.write('It contains specific stats based on advanced analysis about the whole season 2020/2021, and unlike the previous stats these are established only when the season is finished. Some of these stats are win share, box plus minus and others.')
            
            #spiegare variabili che ci sono nel dataset
            st.write('''
            In this dataset there are unusual variables to explain different features about the athletes who play in the NBA.
            All these statistics are provided at the end of the season, because they represent an 'a Posteriori' analysis.
            The dataset is composed by rows and columns. In the rows are listed the whole number of players in the season 2020/21, more precisely there are 540 players but some of them has changed team during the season so they are listed two or even three times for each team they played for.
            In the columns are listed the different stats used in the analysis and those are:
            - Pos: Position;
            - Age: Player's age on February 1 of the season;
            - Tm: Team;
            - G: Games played;
            - MP: Minutes played;
            - PER (Player Efficiency Rating): A measure of per-minute production standardized such that the league average is 15;
            - TS% (True Shooting Percentage): A measure of shooting efficiency that takes into account 2-point field goals, 3-point field goals, and free throws;
            - 3PAr (3-Point Attempt Rate): Percentage of FG Attempts from 3-Point Range;
            - FTr (Free Throw Attempt Rate): Number of FT Attempts Per FG Attempt;
            - ORB% (Offensive Rebound Percentage): An estimate of the percentage of available offensive rebounds a player grabbed while they were on the floor;
            - DRB% (Defensive Rebound Percentage): An estimate of the percentage of available defensive rebounds a player grabbed while they were on the floor;
            - TRB% (Total Rebound Percentage): An estimate of the percentage of available rebounds a player grabbed while they were on the floor;
            - AST% (Assist Percentage): An estimate of the percentage of teammate field goals a player assisted while they were on the floor;
            - STL% (Steal Percentage): An estimate of the percentage of opponent possessions that end with a steal by the player while they were on the floor;
            - BLK% (Block Percentage): An estimate of the percentage of opponent two-point field goal attempts blocked by the player while they were on the floor;
            - TOV% (Turnover Percentage): An estimate of turnovers committed per 100 plays;
            - USG% (Usage Percentage): An estimate of the percentage of team plays used by a player while they were on the floor;
            - OWS (Offensive Win Shares): An estimate of the number of wins contributed by a player due to offense;
            - DWS (Defensive Win Shares): An estimate of the number of wins contributed by a player due to defense;
            - WS (Win Shares): An estimate of the number of wins contributed by a player;
            - WS/48 (Win Shares Per 48 Minutes): An estimate of the number of wins contributed by a player per 48 minutes (league average is approximately .100);
            - OBPM (Offensive Box Plus/Minus): A box score estimate of the offensive points per 100 possessions a player contributed above a league-average player, translated to an average team;
            - DBPM (Defensive Box Plus/Minus): A box score estimate of the defensive points per 100 possessions a player contributed above a league-average player, translated to an average team;
            - BPM (Box Plus/Minus): A box score estimate of the points per 100 possessions a player contributed above a league-average player, translated to an average team;
            - VORP (Value over Replacement Player): A box score estimate of the points per 100 TEAM possessions that a player contributed above a replacement-level (-2.0) player, translated to an average team and prorated to an 82-game season.
            ''')
    
    st.markdown('''
    After providing the initial dataset of every analysis expired, now will be explained how I cleaned and set up the final dataset for each initial one.
    First of all, for the LeBron James datasets it was easier to clean up because the two datasets are smaller than the others.
    In fact, for the LeBron James datasets I had to delete the last four rows initially provided because they contained different summaries about his career: there were one for the whole career and three representing the three teams he played and still plays for.
    After that I deleted some variables not very useful for the analysis, like the league where he plays, the games started because are equal to the game he played, the age because it's similar to the season statistic.
    This cleaning process has been applied for both the datasets, in order to make equal the two datasets encouraging a comparison between them.
    ''')

    # final datasets with download button
    # brief explenation of the steps done

    with st.expander('Final LeBron James Datasets'):
          st.header('Final LeBron James Datasets')
          st.write('The two final datasets for the LeBron James Analysis are here attached.')

          st.download_button('Download CSV', final_LBJ_xG_df.to_csv(index=False))
          st.write(''' For the 'Per Game' stats are been made few changes: 
          The first one change all the stats that are tipically numeric, but initially stored as object, from object to numeric datatype. 
          Secondly, are deleted the summary rows containing the whole career and the different teams he played for.
          Lastly, are deleted three variables that are not significant for the analysis or redundant: Age, GS and Lg.
          ''')
          st.write(''' Here is attached a description of the variables with the main percentiles, the mean and the count.
          With this description is easier to analize the data for the analysis.
          ''')
          st.dataframe(LeB_C_PG_RS.describe()) 

          st.download_button('Download CSV', final_LBJ_Tot_df.to_csv(index=False))
          st.write(''' As for the first dataset, also for the 'Totals' stats haven't been made a lot of changes:
          As for the previous dataset the first step is about the datatype of the different stats, changing it from object to numeric.
          Also the second step is equal, with the deletion of the summary rows.
          The third and last step is different from the 'Per Game' dataset, in fact in it are deleted some variables not so significant for the analysis and those are: Age, GS, Lg, Unnamed: 30 and Trp Dbl.
          ''')
          st.write(''' Here is attached a description of the variables with the main percentiles, the mean and the count.
          With this description is easier to analize the data for the analysis.
          ''')
          st.dataframe(LeB_C_Tot_RS.describe()) 

    with st.expander('Final Season 2020/2021 Datasets'):
          st.header('Final Season 2020/2021 Datasets')
          st.write('The two final datasets for the Season 2020/2021 Analysis are here attached.')

          st.download_button('Download CSV', xG_Stats_1_df.to_csv(index=False))
          st.write(''' For the 'Per Game' stats have been made different changes:
          Firstly, the rows containing the variables names repeated every twenty players have been dropped.
          After that the rows where the player's team was 'TOT' have been dropped, because they present the sum of the statistics for player who played for different teams in the same season, and this fact is not useful for this analysis.
          Also the column 'Rk' has been dropped for the same reason of the 'TOT'.
          Nextly, a 'for' cycle has been used to set every variable in the dataset, with the exeption of 'Player', 'Tm' and 'Pos', from object to numeric value.
          Lastly, all the null value in the percentages have been filled with zero in order to allow comparison also between players who don't shoot from the 3-point line, for example.
          So the final dataset is prepared and it has been changed accordingly to what feature was relevant for every analysis done on it.
          ''')
          st.write(''' Here is attached a description of the variables with the main percentiles, the mean and the count.
          With this description is easier to analize the data for the analysis.
          ''')
          st.dataframe(xG_Stats_1.describe()) 

          st.download_button('Download CSV', Adv_Stats_1_df.to_csv(index=False))
          st.write(''' For the 'Advanced' stats have been made similar changes to the 'Per Game' dataset.
          Firstly, as in the previous dataset, have been dropped all the rows containing the name of the variables repeated every twenty players.
          Then, also in this case the player with the 'TOT' as team and the variable 'Rk' have been dropped. In addition to this, also other two columns have been dropped but they were empty columns.
          Nextly, a 'for' cycle has been used to set every variable in the dataset, with the exeption of 'Player', 'Tm' and 'Pos', from object to numeric value.
          Lastly, all the null value in the percentages have been filled with zero in order to allow comparison also between players who don't shoot from the 3-point line, for example.
          So the final dataset is prepared and it has been changed accordingly to what feature was relevant for every analysis done on it.
          ''')
          st.write(''' Here is attached a description of the variables with the main percentiles, the mean and the count.
          With this description is easier to analize the data for the analysis.
          ''')
          st.dataframe(Adv_Stats_1.describe()) 
        
# mettere nelle singole sezioni le sotto tabelline fatte ma non nella parte di data cleaning dato che non sono salvataggi

if sec == 'LeBron James exploration and analysis':
    st.header('LeBron James exploration and analysis')

    st.write(''' In this section will be exploited the LeBron James exploration and analysis about the 'Per Game' and 'Totals' statistics.
    With this analysis we want to show Lebron's trends from many points of view.
    ''')

    st.subheader('Variables Histograms') 
    st.write('''
    In this subsection will be presented some histograms of the main variables used for the analysis.
    For the histograms are used the 'Per Game' statistics and the main variables included are: PTS, AST, TRB, FG, 3P, STL, TOV and BLK. 
    ''')
    fig = plt.figure(figsize=(14,12))
    plt.subplot(421)
    plt.title('PTS')
    plt.hist(LeB_C_PG_RS['PTS'], bins=20)
    plt.subplot(422)
    plt.title('AST')
    plt.hist(LeB_C_PG_RS['AST'], bins=20)
    plt.subplot(423)
    plt.title('TRB')
    plt.hist(LeB_C_PG_RS['TRB'], bins=20)
    plt.subplot(424)
    plt.title('FG')
    plt.hist(LeB_C_PG_RS['FG'], bins=20)
    plt.subplot(425)
    plt.title('3P')
    plt.hist(LeB_C_PG_RS['3P'], bins=20)
    plt.subplot(426)
    plt.title('STL')
    plt.hist(LeB_C_PG_RS['STL'], bins=20)
    plt.subplot(427)
    plt.title('TOV')
    plt.hist(LeB_C_PG_RS['TOV'], bins=20)
    plt.subplot(428)
    plt.title('BLK')
    plt.hist(LeB_C_PG_RS['BLK'], bins=20)
    st.pyplot(fig)
    st.write(''' As we can see from this histograms there is no big variance in the distribution of these variables, because they are, with the exception of some outlier, quite similar and show a kind of trend in his career.
    For example: PTS are distribuited between 25 and 31, TRB are distribuited between 7 and 8.5 and FG are distribuited between 9 and 10.5.
    So, it can be confirmed that LeBron usually performs in a quite well defined range and also these ranges are higher than the most number of players confirming that LeBron is a superstar in the NBA.  
    ''')

    st.subheader('Correlation Analysis')
    st.write(''' Now will be shown the correlation heatmap of the variables in the 'Per Game' dataset.
    In this way, it will be possibile to understand how the different variables relates each other and how they do this.
    Firstly it will be presented the whole dataset, then it will be proposed a more specific case with less variables.
    ''')
    fig = plt.figure(figsize=(12, 10))
    plt.subplot(111)
    sb.heatmap(LeB_C_PG_RS.corr(), annot=True)
    st.pyplot(fig)
    st.write('''As we can see from the heatmap some variables are really correlated each other and it was predictable because some of them are connected.
    Now will be shown a restricted heatmap with the main variables of the dataset and how they are correlated each other.
    ''')
    fig = plt.figure(figsize=(12, 10))
    plt.subplot(111)
    sb.heatmap(LeB_C_PG_RS.drop(columns=['ORB', 'DRB', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'PF']).corr(), annot=True)
    st.pyplot(fig)
    st.write('''Now is easier to see and recognise a relationship between the different variables. 
    So, now we can see that the percentages are correlated each other and this was predictable, then also MP and STL are correlated and this seems strange but also reasonable.
    MP is also negative correlated to many variables, AST and TRB are quite correlated and so on.
    From the heatmap we can understand many features of the dataset and how it works. 
    ''')

    st.subheader('Analysis with Team') 
    st.write(''' Here will be presented an analysis concerning the three teams with LeBron James played in the league.
    For these teams will be shown some plots and a comparison of his trends in the different experiences with this teams.
    ''')
    Team = list(LeB_C_PG_RS['Tm'])
    CLE = Team.count('CLE')
    MIA = Team.count('MIA')
    LAL = Team.count('LAL')

    Teams = ['CLE', 'MIA', 'LAL']
    count = [CLE, MIA, LAL]
    fig = plt.figure(figsize=(10,6))
    plt.pie(count, labels=Teams, autopct='%.2f%%')
    st.pyplot(fig)

    #stats medie divise per squadra
    st.write(''' From this pie chart we can see that he played most time at Cleveland Cavaliers and spent just four seasons for the other teams each.
    Now we will see how the mean of his statistics are divided between the different experiences he made during his career.
    There are four outputs because the cavs experience is splitted in two parts according to direct succession of the team change made by him.
    He was drafted by Cavs then he went to Heat and after this experience he came back to Cavs and after four years he went to Lakers where he still is. 
    ''')
    stat = st.selectbox('Choose a Stat', LeB_C_PG_RS.columns.drop(['Season', 'Tm', 'Pos']).tolist(), key=3)
    df = LeB_C_PG_RS.groupby(by='Tm').mean()
    st.dataframe(df[stat])

    st.subheader('Analysis with Pos')
    st.write(''' In this subsection will be shown an analysis of the LeBron trends according to the different roles he played during his whole career.
    As for the analysis for the teams also in this case will be presented some plots and a comparison of his trends.
    ''')
    Pos = list(LeB_C_PG_RS['Pos'])
    PG = Pos.count('PG')
    PF = Pos.count('PF')
    SG = Pos.count('SG')
    SF = Pos.count('SF')

    Positions = ['PG', 'PF', 'SG', 'SF']
    count = [PG, PF, SG, SF]
    fig = plt.figure(figsize=(10,6))
    plt.pie(count, labels=Positions, autopct='%.2f%%')
    st.pyplot(fig)

    #stats medie divise per ruolo
    st.write(''' From this pie chart we can see that he played the most time of his career as SF, but anyway he changed many roles, showing how he is versatile.
    Now we will analyse the mean of his statistics accordingly to the role he played that season showing how he changed his trends also because of the role.
    ''')
    stat = st.selectbox('Choose a Stat', LeB_C_PG_RS.columns.drop(['Season', 'Tm', 'Pos']).tolist(), key=4)
    df = LeB_C_PG_RS.groupby(by='Pos').mean()
    st.dataframe(df[stat])
    
    #show plot misto con più di una stat
    st.write(''' After all this analysis will be provided some mixed plots containing more variables just to show the trend of these variables.
    For example, the first plot contains the percentages of shooting, the second contains a comparison between assists and rebounds and the last one shows the comparison between steels and turnovers.
    ''')
    Season = list(LeB_C_PG_RS['Season'])
    DPP = list(LeB_C_PG_RS['2P%'])
    TPP = list(LeB_C_PG_RS['3P%'])
    FGP = list(LeB_C_PG_RS['FG%'])
    FTP = list(LeB_C_PG_RS['FT%'])
    AST = list(LeB_C_PG_RS['AST'])
    TRB = list(LeB_C_PG_RS['TRB'])
    STL = list(LeB_C_PG_RS['STL'])
    TOV = list(LeB_C_PG_RS['TOV'])

    fig = plt.figure(figsize=(10, 6))
    plt.plot(Season, DPP, '-o', label='2P%')
    plt.plot(Season, TPP, '-o', label='3P%')
    plt.plot(Season, FGP, '-o', label='FG%')
    plt.plot(Season, FTP, '-o', label='FT%')
    plt.title('LeBron accuracy % in every season')
    plt.xlabel('Seasons')
    plt.ylabel('Shooting percentages')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(Season, AST, '-o', label='AST')
    plt.plot(Season, TRB, '-*', label='TRB')
    plt.title('LeBron AST/TRB averages in every season')
    plt.xlabel('Seasons')
    plt.ylabel('AST-TRB')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(Season, STL, '-o', label='STL')
    plt.plot(Season, TOV, '-o', label='TOV')
    plt.title('LeBron STL/TOV averages in every season')
    plt.xlabel('Seasons')
    plt.ylabel('STL/TOV')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    selection = st.radio('Choose a dataset', ('Per Game Stats', 'Totals Stats'))

    if selection == 'Per Game Stats':
      x = st.selectbox('Choose a Stat', LeB_C_PG_RS.columns.drop(['Season', 'Tm', 'Pos']).tolist(), key=0)

      y = st.selectbox('Choose a feature', ['Max', 'Min', 'Mean'], key=1)

      if y == 'Max':
        ms = LeB_C_PG_RS[x].max()
        num = int(LeB_C_PG_RS[LeB_C_PG_RS[x]==ms].index.to_list()[0])
        a = LeB_C_PG_RS.loc[num,'Season']
        st.write('The max has been registered in the: ', a, ' season. The max for this stat is: ', str(ms)) 
      if y == 'Min':
        ms = LeB_C_PG_RS[x].min()
        num = int(LeB_C_PG_RS[LeB_C_PG_RS[x]==ms].index.to_list()[0])
        a = LeB_C_PG_RS.loc[num, 'Season']
        st.write('The min has been registered in the: ', a, ' season. The min for this stat is: ', str(ms)) 
      if y == 'Mean':
        ms = LeB_C_PG_RS[x].mean()
        st.write('The mean for this stat is: ', str(ms))
      
      Season = list(LeB_C_PG_RS['Season'])
      Peppino = list(LeB_C_PG_RS[x])
      fig = plt.figure(figsize=(10, 6))
      plt.plot(Season, Peppino, '-o')
      plt.title('LeBron ' + x + ' averages in every season')
      plt.xlabel('Seasons')
      plt.ylabel(x)
      plt.xticks(rotation=45)
      st.pyplot(fig)

    
    if selection == 'Totals Stats':
      x = st.selectbox('Choose a Stat', LeB_C_Tot_RS.columns.drop(['Season', 'Tm', 'Pos']).tolist(), key=0)

      y = st.selectbox('Choose a feature', ['Max', 'Min', 'Mean'], key=1)

      if y == 'Max':
        ms = LeB_C_Tot_RS[x].max()
        num = int(LeB_C_Tot_RS[LeB_C_Tot_RS[x]==ms].index.to_list()[0])
        a = LeB_C_Tot_RS.loc[num,'Season']
        st.write('The max has been registered in the: ', a, ' season. The max for this stat is: ', str(ms)) 
      if y == 'Min':
        ms = LeB_C_Tot_RS[x].min()
        num = int(LeB_C_Tot_RS[LeB_C_Tot_RS[x]==ms].index.to_list()[0])
        a = LeB_C_Tot_RS.loc[num, 'Season']
        st.write('The min has been registered in the: ', a, ' season. The min for this stat is: ', str(ms)) 
      if y == 'Mean':
        ms = LeB_C_Tot_RS[x].mean()
        st.write('The mean for this stat is: ', str(ms))
      
      Season = list(LeB_C_Tot_RS['Season'])
      Peppino = list(LeB_C_Tot_RS[x])
      fig = plt.figure(figsize=(10, 6))
      plt.plot(Season, Peppino, '-o')
      plt.title('LeBron ' + x + ' averages in every season')
      plt.xlabel('Seasons')
      plt.ylabel(x)
      plt.xticks(rotation=45)
      st.pyplot(fig)


if sec == 'Predictive model for LeBron James':
    st.header('Predictive model for LeBron James')

# mettere i modelli che ci sono sull'ipynb generici con la possibilità di scegliere la statistica da predire con anche la possibilità di mettere dei valori in input ecc
# vedere che modelli mettere, capire che altre analisi vanno fatte e se sono abbastanza

if sec == 'Season 2020/2021 exploration and analysis':
    st.header('Season 2020/2021 exploration and analysis')

    st.write(''' In this section will be exploited the analysis about the regular season 2020/21 splitted into two datasets, 'Per Game' and 'Advanced'.
    Firstly will be presented a comparison between the different teams and the number of players that played for that team. 
    As we can see the range of players for every team is from 16 to 28. But are included also the players who changed teams and didn't play many matches for that team.
    ''')
    st.dataframe(pd.DataFrame(Adv_Stats_1.groupby(by='Tm').count()['Player']).T)
    st.write(''' Now will be presented a comparison between the roles.
    As we can see the most spread role is the SG and the less spread is the SF. This is not very indicative because some players can play different role, but they are registered with the role they played the most during this season.
    ''')
    st.dataframe(pd.DataFrame(Adv_Stats_1.groupby(by='Pos').count()['Player']))
    st.write(''' Now the analysis will be splitted into the two different dataset and for each of them will be conducted similar analysis.
    ''')

    selection = st.radio('Choose a dataset', ('Per Game stats dataset', 'Advanced stats dataset'))

    if selection == 'Per Game stats dataset':

      x = st.selectbox('Choose a Stat', xG_Stats_1.columns.drop(['Player', 'Pos', 'Tm']).tolist(), key=0)
      y = st.selectbox('Choose a feature', ['Max', 'Min', 'Mean'], key=1)

      if y == 'Max':
        ms = xG_Stats_1[x].max()
        num = int(xG_Stats_1[xG_Stats_1[x]==ms].index.to_list()[0])
        a = xG_Stats_1.loc[num,'Player']
        st.write('The max for this statistic is: ', a, '. The max for this stat is: ', str(ms)) 
      if y == 'Min':
        ms = xG_Stats_1[x].min()
        num = int(xG_Stats_1[xG_Stats_1[x]==ms].index.to_list()[0])
        a = xG_Stats_1.loc[num, 'Player']
        st.write('The min for this statistic is: ', a, '. The min for this stat is: ', str(ms)) 
      if y == 'Mean':
        ms = xG_Stats_1[x].mean()
        st.write('The mean for this statistic is: ', str(ms))

      st.subheader('Variables Histograms')
      fig = plt.figure(figsize=(14,12))
      plt.subplot(431)
      plt.title('PTS')
      plt.hist(xG_Stats_1['PTS'], bins=20)
      plt.subplot(432)
      plt.title('AST')
      plt.hist(xG_Stats_1['AST'], bins=20)
      plt.subplot(433)
      plt.title('TRB')
      plt.hist(xG_Stats_1['TRB'], bins=20)
      plt.subplot(434)
      plt.title('FG%')
      plt.hist(xG_Stats_1['FG%'], bins=20)
      plt.subplot(435)
      plt.title('3P%')
      plt.hist(xG_Stats_1['3P%'], bins=20)
      plt.subplot(436)
      plt.title('2P%')
      plt.hist(xG_Stats_1['2P%'], bins=20)
      plt.subplot(437)
      plt.title('FT%')
      plt.hist(xG_Stats_1['FT%'], bins=20)
      plt.subplot(438)
      plt.title('BLK')
      plt.hist(xG_Stats_1['BLK'], bins=20)
      plt.subplot(439)
      plt.title('G')
      plt.hist(xG_Stats_1['G'], bins=20)
      plt.subplot(4,3,10)
      plt.title('MP')
      plt.hist(xG_Stats_1['MP'], bins=20)
      plt.subplot(4,3,11)
      plt.title('STL')
      plt.hist(xG_Stats_1['STL'], bins=20)
      plt.subplot(4,3,12)
      plt.title('TOV')
      plt.hist(xG_Stats_1['TOV'], bins=20)
      st.pyplot(fig)
      
      team_player_groupby = xG_Stats_1.groupby(['Tm', 'Player'])
      team_df = pd.DataFrame(team_player_groupby[['PTS', 'AST', 'TRB', 'BLK', 'G', 'MP']].mean())
      Team_chosen = st.selectbox('Choose a Team', np.unique(xG_Stats_1['Tm']), key=1)
      team_df.loc[(Team_chosen),:]

      st.subheader('Accuracy Analysis')
      st.write(''' In this subsection we will show the field goal percentage calculated and not taken by the stats.
      We also added a mask on the number of shots tried per game, in order to take only the player who usually play many minuts in a game.
      ''')

      most_FGA = xG_Stats_1.sort_values(by='FGA', ascending=False)[['Player', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA']]
      most_FGA.set_index('Player', inplace=True)
      #Var1_chosen = st.selectbox('Choose a Percentile', most_FGA[['FG', '3P', '2P', 'FT']], key=2)
      #Var2_chosen = st.selectbox('Choose a Percentile', most_FGA[['FGA', '3PA', '2PA', 'FTA']], key=3)
      #most_FGA['ACCURACY'] = most_FGA[Var1_chosen] / most_FGA[Var2_chosen]
      most_FGA['ACCURACY'] = most_FGA['FG'] / most_FGA['FGA']
      most_accurate_mask = most_FGA['ACCURACY'] == most_FGA['ACCURACY'].max()
      n = st.slider('Choose a number of shots', min_value=0, max_value=23)
      at_least_n_shots_xG = most_FGA['FGA'] >= n
      st.write(''' Here we present the players with the highest accuracy without considering the mask on the number of shots.
      ''')
      most_FGA[most_accurate_mask]
      st.write(''' While here we show the players who overpass the mask in this season with their statistics.
      ''')
      most_FGA[at_least_n_shots_xG]
      st.write('''Finally, we show the player with the highest accuracy considering also the mask on the number of shots.
      ''')
      most_FGA[most_FGA['ACCURACY'] == most_FGA[at_least_n_shots_xG]['ACCURACY'].max()]
      st.write(''' In conclusion, we can see that this new variable is the same as 'FG%' with the difference that 'FG%' don't come from approximation on the number of successful shots and tried, but from the exact ratio of the two.
      So we can say that 'ACCURACY' is less precise than 'FG%'.
      ''')
      
      st.subheader('Main Variables Analysis')
      #fare i write
      st.write(''' In this subsection will be shown the some features of the main variables, like PTS, AST and TRB.
      For this three variables will be plotted bar charts in order to show the top 25 in each statistic.
      ''')
      most_points = xG_Stats_1.sort_values(by='PTS', ascending=False)[['Player', 'PTS', 'AST', 'TRB', 'BLK', 'STL']]
      most_points.reset_index().drop(columns='index')
      st.write(most_points)
      player_name = st.selectbox('Player name', xG_Stats_1['Player'], key=0)
      st.write('Main statistics for ', player_name, ' are equal to:', most_points[most_points['Player'] == player_name])

      st.vega_lite_chart(most_points, {'width': 500, 'height': 500, 'mark' : {'type':'circle', 'tooltip':True}, 'encoding' : {
        'x': {'field': 'AST', 'type': 'quantitative'},
        'y': {'field': 'TRB', 'type': 'quantitative'},
        'size': {'field': 'PTS', 'type': 'quantitative'},
        'color': {'field': 'Player', 'type': 'nominal'}}})

      most_points.set_index('Player', inplace=True)  
      x = most_points[:25].index
      y = most_points[:25]['PTS']
      fig = plt.figure(figsize=(10,6))
      plt.title('Top 25 players by points')
      plt.bar(x, y)
      plt.ylabel('Points')
      plt.xticks(rotation=90)
      st.pyplot(fig)

      most_assists = xG_Stats_1.sort_values(by='AST', ascending=False)[['Player', 'PTS', 'AST', 'TRB', 'BLK', 'STL']]
      most_assists.set_index('Player', inplace=True)  
      x = most_assists[:25].index
      y = most_assists[:25]['AST']
      fig = plt.figure(figsize=(10,6))
      plt.title('Top 25 players by assists')
      plt.bar(x, y)
      plt.ylabel('Assists')
      plt.xticks(rotation=90)
      st.pyplot(fig)

      most_rebounds = xG_Stats_1.sort_values(by='TRB', ascending=False)[['Player', 'PTS', 'AST', 'TRB', 'BLK', 'STL']]
      most_rebounds.set_index('Player', inplace=True)  
      x = most_rebounds[:25].index
      y = most_rebounds[:25]['TRB']
      fig = plt.figure(figsize=(10,6))
      plt.title('Top 25 players by rebounds')
      plt.bar(x, y)
      plt.ylabel('Total Rebounds')
      plt.xticks(rotation=90)
      st.pyplot(fig)
      
      #altre cose con per game dataset

    if selection == 'Advanced stats dataset':

      x = st.selectbox('Choose a Stat', Adv_Stats_1.columns.drop(['Player', 'Pos', 'Tm']).tolist(), key=0)
      y = st.selectbox('Choose a feature', ['Max', 'Min', 'Mean'], key=1)

      if y == 'Max':
        ms = Adv_Stats_1[x].max()
        num = int(Adv_Stats_1[Adv_Stats_1[x]==ms].index.to_list()[0])
        a = Adv_Stats_1.loc[num,'Player']
        st.write('The max for this statistic is: ', a, '. The max for this stat is: ', str(ms)) 
      if y == 'Min':
        ms = Adv_Stats_1[x].min()
        num = int(Adv_Stats_1[Adv_Stats_1[x]==ms].index.to_list()[0])
        a = Adv_Stats_1.loc[num, 'Player']
        st.write('The min for this statistic is: ', a, '. The min for this stat is: ', str(ms)) 
      if y == 'Mean':
        ms = Adv_Stats_1[x].mean()
        st.write('The mean for this statistic is: ', str(ms))
      
      st.subheader('Variables Histograms')
      fig = plt.figure(figsize=(14,12))
      plt.subplot(441)
      plt.title('MP')
      plt.hist(Adv_Stats_1['MP'], bins=20)
      plt.subplot(442)
      plt.title('G')
      plt.hist(Adv_Stats_1['G'], bins=20)
      plt.subplot(443)
      plt.title('Age')
      plt.hist(Adv_Stats_1['Age'], bins=20)
      plt.subplot(444)
      plt.title('Pos')
      plt.hist(Adv_Stats_1['Pos'], bins=20)
      plt.subplot(445)
      plt.title('BPM')
      plt.hist(Adv_Stats_1['BPM'], bins=20)
      plt.subplot(446)
      plt.title('DBPM')
      plt.hist(Adv_Stats_1['DBPM'], bins=20)
      plt.subplot(447)
      plt.title('OBPM')
      plt.hist(Adv_Stats_1['OBPM'], bins=20)
      plt.subplot(448)
      plt.title('VORP')
      plt.hist(Adv_Stats_1['VORP'], bins=20)
      plt.subplot(449)
      plt.title('WS')
      plt.hist(Adv_Stats_1['WS'], bins=20)
      plt.subplot(4,4,10)
      plt.title('WS/48')
      plt.hist(Adv_Stats_1['WS/48'], bins=20)
      plt.subplot(4,4,11)
      plt.title('OWS')
      plt.hist(Adv_Stats_1['OWS'], bins=20)
      plt.subplot(4,4,12)
      plt.title('DWS')
      plt.hist(Adv_Stats_1['DWS'], bins=20)
      plt.subplot(4,4,13)
      plt.title('USG%')
      plt.hist(Adv_Stats_1['USG%'], bins=20)
      plt.subplot(4,4,14)
      plt.title('TS%')
      plt.hist(Adv_Stats_1['TS%'], bins=20)
      plt.subplot(4,4,15)
      plt.title('3PAr')
      plt.hist(Adv_Stats_1['3PAr'], bins=20)
      plt.subplot(4,4,16)
      plt.title('PER')
      plt.hist(Adv_Stats_1['PER'], bins=20)
      st.pyplot(fig)

      team_player_groupby = Adv_Stats_1.groupby(['Tm', 'Player'])
      team_df = pd.DataFrame(team_player_groupby[['TS%', '3PAr', 'AST%', 'TRB%', 'VORP', 'USG%', 'G', 'MP']].mean())
      Team_chosen = st.selectbox('Choose a Team', np.unique(Adv_Stats_1['Tm']), key=1)
      team_df.loc[(Team_chosen),:]

      st.subheader('Win Share Analysis')
      st.write(''' As we could know, win share is an important variable that explains how crucial is a player for a team due to the effort he gives in the team victories.
      For this reason here are shown some stats that focuses on this statistic for the season 2020/21.
      ''')
      win_share = Adv_Stats_1.sort_values(by='WS', ascending=False)[['Player', 'WS', 'WS/48', 'OWS', 'DWS']]
      win_share = win_share.reset_index().drop(columns='index')
      st.write(win_share)
      st.write(''' We can see that the best player in this statistics is Nikola Jokic with 15.6, instead some players have also a negative win share and the worst for this statistic is Aleksej Pokusevski.
      For this variable are also provided three similar statistics: those are the win share for 48 minutes, that is a win share standardised on the entire duration of a game;
      the offensive and the difensive win share that summed up compose the win share statistic.
      ''')
      player_name = st.selectbox('Player name', Adv_Stats_1['Player'], key=0)
      st.write('Ws statistics for ', player_name, ' are equal to:', win_share[win_share['Player'] == player_name])

      st.vega_lite_chart(win_share, {'width': 500, 'height': 500, 'mark' : {'type':'circle', 'tooltip':True}, 'encoding' : {
        'x': {'field': 'OWS', 'type': 'quantitative'},
        'y': {'field': 'DWS', 'type': 'quantitative'},
        'color': {'field': 'Player', 'type': 'nominal'}}})

      win_share.set_index('Player', inplace=True)  
      x = win_share[:25].index
      y = win_share[:25]['WS']
      fig = plt.figure(figsize=(10,6))
      plt.title('Top 25 players by win share')
      plt.bar(x, y)
      plt.ylabel('Win Share')
      plt.xticks(rotation=90)
      st.pyplot(fig)

      st.subheader('Box Plus/Minus Analysis')
      st.write(''' Another important variable for the analysis of the players is the box plus/minus. 
      This variable shows how many benefits or malus gives a player to his team while playing.
      If this variable is high this means that when the player is in the field he provide his team a huge benefit, instead when this variable is lower or negative, the player is not so useful for his team.
      For this reason here are shown some stats that focuses on this statistic for the season 2020/21.
      ''')
      most_BPM = Adv_Stats_1.sort_values(by='BPM', ascending=False)[['Player', 'BPM', 'OBPM', 'DBPM', 'VORP']]
      most_BPM = most_BPM.reset_index().drop(columns='index')
      st.write(most_BPM)
      st.write(''' As we can see from the dataframe, there is a player, Udonis Haslem, that has provided his team, while playing, more than 30 points.
      This incredible results leads him to be the best player on this statistic. For the other players, the results are more intuitive, becuase they have a range between 12 and -6 for most players. 
      There are also many many players with a huge negative impact in their team and the worst is Anžejs Pasečņiks with -46.6.
      ''')
      player_name = st.selectbox('Player name', Adv_Stats_1['Player'], key=1)
      st.write('Ws statistics for ', player_name, ' are equal to:', most_BPM[most_BPM['Player'] == player_name])

      st.vega_lite_chart(most_BPM, {'width': 500, 'height': 500, 'mark' : {'type':'circle', 'tooltip':True}, 'encoding' : {
        'x': {'field': 'OBPM', 'type': 'quantitative'},
        'y': {'field': 'DBPM', 'type': 'quantitative'},
        'color': {'field': 'Player', 'type': 'nominal'}}})
      
      most_BPM.set_index('Player', inplace=True)  
      x = most_BPM[:25].index
      y = most_BPM[:25]['BPM']
      fig = plt.figure(figsize=(10,6))
      plt.title('Top 25 players by box plus/minus')
      plt.bar(x, y)
      plt.ylabel('Box Plus/Minus')
      plt.xticks(rotation=90)
      st.pyplot(fig)

      st.write(''' To reduce the effect of outliers, to calculate the BPM we add a mask that include the players that have played at least 30 games.
      With this mask we can see that now the players in the chart are well-known and still they have good BPM value.
      ''')
      BPM_with_mask = Adv_Stats_1.sort_values(by='BPM', ascending=False)[['Player', 'BPM', 'OBPM', 'DBPM', 'VORP', 'G']]
      BPM_with_mask.set_index('Player', inplace=True)
      games_mask_30 = BPM_with_mask['G'] > 30
      x = BPM_with_mask[games_mask_30][:25].index
      y = BPM_with_mask[games_mask_30][:25]['BPM']
      fig = plt.figure(figsize=(10,6))
      plt.title('Top 25 players by box plus/minus with at least 30 games played')
      plt.bar(x, y)
      plt.ylabel('Box Plus/Minus')
      plt.xticks(rotation=90)
      st.pyplot(fig)

      #altre cose con advanced dataset

if sec == 'Predictive model for Season 2020/2021':
    st.header('Predictive model for Season 2020/2021')
    st.write('''In this section will be exploited the predictive part of the project concerning the season 2020/21.
    Here will be made different regressions according to the two datasets used.
    ''')

    selection = st.radio('Choose a dataset', ('Per Game stats dataset', 'Advanced stats dataset'))
    
    if selection == 'Per Game stats dataset':

      st.subheader('Per Game Dataset')
      st.write(''' In this subsection will be provided a regression to predict the PTS realized by a player setting determinated values on the variable.
      ''')

      #regressione con vari metodi
      xG_data = xG_Stats_1[['Player', 'PTS', 'AST', 'TRB', 'MP', 'BLK', 'STL', 'TOV', 'FGA', 'Age', 'Pos']]
      xG_data.sort_values(by='Player')
      xG_datum = pd.concat([xG_data, pd.get_dummies(xG_data['Pos'])], axis=1).drop(columns='Pos')
      x = xG_datum.drop(columns=['Player', 'PTS'])
      y = xG_datum['PTS']
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

      fig = plt.figure(figsize=(12, 10))
      plt.subplot(111)
      sb.heatmap(xG_datum.corr(), annot=True)
      st.pyplot(fig)

      model_1 = st.selectbox('Choose a Regression', [RandomForestRegressor, LinearRegression, Lasso, Ridge])
      model = model_1().fit(x_train, y_train)
      y_pred = model.predict(x_test)

      #variables to be setted for the prediction
      st.write('Number of assist per game:')
      first = st.slider('Slide me', min_value= 0.0, max_value=20.0, key=0)
      st.write('Number of total rebound per game:')
      second = st.slider('Slide me', min_value= 0.0, max_value=20.0, key=1)
      st.write('Number of minutes played per game:')
      third = st.slider('Slide me', min_value= 0.0, max_value=42.0, key=2)
      st.write('Number of blocks per game:')
      fourth = st.slider('Slide me', min_value= 0.0, max_value=8.0, key=3)
      st.write('Number of steels per game:')
      fifth = st.slider('Slide me', min_value= 0.0, max_value=8.0, key=4)
      st.write('Number of turnovers per game:')
      sixth = st.slider('Slide me', min_value= 0.0, max_value=8.0, key=5)
      st.write('Number of field goal attempted per game:')
      seventh = st.slider('Slide me', min_value= 0.0, max_value=30.0, key=6)
      st.write('Age:')
      eighth = st.slider('Slide me', min_value= 17, max_value=46, step=1, key=7)
      st.write('Role played:')
      dummy = st.select_slider('Slide me', ['C', 'PG', 'PF', 'SG', 'SF'], key=8)
      ninth = 0
      tenth = 0
      eleventh = 0
      twelth = 0
      thirteenth = 0
      if dummy == 'C':
        ninth = 1
      if dummy == 'PG':
        tenth = 1
      if dummy == 'PF':
        eleventh = 1
      if dummy == 'SG':
        twelth = 1
      if dummy == 'SF':
        thirteenth = 1
          
      if st.button('Calculate'):
        pts = model.predict(X=[[first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelth, thirteenth]])
        st.write('The predicted number of points with these inputs is: ', str(list(pts)))

      st.write('''With this model we can predict the expected number of points realized by a player according to setted values for the statistics.
      This model is quite well performed according to the R^2 score it has: ''', str(model.score(x_test, y_test)) , '''. To realize this regression, Random Forest Regressor has been used and the dataset has been splitted using the comand train_test_split.
      ''')

      # mettere dei plot per far vedere la distribuzione ecc della regressione



    if selection == 'Advanced stats dataset':

      st.subheader('Advanced Dataset')
      st.write(''' In this subsection will be provided three different regression: the first to predict the VORP, the second to predict the BPM, and the last one to predict the WS.
      ''')

      sel = st.radio('Choose a Regression', ('VORP', 'BPM', 'WS'))

      if sel == 'VORP':
        st.subheader('VORP Regression')

        Adv_data = Adv_Stats_1[['Player', 'VORP', 'BPM', 'WS', 'TS%', 'TRB%', 'AST%', 'PER', 'USG%', 'Age', 'Pos']]
        Adv_data.sort_values(by='Player')
        Adv_datum = pd.concat([Adv_data, pd.get_dummies(Adv_data['Pos'])], axis=1).drop(columns='Pos')
        x = Adv_datum.drop(columns=['Player', 'VORP'])
        y = Adv_datum['VORP']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        fig = plt.figure(figsize=(12, 10))
        plt.subplot(111)
        sb.heatmap(Adv_datum.corr(), annot=True)
        st.pyplot(fig)

        model_1 = st.selectbox('Choose a Regression', [RandomForestRegressor, LinearRegression, Lasso, Ridge], key=0)
        model = model_1().fit(x_train, y_train)
        y_pred = model.predict(x_test)

        #variables to be setted for the prediction
        st.write('Value of BPM:')
        first = st.slider('Slide me', min_value= -25.0, max_value=25.0, key=0)
        st.write('Value of WS:')
        second = st.slider('Slide me', min_value= -10.0, max_value=20.0, key=1)
        st.write('True Shooting Percentage:')
        third = st.slider('Slide me', min_value= 0.0, max_value=1.0, key=2)
        st.write('Toal Rebound Percentage:')
        fourth = st.slider('Slide me', min_value= 0.0, max_value=30.0, key=3)
        st.write('Assist Percentage:')
        fifth = st.slider('Slide me', min_value= 0.0, max_value=50.0, key=4)
        st.write('Player Efficiency Rating:')
        sixth = st.slider('Slide me', min_value= 5.0, max_value=35.0, key=5)
        st.write('Usage Percentage:')
        seventh = st.slider('Slide me', min_value= 5.0, max_value=40.0, key=6)
        st.write('Age:')
        eighth = st.slider('Slide me', min_value= 17, max_value=46, step=1, key=7)
        st.write('Role played:')
        dummy = st.select_slider('Slide me', ['C', 'PG', 'PF', 'SG', 'SF'], key=8)
        ninth = 0
        tenth = 0
        eleventh = 0
        twelth = 0
        thirteenth = 0
        if dummy == 'C':
          ninth = 1
        if dummy == 'PG':
          tenth = 1
        if dummy == 'PF':
          eleventh = 1
        if dummy == 'SG':
          twelth = 1
        if dummy == 'SF':
          thirteenth = 1
            
        if st.button('Calculate'):
          vorp = model.predict(X=[[first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelth, thirteenth]])
          st.write('The predicted value over replacement player with these inputs is: ', str(list(vorp)))

        st.write('''With this model we can predict the expected value over replacement player, that is A box score estimate of the points per 100 TEAM possessions that a player contributed above a replacement-level (-2.0) player, translated to an average team and prorated to an 82-game season.
        Value over Replacement Player (VORP) converts the BPM rate into an estimate of each player's overall contribution to the team, measured vs. what a theoretical "replacement player" would provide, where the "replacement player" is defined as a player on minimum salary or not a normal member of a team's rotation.
        This model is quite well performed according to the R^2 score it has: ''', str(model.score(x_test, y_test)) , '''. To realize this regression, the dataset has been splitted using the comand train_test_split.
        ''')

      if sel == 'BPM':
        st.subheader('BPM Regression')

        data = Adv_Stats_1[['Player', 'BPM', 'G', 'VORP', 'WS', 'WS/48', 'PER', 'USG%', 'TS%', 'Pos']]
        data.sort_values(by='Player')
        data1 = pd.concat([data, pd.get_dummies(data['Pos'])], axis=1).drop(columns='Pos')
        x = data1.drop(columns=['Player', 'BPM'])
        y = data1['BPM']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        fig = plt.figure(figsize=(12, 10))
        plt.subplot(111)
        sb.heatmap(data.corr(), annot=True)
        st.pyplot(fig)

        model_1 = st.selectbox('Choose a Regression', [RandomForestRegressor, LinearRegression, Lasso, Ridge], key=1)
        model = model_1().fit(x_train, y_train)
        y_pred = model.predict(x_test)

        #variables to be setted for the prediction
        st.write('Number of Games played:')
        first = st.slider('Slide me', min_value= 0, max_value=72, key=0)
        st.write('Value of VORP:')
        second = st.slider('Slide me', min_value= -5.0, max_value=10.0, key=1)
        st.write('Value of WS:')
        third = st.slider('Slide me', min_value= -10.0, max_value=20.0, key=2)
        st.write('Value of WS/48:')
        fourth = st.slider('Slide me', min_value= -0.5, max_value=1.0, key=3)
        st.write('Player Efficiency Rating:')
        fifth = st.slider('Slide me', min_value= 5.0, max_value=35.0, key=4)
        st.write('Usage Percentage:')
        sixth = st.slider('Slide me', min_value= 5.0, max_value=40.0, key=5)
        st.write('True Shooting Percentage:')
        seventh = st.slider('Slide me', min_value= 0.0, max_value=1.0, key=6)
        st.write('Role played:')
        dummy = st.select_slider('Slide me', ['C', 'PG', 'PF', 'SG', 'SF'], key=7)
        eighth = 0
        ninth = 0
        tenth = 0
        eleventh = 0
        twelth = 0
        if dummy == 'C':
          eighth = 1
        if dummy == 'PG':
          ninth = 1
        if dummy == 'PF':
          tenth = 1
        if dummy == 'SG':
          eleventh = 1
        if dummy == 'SF':
          twelth = 1
            
        if st.button('Calculate'):
          bpm = model.predict(X=[[first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelth]])
          st.write('The predicted value over replacement player with these inputs is: ', str(list(bpm)))

        st.write('''With this model we can predict the expected number of BPM, that is A box score estimate of the points provided by a player while playing.
        This model is quite well performed according to the R^2 score it has: ''', str(model.score(x_test, y_test)) , '''. To realize this regression, the dataset has been splitted using the comand train_test_split.
        ''')

      if sel == 'WS':
        st.subheader('WS Regression')

        data = Adv_Stats_1[['Player', 'WS', 'G', 'VORP', 'BPM', 'WS/48', 'PER', 'USG%', 'TS%', 'Pos']]
        data.sort_values(by='Player')
        data1 = pd.concat([data, pd.get_dummies(data['Pos'])], axis=1).drop(columns='Pos')
        x = data1.drop(columns=['Player', 'WS'])
        y = data1['WS']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        fig = plt.figure(figsize=(12, 10))
        plt.subplot(111)
        sb.heatmap(data.corr(), annot=True)
        st.pyplot(fig)

        model_1 = st.selectbox('Choose a Regression', [RandomForestRegressor, LinearRegression, Lasso, Ridge], key=2)
        model = model_1().fit(x_train, y_train)
        y_pred = model.predict(x_test)

        #variables to be setted for the prediction
        st.write('Number of Games played:')
        first = st.slider('Slide me', min_value= 0, max_value=72, key=0)
        st.write('Value of VORP:')
        second = st.slider('Slide me', min_value= -5.0, max_value=10.0, key=1)
        st.write('Value of BPM:')
        third = st.slider('Slide me', min_value= -25.0, max_value=25.0, key=2)
        st.write('Value of WS/48:')
        fourth = st.slider('Slide me', min_value= -0.5, max_value=1.0, key=3)
        st.write('Player Efficiency Rating:')
        fifth = st.slider('Slide me', min_value= 5.0, max_value=35.0, key=4)
        st.write('Usage Percentage:')
        sixth = st.slider('Slide me', min_value= 5.0, max_value=40.0, key=5)
        st.write('True Shooting Percentage:')
        seventh = st.slider('Slide me', min_value= 0.0, max_value=1.0, key=6)
        st.write('Role played:')
        dummy = st.select_slider('Slide me', ['C', 'PG', 'PF', 'SG', 'SF'], key=7)
        eighth = 0
        ninth = 0
        tenth = 0
        eleventh = 0
        twelth = 0
        if dummy == 'C':
          eighth = 1
        if dummy == 'PG':
          ninth = 1
        if dummy == 'PF':
          tenth = 1
        if dummy == 'SG':
          eleventh = 1
        if dummy == 'SF':
          twelth = 1
            
        if st.button('Calculate'):
          ws = model.predict(X=[[first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelth]])
          st.write('The predicted value over replacement player with these inputs is: ', str(list(ws)))

        st.write('''With this model we can predict the expected number of WS, that is an estimate of the number of wins contributed by a player..
        This model is quite well performed according to the R^2 score it has: ''', str(model.score(x_test, y_test)) , '''. To realize this regression, the dataset has been splitted using the comand train_test_split.
        ''')

      # si potrebbero aggiungere dei grafici
      

# mettere i modelli che ci sono sull'ipynb generici con la possibilità di scegliere la statistica da predire con anche la possibilità di mettere dei valori in input ecc
# vedere che modelli mettere, capire che altre analisi vanno fatte e se sono abbastanza