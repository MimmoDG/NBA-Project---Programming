import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skm
import time
import streamlit.components.v1 as components
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn import cluster
from sklearn.cluster import KMeans
import streamlit_option_menu as som
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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

# forse mettere le heatmap con qualche spiegazione oppure nelle sezioni giuste          
# mettere nelle singole sezioni le sotto tabelline fatte ma non nella parte di data cleaning dato che non sono salvataggi
# controllare se il describe con i vari dataset è giusto per far vedere le variabili 

if sec == 'LeBron James exploration and analysis':
    st.header('LeBron James exploration and analysis')

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
      
    st.write('''
    After showing these statistics now will be presented some histograms of the main variables used for the analysis.
    For the histograms are used the 'Per Game' statistics and the main variables included are: PTS, AST, TRB, FG, 3P, STL, TOV and BLK. 
    ''')
    fig = LeB_C_PG_RS[['PTS', 'AST', 'TRB', 'FG', '3P', 'STL', 'TOV', 'BLK']].hist(bins=20, figsize=(14, 10))
    st.pyplot(fig)
    st.write(''' As we can see from this histograms there is no big variance in the distribution of these variables, because they are, with the exception of some outlier, quite similar and show a kind of trend in his career.
    For example: PTS are distribuited between 25 and 31, TRB are distribuited between 7 and 8.5 and FG are distribuited between 9 and 10.5.
    So, it can be confirmed that LeBron usually performs in a quite well defined range and also these ranges are higher than the most number of players confirming that LeBron is a superstar in the NBA.  
    ''')

    #st.write con spiegazione delle correlazioni e plot
    st.write(''' Now will be shown the correlation heatmap of the variables in the 'Per Game' dataset.
    In this way, it will be possibile to understand how the different variables relates each other and how they do this.
    Firstly it will be presented the whole dataset, then it will be proposed a more specific case with less variables.
    ''')
    LeB_C_PG_RS.corr()
    plt.figure(figsize=(18, 14))
    sb.heatmap(LeB_C_PG_RS.corr(), annot=True)
    st.write('''As  
    ''')
    #continuare descrizione e fare l'altra correlazione

    #st.write con spiegazione del pie plot sulle squadre 
    Team = list(LeB_C_PG_RS['Tm'])
    CLE = Team.count('CLE')
    MIA = Team.count('MIA')
    LAL = Team.count('LAL')

    Teams = ['CLE', 'MIA', 'LAL']
    count = [CLE, MIA, LAL]
    fig = plt.figure(figsize=(10,6))
    plt.pie(count, labels=Teams, autopct='%.2f%%', shadow=True)
    st.pyplot(fig)

    #stats medie divise per squadra
    #st.write con spiegazione e dire che sono stats per game
    stat = st.selectbox('Choose a Stat', LeB_C_PG_RS.columns.drop(['Season', 'Tm', 'Pos']).tolist(), key=3)
    a = LeB_C_PG_RS[stat].head(7).mean() #medie ai cavs (prima esperienza)
    b = LeB_C_PG_RS[stat].iloc[7:11].mean() #medie agli heat
    c = LeB_C_PG_RS[stat].iloc[11:15].mean() #medie ai cavs (seconda esperienza)
    d = LeB_C_PG_RS[stat].tail(4).mean() #medie ai lakers
    [a, b, c, d]
    #da mettere a posto

    #st.write con spiegazione del pie plot sui ruoli 
    Pos = list(LeB_C_PG_RS['Pos'])
    PG = Pos.count('PG')
    PF = Pos.count('PF')
    SG = Pos.count('SG')
    SF = Pos.count('SF')

    Positions = ['PG', 'PF', 'SG', 'SF']
    count = [PG, PF, SG, SF]
    fig = plt.figure(figsize=(10,6))
    plt.pie(count, labels=Positions, autopct='%.2f%%', shadow=True)
    st.pyplot(fig)

    #stats medie divise per ruolo
    #st.write con spiegazione e dire che sono stats per game
    stat = st.selectbox('Choose a Stat', LeB_C_PG_RS.columns.drop(['Season', 'Tm', 'Pos']).tolist(), key=3)
    a = LeB_C_PG_RS[stat].head(7).mean() #medie ai cavs (prima esperienza)
    b = LeB_C_PG_RS[stat].iloc[7:11].mean() #medie agli heat
    c = LeB_C_PG_RS[stat].iloc[11:15].mean() #medie ai cavs (seconda esperienza)
    d = LeB_C_PG_RS[stat].tail(4).mean() #medie ai lakers
    [a, b, c, d]
    #da mettere a posto
    
    #show plot misto con più di una stat (le percentuali tutte insieme o altro)
    for x in LeB_C_PG_RS.columns.drop(['Season', 'Tm', 'Pos']).tolist():
      Season = list(LeB_C_Tot_RS['Season'])
      Peppino = list(LeB_C_Tot_RS[x])
      fig = plt.figure(figsize=(10, 6))
      plt.plot(Season, Peppino, '-o')
      plt.title('LeBron ' + x + ' averages in every season')
      plt.xlabel('Seasons')
      plt.ylabel(x)
      plt.xticks(rotation=45)
      st.pyplot(fig)

# show other features, other stats analyzed
# plots su Teams and Pos + stats on the same columns
# plot misti con più stats (FG%, 3P%, 2P%, FT%) etc.
# plot distribuzioni (hist, di tutte le variabili)

if sec == 'Predictive model for LeBron James':
    st.header('Predictive model for LeBron James')

# mettere i modelli che ci sono sull'ipynb generici con la possibilità di scegliere la statistica da predire con anche la possibilità di mettere dei valori in input ecc
# vedere che modelli mettere, capire che altre analisi vanno fatte e se sono abbastanza

if sec == 'Season 2020/2021 exploration and analysis':
    st.header('Season 2020/2021 exploration and analysis')

# show other features, other stats analysed
# plots su Teams and Pos + stats on the same columns
# plot misti con più stats (FG%, 3P%, 2P%, FT%) etc.
# plot distribuzioni (hist, di tutte le variabili)

if sec == 'Predictive model for Season 2020/2021':
    st.header('Predictive model for Season 2020/2021')

# mettere i modelli che ci sono sull'ipynb generici con la possibilità di scegliere la statistica da predire con anche la possibilità di mettere dei valori in input ecc
# vedere che modelli mettere, capire che altre analisi vanno fatte e se sono abbastanza