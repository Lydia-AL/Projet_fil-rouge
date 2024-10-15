import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.features.DataProcessing import loaddata

st.set_page_config(page_title = "Preprocessing",layout="wide")

@st.cache_data
def load_data():
    #load data
    drive = os.path.abspath('../..')
    DataFile = Path(os.path.abspath(drive+'/data/raw/data.csv'))
    df = loaddata(DataFile)
    return df

@st.cache_data
def getgraph2data(df) :
    res = df[['Ewltp (g/km)','m (kg)']]
    return res.sort_values(by='m (kg)', ascending=True)

@st.cache_data
def getgraph3data(df) :
    res = df[['Ewltp (g/km)','Fuel consumption ']]
    return res.sort_values(by='Fuel consumption ', ascending=True)

@st.cache_data
def load_data_descrition():
    drive = os.path.abspath('../..')
    DataFile = Path(os.path.abspath(drive+'/data/Dataanalysis.csv'))
    df = pd.read_csv(DataFile, index_col=0, sep=';')
    return(df)

df = load_data()
descdata = load_data_descrition()

custom_css = """
<style>
    .dataframe th, .dataframe td {
        white-space: pre-wrap;
        vertical-align: top;
        font-size: 20px;
    }
    .dataframe .blank, .dataframe .nan {
        color: #ccc;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.title("Exploration des variables du dataset")


st.subheader('Aperçu du Dataset',divider=True)
st.dataframe(df.head())

st.subheader('description des colonnes du Dataset',divider=True)
st.dataframe(descdata)


# Graphique 1 : Matrice de corrélation sur les variables quantitatives
st.header("Heatmap on quantitative data")
df_quantitative = df.select_dtypes(['int','float'])
#df_quantitative = df[['Mt', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)', 'Fuel consumption ']]
fig1, ax1 = plt.subplots(figsize = (10,8))
sns.heatmap(df_quantitative.corr(), annot=True, cmap='RdBu_r', center=0, ax=ax1)
st.pyplot(fig1)

# Graphique 2 : Scatterplot Ewltp (g/km) vs m (kg)
st.header("Vehicle mass (m (kg)) vs CO2 emissions (Ewltp (g/km))")
fig2, ax2 = plt.subplots()
data2 = getgraph2data(df)
sns.scatterplot(data2, x='Ewltp (g/km)', y='m (kg)', ax=ax2)
#sns.scatterplot(df.sort_values(by='m (kg)', ascending=True), x='Ewltp (g/km)', y='m (kg)', ax=ax2)
st.pyplot(fig2)

# Graphique 3 : Scatterplot Ewltp (g/km) vs Fuel consumption
st.header("Fuel consumption vs CO2 emissions (Ewltp (g/km))")
fig3, ax3 = plt.subplots()
data3 = getgraph3data(df)
sns.scatterplot(data3, x='Ewltp (g/km)', y='Fuel consumption ', ax=ax3)
#sns.scatterplot(df.sort_values(by='Fuel consumption ', ascending=True), x='Ewltp (g/km)', y='Fuel consumption ', ax=ax3)
st.pyplot(fig3)

# Graphique 4 : Scatterplot Ewltp (g/km) vs Enedc (g/km)
st.header("CO2 emissions according to the WLTP standard (Ewltp (g/km)) vs CO2 emissions according to the NEDC standard (Enedc (g/km))")
fig4, ax4 = plt.subplots()
sns.scatterplot(x='Ewltp (g/km)', y='Enedc (g/km)', data=df, ax=ax4)
st.pyplot(fig4)

# Graphique 5 : Visualisation par mois pour Ewltp
st.header("Evolution par mois des émissions Ewltp (g/km)")
if 'Date of registration' in df.columns:
    df['datetime'] = pd.to_datetime(df['Date of registration'])
    groupby_mois_ewltp = df.groupby(pd.Grouper(key='datetime', freq='M')).agg({'Ewltp (g/km)': 'mean'})
    fig5, ax5 = plt.subplots(figsize=(15, 5))
    groupby_mois_ewltp.plot(style='o-', ax=ax5)
    st.pyplot(fig5)
else:
    st.write("La colonne 'Date of registration' est manquante dans le dataset.")

# Graphique 6 : Nombre de voitures immatriculées par mois
st.header("Number of cars registered per month")
if 'Date of registration' in df.columns:
    df['datetime'] = pd.to_datetime(df['Date of registration'])
    df['mois'] = df['datetime'].dt.month  
    df['mois_nom'] = df['datetime'].dt.strftime('%B')
    #st.header("Nombre de voitures par mois")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='mois_nom', data=df, order=["January", "February", "March", "April", "May", "June", 
                                                "July", "August", "September", "October", "November", "December"], ax=ax)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de voitures")
    ax.set_title("Nombre de voitures enregistrées par mois")
    st.pyplot(fig)

else:
    st.write("La colonne 'Date of registration' est manquante dans le dataset.")