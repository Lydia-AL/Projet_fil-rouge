import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title = "Graphs page",layout="wide")

@st.cache
def load_data():
    file_path = './data/raw/data.csv'
    df = pd.read_csv(file_path)
    return df

df = load_data()

st.title("Exploration des variables du dataset")

st.header("Aperçu du Dataset")
st.write(df.head())

# Graphique 1 : Matrice de corrélation sur les variables quantitatives
st.header("Heatmap on quantitative data")
df_quantitative = df[['Mt', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)', 'Fuel consumption ']]
fig1, ax1 = plt.subplots()
sns.heatmap(df_quantitative.corr(), annot=True, cmap='RdBu_r', center=0, ax=ax2)
st.pyplot(fig2)

# Graphique 2 : Scatterplot Ewltp (g/km) vs m (kg)
st.header("Vehicle mass (m (kg)) vs CO2 emissions (Ewltp (g/km))")
fig2, ax2 = plt.subplots()
sns.scatterplot(df.sort_values(by='m (kg)', ascending=True), x='Ewltp (g/km)', y='m (kg)', ax=ax1)
st.pyplot(fig1)

# Graphique 3 : Scatterplot Ewltp (g/km) vs Fuel consumption
st.header("Fuel consumption vs CO2 emissions (Ewltp (g/km))")
fig3, ax3 = plt.subplots()
sns.scatterplot(df.sort_values(by='Fuel consumption ', ascending=True), x='Ewltp (g/km)', y='Fuel consumption ', ax=ax3)
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
    st.header("Nombre de voitures par mois")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='mois_nom', data=df, order=["January", "February", "March", "April", "May", "June", 
                                                "July", "August", "September", "October", "November", "December"], ax=ax)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de voitures")
    ax.set_title("Nombre de voitures enregistrées par mois")
    st.pyplot(fig)

else:
    st.write("La colonne 'Date of registration' est manquante dans le dataset.")