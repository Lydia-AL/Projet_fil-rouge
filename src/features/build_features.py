import numpy as np
import pandas as pd
import os

print(os.listdir('../dec23_cds_co2'))

df = pd.read_csv("../dec23_cds_co2/data/raw/data.csv", index_col=0)

# Supression des colonnes à 100% de NA
for column in df.columns[df.isnull().any()]:
    if df[column].isnull().sum()*100.0/df.shape[0] == 100:
        df.drop(column,axis=1, inplace=True)

# Suppression de :     electrique   autre pays que FR et DE   autre que petrol, diesel
df = df.drop(['z (Wh/km)','Electric range (km)'], axis = 1)
df = df[df['Country'].isin(['FR','DE']) ]
df = df[df['Ft'].isin(['petrol','diesel','petrol/electric','lpg','e85','diesel/electric','ng'])]

# suppression des colonnes
# year, r, man, date of registration, status,T,va,ve,vfn,country,Mp, m(kg), Fm, At2
# Tan, Cn

df = df.drop(['VFN', 'Tan', 'T', 'Va', 'Ve', 'Cn', 'Fm', 'Man','Mp',
              'Status','Date of registration','m (kg)','Enedc (g/km)','At2 (mm)','year','r'], axis = 1)

# pour les variables numériques on supprime les lignes si le taux de valeur manquantes < 1%

cols = []

for column in df.select_dtypes(['int','float']).columns[df.select_dtypes(['int','float']).isnull().any()]:
    if df[column].isnull().sum()*100.0/df.shape[0] < 1:
        cols.append(column)
        
df.dropna(subset=cols, inplace=True)    

# Sauvegarde du fichier FinalData.csv
df.to_csv(path_or_buf='../dec23_cds_co2/data/processed/FinalData.csv')
