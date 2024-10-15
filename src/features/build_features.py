import numpy as np
import pandas as pd
import os

df = pd.read_csv("../dec23_cds_co2/data/raw/data.csv")

# Supression des colonnes à 100% de NA
for column in df.columns[df.isnull().any()]:
    if df[column].isnull().sum()*100.0/df.shape[0] == 100:
        df.drop(column,axis=1, inplace=True)

# Suppression de :     electrique   autre pays que FR et DE   autre que petrol, diesel
print(df.columns)
df = df.drop(['z (Wh/km)','Electric range (km)','Cr','Mk','ID'], axis = 1)
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
df.drop_duplicates(inplace=True)

# Sauvegarde du fichier FinalData.csv
print(df.columns)
df.to_csv(path_or_buf='../dec23_cds_co2/data/processed/FinalData.csv', index=False)
