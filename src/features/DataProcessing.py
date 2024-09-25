import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def loaddata(filename):
    df = pd.read_csv(filename, index_col=0)
    return(df)
    
def PrepareDataForRegression(df,excludeFC:False):
    # df source dataFame
    # excludeFC bool to decide if column Fuel consumption should be excluded of the result
    df.drop_duplicates(inplace=True)

    df_Categorielle = df.drop(['Mt', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'ec (cm3)', 'ep (KW)',
                            'Erwltp (g/km)'], axis=1)
    if (excludeFC):
        df_quantitative = df[['Mt', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'ec (cm3)', 'ep (KW)',
                              'Erwltp (g/km)']]
    else:
        df_quantitative = df[['Mt', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'ec (cm3)', 'ep (KW)',
                              'Erwltp (g/km)', 'Fuel consumption ']]

    df_quantitative = df_quantitative.fillna(df_quantitative.std()) 

    df_Categorielle = df_Categorielle.join(pd.get_dummies(df_Categorielle[['Country','Ct','Ft','Cr']]))
    df_Categorielle = df_Categorielle.drop(['Country','Ct','Ft','Cr'],axis=1)

    le = LabelEncoder()
    df_Categorielle['Mk'] = le.fit_transform(df_Categorielle['Mk'])
    df_Categorielle['Mh'] = le.fit_transform(df_Categorielle['Mh'])
    df_Categorielle['IT'] = le.fit_transform(df_Categorielle['IT'])

    df_Final = df_Categorielle.join(df_quantitative)

    data = df_Final.drop('Ewltp (g/km)',axis=1)
    target = df_Final['Ewltp (g/km)']

    return data,target

def PrepareDataForClassification(df,excludeFC:False):
    # df source dataFame
    # excludeFC bool to decide if column Fuel consumption should be excluded of the result
    data,target = PrepareDataForRegression(df,excludeFC)


    def new_target(x):
        res = 0
        if(x <= 100) : res = 1
        elif((x > 100 ) & (x <= 120)):res = 2
        elif((x > 120) & (x <= 140)): res = 3
        elif((x > 140) & (x <= 160)): res = 4
        elif((x > 160) & (x <= 200)): res = 5
        elif((x > 200) & (x <= 250)): res = 6
        elif((x > 250)): res = 7
        return res

    target = target.apply(new_target)
    
    return data,target

def gettesttraindata(data,target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=123)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled,X_test_scaled,y_train,y_test

