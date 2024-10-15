import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from src.features.DataProcessing import loaddata, gettesttraindata,PrepareDataForClassification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

def GridCVClassification(modeltype):
    #load data
    DataFile = Path('data/processed/FinalData.csv')
    df = loaddata(DataFile)
    data, target = PrepareDataForClassification(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    if (modeltype == 'DTC') or (modeltype == 'ALL'):
        param_grid_DTC = {
            'criterion' : ['gini', 'entropy'],
            'max_depth':  [None, 2, 4, 6, 8, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 20],
            }
        
        grid_dtc = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid_DTC, n_jobs=-1,
                                cv=3, verbose=1,refit= True)
        grid_dtc.fit(X_train_scaled,y_train)
        
        # Save the trained model, StandardScaler, and LabelEncoder for later use
        ModelName = Path('models/dtc_GCV_model.sav')
        joblib.dump(grid_dtc, ModelName)

    if (modeltype == 'GBC') or (modeltype == 'ALL'):
        param_gridGBC = {
            'learning_rate': [0.01, 0.05],     
            'n_estimators': [100, 200],        
            'max_depth': [3, 5],
            }
        grid_GBC = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_gridGBC, n_jobs=-1,
                                cv=2, verbose=1,refit= True)
        grid_GBC.fit(X_train_scaled,y_train)

        # Save the trained model, StandardScaler, and LabelEncoder for later use
        ModelName = Path('models/gbc_GCV_model.sav')
        joblib.dump(grid_GBC, ModelName)

    if (modeltype == 'RFC') or (modeltype == 'ALL'):
        param_gridRFC = {
            'n_estimators': [50, 100, 200],  
            'criterion': ['gini', 'entropy'],  
            'max_depth': [None, 10, 20, 30],  
            'min_samples_split': [2, 5, 10],  
            }
        grid_RFC = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_gridRFC, n_jobs=-1,
                            cv=2, verbose=1,refit= True)
        grid_RFC.fit(X_train_scaled,y_train)

        # Save the trained model, StandardScaler, and LabelEncoder for later use
        ModelName = Path('models/rfc_GCV_model.sav')
        joblib.dump(grid_RFC, ModelName)

GridCVClassification('ALL')
