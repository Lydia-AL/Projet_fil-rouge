import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from src.features.DataProcessing import loaddata, gettesttraindata,PrepareDataForRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

def GridCVRegression(modeltype):
    #load data
    DataFile = Path('data/processed/FinalData.csv')
    df = loaddata(DataFile)
    data, target = PrepareDataForRegression(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    if (modeltype == 'RF') or (modeltype == 'ALL'):
        param_grid_RF = {
            'n_estimators': [50,100,150],
            'criterion' : ['squared_error','friedman_mse','poisson'],
            'max_depth': [10, 20, 30],
            }
        
        grid_clf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid_RF, n_jobs=-1,
                                cv=3, verbose=1,refit= True)
        grid_clf.fit(X_train_scaled,y_train)
        
        # Save the trained model, StandardScaler, and LabelEncoder for later use
        ModelName = Path('models/rf_GCV_model.sav')
        joblib.dump(grid_clf, ModelName)

    if (modeltype == 'GBR') or (modeltype == 'ALL'):
        param_gridGBR = {
            'learning_rate' : [0.1,0.2,0.3,0.4,0.5],
            'n_estimators':  [100, 200, 500],
            }
        grid_GBC = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_gridGBR, n_jobs=-1,
                                cv=2, verbose=1,refit= True)
        grid_GBC.fit(X_train_scaled,y_train)

        # Save the trained model, StandardScaler, and LabelEncoder for later use
        ModelName = Path('models/gbr_GCV_model.sav')
        joblib.dump(grid_GBC, ModelName)

    if (modeltype == 'LR') or (modeltype == 'ALL'):

        params_lr = {'solver': [ 'sag', 'saga', 'lbfgs' ],
                'C':  [0.1, 1, 10, 100]}

        grid_LR = GridSearchCV( LogisticRegression(max_iter=300), param_grid=params_lr, n_jobs=-1,
                            verbose=1,refit= True)
        grid_LR.fit(X_train_scaled,y_train)

        # Save the trained model, StandardScaler, and LabelEncoder for later use
        ModelName = Path('models/lr_GCV_model.sav')
        joblib.dump(grid_LR, ModelName)

    if (modeltype == 'RDG') or (modeltype == 'ALL'):

        param_RDG = {'alpha' : [0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
              'solver' : ['auto', 'svd','cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

        grid_RDG = GridSearchCV(Ridge(), param_grid=param_RDG, n_jobs=-1,
                            verbose=1,refit= True)
        grid_RDG.fit(X_train_scaled,y_train)

        # Save the trained model, StandardScaler, and LabelEncoder for later use
        ModelName = Path('models/rdg_GCV_model.sav')
        joblib.dump(grid_RDG, ModelName)


GridCVRegression('RDG')