import numpy as np
import pandas as pd
from src.features.DataProcessing import loaddata, gettesttraindata,PrepareDataForRegression, PrepareDataForClassification
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import joblib


def GridCVRegression():
    #load data
    df = loaddata('FinalData.csv')
    data, target = PrepareDataForRegression(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    param_grid_RF = {
        'n_estimators': [50,100,150],
        'criterion' : ['squared_error','friedman_mse','poisson'],
        'max_depth': [10, 20, 30],
    }
    grid_clf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid_RF, n_jobs=-1,
                            cv=3, verbose=1,refit= True)
    grille = grid_clf.fit(X_train_scaled,y_train)
    
    # Save the trained model, StandardScaler, and LabelEncoder for later use
    joblib.dump(grid_clf, 'rf_GCV_model.sav')

    param_gridGBR = {
    'learning_rate' : [0.1,0.2,0.3,0.4,0.5],
    'n_estimators':  [100, 200, 500],
    }
    grid_GBC = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_gridGBR, n_jobs=-1,
                            cv=2, verbose=1,refit= True)
    grille = grid_GBC.fit(X_train_scaled,y_train)

    print('le meilleur paramètre de GradientBoostingClassifier est :',grid_GBC.best_params_)
    print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']])

    params_lr = {'solver': ['newton-cg', 'sag', 'saga', 'lbfgs' ],
             'penalty' : ['l2', 'None'],
             'C':  [0.1, 1, 10, 100]}

    grid_LR = GridSearchCV( LogisticRegression(), param_grid=params_lr, n_jobs=-1,
                        cv=2, verbose=1,refit= True)
    grille = grid_LR.fit(X_train_scaled,y_train)


    print('le meilleur paramètre de LogisticRegression est :',grid_LR.best_params_)
    print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']])