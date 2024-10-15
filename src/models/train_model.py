import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from src.features.DataProcessing import loaddata, gettesttraindata,PrepareDataForRegression, PrepareDataForClassification
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import joblib
from pathlib import Path

def TrainRegression():
    #load data
    DataFile = Path('data/processed/FinalData.csv')
    df = loaddata(DataFile)
    data, target = PrepareDataForRegression(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    #create RandomForestRegressor
    print('train RandomForestRegressor')
    rf = RandomForestRegressor(criterion = 'friedman_mse', max_depth= 20, n_estimators=100)
    rf.fit(X_train_scaled,y_train)

    #Save Model
    ModelName = Path('models/rf_model.sav')
    joblib.dump(rf, ModelName)
                

    #create GradientBoostingRegressor
    print('train GradientBoostingRegressor')
    gbr = GradientBoostingRegressor(learning_rate = 0.5, n_estimators = 500)
    gbr.fit(X_train_scaled,y_train)

    #Save Model
    ModelName = Path('models/gbr_model.sav')
    joblib.dump(rf, ModelName)
 
    #create RidgeRegression
    print('train RidgeRegression')
    ridge = Ridge(alpha = 0.02, solver='sparse_cg')
    ridge.fit(X_train_scaled,y_train)

    #Save Model
    ModelName = Path('models/rdg_model.sav')
    joblib.dump(ridge, ModelName)

def TrainClassification(modeltype):
    #load data
    DataFile = Path('data/processed/FinalData.csv')
    df = loaddata(DataFile)
    data, target = PrepareDataForClassification(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    # create model DecisionTreeClassifier
    if (modeltype == 'DTC') or (modeltype == 'ALL'):
        dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_split = 10,random_state=123)
        dt_clf.fit(X_train_scaled,y_train)
        
        #Save Model
        ModelName = Path('models/dtc_model.sav')
        joblib.dump(dt_clf, ModelName)

    #create model GradientBoostingClassifier
    if (modeltype == 'GBC') or (modeltype == 'ALL'):
        gb_clf = GradientBoostingClassifier( learning_rate = 0.05, max_depth = 5, n_estimators = 200)
        gb_clf.fit(X_train_scaled,y_train)
        
        #Save Model
        ModelName = Path('models/gb_model.sav')
        joblib.dump(gb_clf, ModelName)

    #create model RandomForestClassifier
    if (modeltype == 'RFC') or (modeltype == 'ALL'):
        rfc_clf = RandomForestClassifier(criterion = 'gini', max_depth = 20, min_samples_split = 5, n_estimators = 100)
        rfc_clf.fit(X_train_scaled,y_train)

        #Save Model
        ModelName = Path('models/rfc_model.sav')
        joblib.dump(rfc_clf, ModelName)

def TrainDeepLearningClassification():
    print('training deep learning')
    #load data
    DataFile = Path('data/processed/FinalData.csv')
    df = loaddata(DataFile)    
    data, target = PrepareDataForClassification(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    #num_var_exp = X_train_scaled.shape[1]

    model = Sequential()

    # Part 1 : Input Layer
    model.add(Dense(32, activation='relu'))

    # Part 2 : Hidden Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))

    # Part 3 : Output Layer
    model.add(Dense(8, activation='softmax'))

    # Loss function 
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['acc'])

    # Learning
    history = model.fit(X_train_scaled, y_train.values, validation_data=(X_test_scaled ,y_test.values), verbose=1, batch_size=64, epochs=30)
    
    #Save Model
    ModelName = Path('models/DeepClass_model.sav')
    joblib.dump(model, ModelName)
    
    historyName = Path('models/DeepClassHistory_model.sav')
    joblib.dump(history, historyName)

def TrainDeepLearningRegression():
    #load data
    DataFile = Path('data/processed/FinalData.csv')
    df = loaddata(DataFile)
    data, target = PrepareDataForRegression(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    num_var_exp = X_train_scaled.shape[1]

    model = Sequential()

    # Part 1 : Input Layer
    model.add(Dense(32, activation='relu'))

    # Part 2 : Hidden Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))

    # Part 3 : Output Layer
    model.add(Dense(1, activation='linear'))

    # Loss function 
    model.compile(optimizer="adam", loss='mean_absolute_error', metrics=['mean_absolute_error'])

    # Learning
    history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled ,y_test), verbose=1, batch_size=64, epochs=30)

    #Save Model
    ModelName = Path('models/DeepReg_model.sav')
    joblib.dump(model, ModelName)
    historyName = Path('models/DeepRegHistory_model.sav')
    joblib.dump(history, historyName)

TrainClassification('ALL')
#TrainRegression()
#TrainDeepLearningClassification()
#TrainDeepLearningRegression()