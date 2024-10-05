import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from src.features.DataProcessing import loaddata, gettesttraindata,PrepareDataForRegression, PrepareDataForClassification
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
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
    rf = RandomForestRegressor(criterion = 'poisson', max_depth= 20, n_estimators=150)
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
 
    #create LogisticRegression
    print('train LogisticRegression')
    clf = LogisticRegression(C=1,solver='lbfgs')
    clf.fit(X_train_scaled,y_train)

    #Save Model
    ModelName = Path('models/lr_model.sav')
    joblib.dump(rf, ModelName)

def TrainClassification():
    #load data
    df = loaddata()
    data, target = PrepareDataForClassification(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    # create model DecisionTreeClassifier
    dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=4,random_state=123)
    dt_clf.fit(X_train_scaled,y_train)
    
    #Save Model
    joblib.dump(dt_clf, "dt_model.sav")

    #create model GradientBoostingClassifier
    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X_train_scaled,y_train)
    
    #Save Model
    joblib.dump(gb_clf, "gb_model.sav")

    #create model KNeighborsClassifier
    knn_clf = KNeighborsClassifier(n_neighbors=7,p=2, metric="minkowski")
    knn_clf.fit(X_train_scaled,y_train)

    #Save Model
    joblib.dump(knn_clf, "knn_model.sav")

def TrainDeepLearningClassification():
    print('training deep learning')
    #load data
    DataFile = Path('data/processed/FinalData.csv')
    df = loaddata(DataFile)    
    data, target = PrepareDataForClassification(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    num_var_exp = X_train_scaled.shape[1]

    model = Sequential()

    # Part 1 : Input Layer
    model.add(Dense(32, activation='relu'))

    # Part 2 : Hidden Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))

    # Part 3 : Output Layer
    model.add(Dense(num_var_exp, activation='softmax'))

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
    model.add(Dense(32, activation='linear'))

    # Loss function 
    model.compile(optimizer="adam", loss='mean_absolute_error', metrics=['mean_absolute_error'])

    # Learning
    history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled ,y_test), verbose=1, batch_size=64, epochs=30)

    #Save Model
    ModelName = Path('models/DeepReg_model.sav')
    joblib.dump(model, ModelName)
    historyName = Path('models/DeepRegHistory_model.sav')
    joblib.dump(history, historyName)

TrainRegression()
#TrainDeepLearningClassification()
#TrainDeepLearningRegression()