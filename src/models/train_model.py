import numpy as np
import pandas as pd
from src.features.DataProcessing import loaddata, gettesttraindata,PrepareDataForRegression, PrepareDataForClassification
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

def TrainRegression():
    #load data
    df = loaddata()
    data, target = PrepareDataForRegression(df,False)
    X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

    #create RandomForestRegressor
    rf = RandomForestRegressor(criterion = 'poisson', max_depth= 20, n_estimators=1503)
    rf.fit(X_train_scaled,y_train)

    #Save Model
    joblib.dump(rf, "rf_model.sav")
                

    #create GradientBoostingRegressor
    gbr = GradientBoostingRegressor(learning_rate = 0.5, n_estimators = 500)
    gbr.fit(X_train_scaled,y_train)

    #Save Model
    joblib.dump(gbr, "gbr_model.sav")

    #create LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled,y_train)

    #Save Model
    joblib.dump(clf, "lr_model.sav")

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