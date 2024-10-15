import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import joblib
import seaborn as sns
from visualization.RegressionVisualisation import GetFeatureImportance, GetTestPredictions, GetGridSearchScore,GetFeatureCoef
from features.DataProcessing import loaddata, PrepareDataForRegression,gettesttraindata
import numpy as np 
from sklearn.metrics import mean_squared_error

from pathlib import Path

st.set_page_config(page_title = "Regression page",layout="wide")

@st.cache_data
def loadmodel(modelselected):
    if modelselected == 'Random Forest' : 
        modelname = '/models/rf_model.sav'
        gridsearchname = '/models/rf_GCV_model.sav'
    elif modelselected == 'Gradient Booting' : 
        modelname = '/models/gbr_model.sav'
        gridsearchname = '/models/gbr_GCV_model.sav'
    elif modelselected == 'Ridge Regression' : 
        modelname = '/models/rdg_model.sav'
        gridsearchname = '/models/rdg_GCV_model.sav'
    else :
        modelname = ''

    if modelname != '':
        drive = os.path.abspath('../..')
        path = Path(os.path.abspath(drive+modelname))
        loaded_model = joblib.load(path)
        path = Path(os.path.abspath(drive+gridsearchname))
        loaded_gridsearchmodel =  joblib.load(path)
        return True, loaded_model, loaded_gridsearchmodel
    else:
        return False

@st.cache_data
def LoadData():
    #load data
    drive = os.path.abspath('../..')
    DataFile = Path(os.path.abspath(drive+'/data/processed/FinalData.csv'))
    df = loaddata(DataFile)
    data, target = PrepareDataForRegression(df,False)
    return data, target

@st.cache_data
def GetPrediction(model, X_train, X_test):
    res_train =  model.predict(X_train)
    res_test = model.predict(X_test)
    return res_train, res_test

@st.cache_data
def Getmean_squared_errorTest(_model, X_test, y_test):
    predtest = _model.predict(X_test)
    return mean_squared_error(y_test,predtest)

#@st.cache_data
def Getmean_squared_errorTrain(_model, X_train, y_train):
    predtrain = _model.predict(X_train)
    return mean_squared_error(y_train,predtrain)

modeltype = st.sidebar.selectbox("Choix du model",('Random Forest','Gradient Booting','Ridge Regression'))
st.write("model sélectionné:",modeltype)

res, loaded_model, GridSearchModel = loadmodel(modeltype)
data , target = LoadData()
X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

if res : 
    st.subheader('Grid search scores',divider=True)
    dfscore = GetGridSearchScore(GridSearchModel)
    st.dataframe(dfscore[["params", "rank_test_score", "mean_test_score", "std_test_score"]])

    st.subheader('Feature importance',divider=True)
    if modeltype == 'Ridge Regression':
        plot = GetFeatureCoef(loaded_model,data)
        st.pyplot(plot.get_figure())
    else :
        plot = GetFeatureImportance(loaded_model,data)
        st.pyplot(plot.get_figure())
    

    st.subheader('Prédiction VS réel',divider=True)
    plotpred = GetTestPredictions(loaded_model,X_train_scaled,X_test_scaled,y_train,y_test)
    st.pyplot(plotpred.get_figure())

    st.subheader('test overfitting',divider=True)
    predtrain = loaded_model.predict(X_train_scaled)
    predtest = loaded_model.predict(X_test_scaled)
     

    st.write('erreur quadratique moyenne de prédiction des données train : ' ,mean_squared_error(predtrain, y_train))
    st.write('erreur quadratique moyenne de prédiction des données test : ',mean_squared_error(predtest, y_test))

    #st.write('erreur quadratique moyenne de prédiction des données train : ' ,Getmean_squared_errorTest(loaded_model, X_train_scaled, y_train))
    #st.write('erreur quadratique moyenne de prédiction des données test : ',Getmean_squared_errorTest(loaded_model, X_test_scaled, y_test))