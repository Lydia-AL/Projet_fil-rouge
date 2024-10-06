import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import joblib
import seaborn as sns
from visualization.RegressionVisualisation import GetFeatureImportance, GetTestPredictions, GetGridSearchScore
from features.DataProcessing import loaddata, PrepareDataForRegression,gettesttraindata
import numpy as np 

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
    elif modelselected == 'Logistic Regression' : 
        modelname = '/models/lr_model.sav'
        gridsearchname = '/models/lr_GCV_model.sav'
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

model = st.sidebar.selectbox("Choix du model",('Random Forest','Gradient Booting','Logistic Regression'))
st.write("model sélectionné:",model)

res, model, GridSearchModel = loadmodel(model)
data , target = LoadData()
X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

if res : 
    st.subheader('Feature importance',divider=True)
    plot = GetFeatureImportance(model,data)
    st.pyplot(plot.get_figure())

    st.subheader('Prédiction VS réel',divider=True)
    plotpred = GetTestPredictions(model,X_train_scaled,X_test_scaled,y_train,y_test)
    st.pyplot(plotpred.get_figure())

    st.subheader('Grid search scores',divider=True)
    dfscore = GetGridSearchScore(GridSearchModel)
    st.dataframe(dfscore[["params", "rank_test_score", "mean_test_score", "std_test_score"]])
