import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import streamlit as st
import joblib
import seaborn as sns
from sklearn.metrics import classification_report
from visualization.ClassificationVisualisation import GetFeatureImportance, GetConfusionMatrices, GetGridSearchScore
from features.DataProcessing import loaddata, PrepareDataForClassification, gettesttraindata

from pathlib import Path

st.set_page_config(page_title = "Classification page",layout="wide")

@st.cache_data
def loadmodel(modelselected):
    if modelselected == 'Decision Tree Classifier' : 
        modelname = '/models/dtc_model.sav'
        gridsearchname = '/models/dtc_GCV_model.sav'
    elif modelselected == 'Random Forest Classifier' : 
        modelname = '/models/rfc_model.sav'
        gridsearchname = '/models/rfc_GCV_model.sav'
    elif modelselected == 'Gradient Boosting' : 
        modelname = '/models/gb_model.sav'
        gridsearchname = '/models/gbc_GCV_model.sav'
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
    data, target = PrepareDataForClassification(df,False)
    return data, target


model = st.sidebar.selectbox("Choix du model",('Decision Tree Classifier','Random Forest Classifier','Gradient Boosting'))
st.write("model sélectionné:",model)

res, loadedmodel, GridSearchModel = loadmodel(model)
data , target = LoadData()
X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

if res : 
    st.subheader('Grid search scores',divider=True)
    dfscore = GetGridSearchScore(GridSearchModel)
    st.dataframe(dfscore[["params", "rank_test_score", "mean_test_score", "std_test_score"]])

    st.subheader('Feature importance',divider=True)
    plot = GetFeatureImportance(loadedmodel,data)
    st.pyplot(plot.get_figure())

    st.subheader('Prédiction VS réel',divider=True)
    plotpred = GetConfusionMatrices(loadedmodel,X_train_scaled,X_test_scaled,y_train,y_test)
    st.pyplot(plotpred.figure_)
    
    y_train_pred = loadedmodel.predict(X_train_scaled)
    y_test_pred = loadedmodel.predict(X_test_scaled)

    target_names = ['1','2','3','4','5','6','7']
    st.subheader("Classification Report Train:",divider=True)
    st.dataframe(pd.DataFrame(classification_report(y_train, y_train_pred, target_names=target_names, output_dict=True)).transpose())
    st.subheader("Classification Report test:",divider=True)
    st.dataframe(pd.DataFrame(classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True)).transpose())


