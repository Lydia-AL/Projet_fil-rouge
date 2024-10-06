import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import joblib
import seaborn as sns
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
        gridsearchname = '/models/gb_GCV_model.sav'
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

res, model, GridSearchModel = loadmodel(model)
data , target = LoadData()
X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

if res : 
    st.subheader('Feature importance',divider=True)
    plot = GetFeatureImportance(model,data)
    st.pyplot(plot.get_figure())

    st.subheader('Prédiction VS réel',divider=True)
    plotpred = GetConfusionMatrices(model,X_train_scaled,X_test_scaled,y_train,y_test)
    st.pyplot(plotpred.get_figure())

    st.subheader('Grid search scores',divider=True)
    dfscore = GetGridSearchScore(GridSearchModel)
    st.dataframe(dfscore[["params", "rank_test_score", "mean_test_score", "std_test_score"]])

