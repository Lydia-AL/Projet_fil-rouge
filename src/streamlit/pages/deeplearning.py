import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import joblib
import seaborn as sns
from visualization.DeepVisualisation import GetLossByEpoch,GetAccByEpoch, GetMaeByEpoch
from features.DataProcessing import loaddata, PrepareDataForRegression,gettesttraindata

from pathlib import Path


st.set_page_config(page_title = "Deep Learning page",layout="wide")

@st.cache_data
def loadmodel(modelselected):
    if modelselected == 'Classification' : 
        modelname = '/models/DeepClass_model.sav'
        historyname = '/models/DeepClassHistory_model.sav'
    elif modelselected == 'Regression' : 
        modelname = '/models/DeepReg_model.sav'
        historyname = '/models/DeepRegHistory_model.sav'
    else :
        modelname = ''

    if modelname != '':
        drive = os.path.abspath('../..')
        path = Path(os.path.abspath(drive+modelname))
        loaded_model = joblib.load(path)
        path = Path(os.path.abspath(drive+historyname))
        loaded_history =  joblib.load(path)
        return True, loaded_model, loaded_history
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

model = st.sidebar.selectbox("Choix du model",('Classification','Regression'))
st.write("model sélectionné:",model)

res, loadedmodel, history = loadmodel(model)
data , target = LoadData()
X_train_scaled,X_test_scaled,y_train,y_test = gettesttraindata(data,target)

if res : 
    st.subheader('Model loss by epoch',divider=True)
    plot = GetLossByEpoch(history)
    st.pyplot(plot.get_figure())
    st.write(model)
    if model == 'Classification' : 
        st.subheader('Model acc by epoch',divider=True)
        plotpred = GetAccByEpoch(history)
        st.pyplot(plotpred.get_figure())

    if model == 'Regression' : 
        st.subheader('Model val_mean_absolute_error by epoch',divider=True)
        plotpred = GetMaeByEpoch(history)
        st.pyplot(plotpred.get_figure())
