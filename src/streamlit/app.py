import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st

#st.set_page_config(page_title = "main page",layout="wide")



main = st.Page("pages/main.py", title='main page')
graph = st.Page("pages/graphs.py", title='graph page')
classification = st.Page("pages/classifications.py", title='classification page')
regression = st.Page("pages/regressions.py", title='regression page')
deep = st.Page("pages/deeplearning.py", title='deep Learning page')

pg = st.navigation([main,graph,classification,regression,deep])

pg.run()

