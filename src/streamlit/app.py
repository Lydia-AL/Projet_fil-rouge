import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st

#st.set_page_config(page_title = "main page",layout="wide")



main = st.Page("pages/main.py", title='main')
graph = st.Page("pages/graphs.py", title='Preprocessing')
classification = st.Page("pages/classifications.py", title='classification')
regression = st.Page("pages/regressions.py", title='regression')
deep = st.Page("pages/deeplearning.py", title='deep Learning')
conclusion = st.Page("pages/conclusion.py", title='Perspectives et recommandation')

pg = st.navigation([main,graph,classification,regression,deep,conclusion])

pg.run()

