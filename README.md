# 🚗 Analyse des émissions de CO₂ des véhicules avec la Data Science 🌍
Ce projet explore les facteurs influençant les émissions de CO₂ des véhicules à partir de données techniques (consommation de carburant, puissance du moteur, etc.). Il applique des modèles de machine learning et propose des recommandations pour réduire ces émissions.

## 📊 Données & Méthodologie  
Les données utilisées proviennent de l’**European Environment Agency (EEA)**, qui recense les émissions de CO₂ des véhicules en Europe. Elles incluent des informations sur la consommation de carburant, le type de carburant, la puissance du moteur et la masse totale du véhicule.  


## 📊 Données & Méthodologie
- **Exploration** : Visualisations et analyses des tendances.
- **Pré-processing** : Nettoyage des données et transformation des variables.
- **Modélisation** : Régression (Random Forest, Gradient Boosting, Ridge) et classification.
- **Deep Learning** : Test d’un réseau de neurones.
- **Déploiement** : Présentation interactive sur Streamlit.


==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
