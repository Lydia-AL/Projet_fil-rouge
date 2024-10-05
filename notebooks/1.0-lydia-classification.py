import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,precision_score,confusion_matrix,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#Préparation des données

df = pd.read_csv('FinalData.csv', index_col=0)
df.drop_duplicates(inplace=True)

df_Categorielle = df.drop(['Mt', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'ec (cm3)', 'ep (KW)',
                           'Erwltp (g/km)', 'Fuel consumption '], axis=1)
df_quantitative = df[['Mt', 'Ewltp (g/km)', 'W (mm)', 'At1 (mm)', 'ec (cm3)', 'ep (KW)',
                       'Erwltp (g/km)', 'Fuel consumption ']]
df_quantitative = df_quantitative.fillna(df_quantitative.std()) 

df_Categorielle = df_Categorielle.join(pd.get_dummies(df_Categorielle[['Country','Ct','Ft','Cr']]))
df_Categorielle = df_Categorielle.drop(['Country','Ct','Ft','Cr'],axis=1)

le = LabelEncoder()
df_Categorielle['Mk'] = le.fit_transform(df_Categorielle['Mk'])
df_Categorielle['Mh'] = le.fit_transform(df_Categorielle['Mh'])
df_Categorielle['IT'] = le.fit_transform(df_Categorielle['IT'])

df_Final = df_Categorielle.join(df_quantitative)

data = df_Final.drop('Ewltp (g/km)',axis=1)
target = df_Final['Ewltp (g/km)']

def new_target(x):
    res = 0
    if(x <= 100) : res = 1
    elif((x > 100 ) & (x <= 120)):res = 2
    elif((x > 120) & (x <= 140)): res = 3
    elif((x > 140) & (x <= 160)): res = 4
    elif((x > 160) & (x <= 200)): res = 5
    elif((x > 200) & (x <= 250)): res = 6
    elif((x > 250)): res = 7
    return res

target = target.apply(new_target)
target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=123)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Decision tree classifier

dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=4,random_state=123)
dt_clf.fit(X_train_scaled,y_train)

y_pred = dt_clf.predict(X_test_scaled)
pd.crosstab(y_test, y_pred, rownames=['reel'], colnames=['Predicted'])

#Feature importance DecisionTree Classifier

cols = data.columns
result = {}

for i,j in zip(cols,dt_clf.feature_importances_):
    if j > 0 : 
        #print(i,j)
        result[i] = j

importante_feature = pd.DataFrame.from_dict(result,orient='index', columns=['Feature'])
print(importante_feature.sort_values(by="Feature",ascending=False))

#Gradient boosting

gb = GradientBoostingClassifier()
gb.fit(X_train_scaled,y_train)

y_pred_gb = gb.predict(X_test_scaled)
pd.crosstab(y_test, y_pred, rownames=['reel'], colnames=['Predicted'])

#Feature importance Gradient Boosting

#KNN

knn = KNeighborsClassifier(n_neighbors=7,p=2, metric="minkowski")
knn.fit(X_train_scaled,y_train)

y_pred_KNN = knn.predict(X_test_scaled)
pd.crosstab(y_test, y_pred_KNN, rownames=['reel'], colnames=['Predicted'])

#GRIDSEARCH

#Gridsearch_Decision tree classifier

param_gridDTC = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'max_depth':  [None, 2, 4, 6, 8, 10],
}
grid_DTC = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_gridDTC, n_jobs=-1,
                        cv=2, verbose=1,refit= True)
grille = grid_DTC.fit(X_train_scaled,y_train)
print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']])

print('le meilleur paramètre de DecisionTreeClassifier est :',grid_DTC.best_params_)

#Gridsearch Gradient Boosting 90 min

param_gridGBC = {
    'learning_rate' : [0.1,0.2,0.3,0.4,0.5],
    'n_estimators':  [100, 200, 500],
}
grid_GBC = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_gridGBC, n_jobs=-1,
                        cv=2, verbose=1,refit= True)
grille = grid_GBC.fit(X_train_scaled,y_train)

print('le meilleur paramètre de GradientBoostingClassifier est :',grid_GBC.best_params_)
print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']])

#Gridsearch KNeighboor Classifier

param_gridKNN= {
    'metric' : ['minkowski','cityblock','cosine','euclidean','haversine','l1','l2','manhattan','nan_euclidean'],
    'n_neighbors':np.arange(2,10)
}


grid_KNN = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_gridKNN, n_jobs=-1,
                        cv=2, verbose=1,refit= True)
grille = grid_KNN.fit(X_train_scaled,y_train)

print('le meilleur paramètre de KNeighborsClassifier est :',grid_KNN.best_params_)
print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']])