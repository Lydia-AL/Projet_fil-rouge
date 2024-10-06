import sys
import os
# Ajouter dynamiquement le chemin de la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,precision_score,confusion_matrix,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('C:\Users\lydia\Documents\GitHub\dec23_cds_co2\data\processed\FinalData.csv', index_col=0)
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

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Decision tree classifier

dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=4,random_state=123)
dt_clf.fit(X_train_scaled,y_train)
y_test_pred = dt_clf.predict(X_test_scaled)
y_train_pred = dt_clf.predict(X_train_scaled)

dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=4,random_state=123)
dt_clf.fit(X_train_scaled,y_train)
y_test_pred_dt = dt_clf.predict(X_test_scaled)
y_train_pred_dt = dt_clf.predict(X_train_scaled)

#Matrice de confusion
print("Confusion Matrix train")
print(pd.crosstab(y_train, y_train_pred_dt, rownames=['reel'], colnames=['Predicted']))
print("\n")
print("Confusion Matrix test")
print(pd.crosstab(y_test, y_test_pred_dt, rownames=['reel'], colnames=['Predicted']))


#Calcul des scores

#accuracy
train_accuracy = accuracy_score(y_train, y_train_pred_dt)
test_accuracy = accuracy_score(y_test, y_test_pred_dt)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("\n")

#precision_score
train_precision = precision_score(y_train, y_train_pred_dt, average='weighted')
test_precision = precision_score(y_test, y_test_pred_dt, average='weighted')
print(f"Train precision: {train_precision * 100:.2f}%")
print(f"Test precision: {test_precision * 100:.2f}%")
print("\n")

#recall_score
train_recall = recall_score(y_train, y_train_pred_dt, average='weighted')
test_recall = recall_score(y_test, y_test_pred_dt, average='weighted')
print(f"Train recall: {train_recall * 100:.2f}%")
print(f"Test recall: {test_recall * 100:.2f}%")
print("\n")

#f1_score
train_f1 = f1_score(y_train, y_train_pred_dt, average='weighted')
test_f1 = f1_score(y_test, y_test_pred_dt, average='weighted')
print(f"Train f1: {train_f1 * 100:.2f}%")
print(f"Test f1: {test_f1 * 100:.2f}%")
print("\n")

#classification_report
class_report_train = classification_report(y_train, y_train_pred_dt)
class_report_test = classification_report(y_test, y_test_pred_dt)
print("Classification Report Train:", class_report_train)
print("Classification Report Test:", class_report_test)

#cross_validation_score
scores = cross_val_score(dt_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f'Accuracy moyenne en validation croisée: {scores.mean():.4f}')


#Features importance DecisionTree Classifier

feats = {}
for feature, importance in zip(df_Final.columns, dt_clf.feature_importances_):
    feats[feature] = importance 
    
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
importances.sort_values(by='Importance', ascending=False)



#Features importance visualisation

importances_sorted = importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x=importances_sorted['Importance'], y=importances_sorted.index, palette='Blues_d')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()

#Gridsearch_Decision tree classifier

param_gridDTC = {
    'criterion' : ['gini', 'entropy'],
    'max_depth':  [None, 2, 4, 6, 8, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
}
grid_DTC = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_gridDTC, n_jobs=-1,
                        cv=2, verbose=1,refit= True)
grille = grid_DTC.fit(X_train_scaled,y_train)
print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']])

print('le meilleur paramètre de DecisionTreeClassifier est :',grid_DTC.best_params_)

