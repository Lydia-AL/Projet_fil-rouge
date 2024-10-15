import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def GetFeatureImportance(model, data):
    # Plot features importances using Seaborn
    feats = {}
    for feature, importance in zip(data.columns, model.feature_importances_):
        feats[feature] = importance

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
    importances = importances.sort_values(by='Importance', ascending=False)

    plot = sns.barplot(x=importances.Importance, y=importances.index)
   
    return plot


def GetConfusionMatrices(model, X_train_scaled, X_test_scaled, y_train, y_test):
    
    # ======== Matrice de confusion pour l'ensemble de test ========
    
    y_pred_test = model.predict(X_test_scaled)
    cm_test = confusion_matrix(y_test, y_pred_test)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    
    plt.figure(figsize=(8, 8))
    plot = disp_test.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion - Données de test")
    plt.show()

    # ======== Matrice de confusion pour l'ensemble d'entraînement ========
    
    y_pred_train = model.predict(X_train_scaled)
    cm_train = confusion_matrix(y_train, y_pred_train)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    
    plt.figure(figsize=(8, 8))
    disp_train.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion - Données d'entraînement")
    plt.show()

    return plot #disp_test, disp_train


def GetGridSearchScore(GCV):
    results_df = pd.DataFrame(GCV.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")
    return results_df