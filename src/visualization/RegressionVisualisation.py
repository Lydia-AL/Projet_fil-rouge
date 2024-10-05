import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def GetFeatureImportance(model, data):
    # Plot feature importance using Seaborn
    feats = {}
    for feature, importance in zip(data.columns, model.feature_importances_):
        feats[feature] = importance

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
    importances.sort_values(by='Importance', ascending=False)

    plot = sns.barplot(x=importances.Importance, y=importances.index)
   
    return plot
#, bestparam

def GetTestPredictions(model,X_train_scaled,X_test_scaled,y_train,y_test):
    
    y_pred_rf = model.predict(X_test_scaled)
    model.score(X_test_scaled, y_test)

    plt.subplots(figsize = (10,8))
    plot = plt.scatter(y_pred_rf,y_test)
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()),'r');

    return plot

def GetGridSearchScore(GCV):
    results_df = pd.DataFrame(GCV.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")
    return results_df