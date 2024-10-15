import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics 

def GetLossByEpoch(history):
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='right')
    return fig 

def GetMaeByEpoch(history):
    fig = plt.figure()
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Model val_mean_absolute_error by epoch')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='right')
    return fig

def GetAccByEpoch(history):
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model acc by epoch')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='right')
    return fig

def GetConfusionMatrices(model, X_train_scaled, X_test_scaled, y_train, y_test):
    test_pred = model.predict(X_test_scaled)
    test_pred_class = test_pred.argmax(axis = 1)
    y_test_class = y_test

    cnf_matrix = metrics.confusion_matrix(y_test_class, test_pred_class)
    disp_test = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    
    plt.figure(figsize=(8, 8))
    plot = disp_test.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion")
    plt.show()

    return plot

def GetTestPredictions(model,X_train_scaled,X_test_scaled,y_train,y_test):
    
    y_pred = model.predict(X_test_scaled)
    plt.subplots(figsize = (10,8))
    plot = plt.scatter(y_pred[:,0],y_test)
    plt.xlabel('predicted')
    plt.ylabel('reel')
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()),'r');

    return plot