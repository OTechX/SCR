import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,roc_curve,balanced_accuracy_score,recall_score,precision_score,confusion_matrix,f1_score,auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt
import time


def get_metrics(y_true, y_pre):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pre).ravel()
    tpr=tp/(tp+fn)#recall
    tnr=tn/(tn+fp)
    bac=0.5 * (tpr + tnr)
    fpr = 1-tnr
    far = fpr
    frr = 1-tpr
    return bac, far, frr


def get_eer(fpr, tpr, thresholds): 
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer


def data_train_test(x_leg,y_leg,x_ill,y_ill,train_index_leg,test_index_leg,leg_user):
    x_train_leg, x_test_leg = x_leg[train_index_leg], x_leg[test_index_leg]
    y_train_leg, y_test_leg = y_leg[train_index_leg], y_leg[test_index_leg]
    x_test = np.concatenate([x_test_leg,x_ill],axis = 0)
    y_true = np.concatenate([y_test_leg,y_ill]) 
    y_true[np.where(y_true == leg_user)] = 1
    y_true[np.where(y_true != '1')] = -1
    y_true = [int(y) for y in y_true ]
    return x_train_leg, x_test, y_true


def svm_pre_score(x_train_leg, x_test):
    clf = OneClassSVM(gamma='scale',kernel='rbf',nu=0.01)
    clf.fit(x_train_leg)
    y_pre = clf.predict(x_test)
    pre_score = clf.decision_function(x_test)
    return y_pre, pre_score


def lof_pre_score(x_train_leg, x_test):
    clf = LocalOutlierFactor(novelty=True,contamination='auto',n_neighbors=3)
    clf.fit(x_train_leg)
    y_pre = clf.predict(x_test)
    pre_score = clf.decision_function(x_test)
    return y_pre, pre_score


