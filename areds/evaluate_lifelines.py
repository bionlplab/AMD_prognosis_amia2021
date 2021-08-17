from sksurv.metrics import (concordance_index_censored,
                            concordance_index_ipcw,
                            cumulative_dynamic_auc)
from lifelines.utils import concordance_index
import numpy as np

import os  # here
import torch  # here
import sklearn  # here
import collections  # here
import pandas

import rpy2

print(rpy2.__version__)
'''
    Change below to your home R environment 

    in R run 
        R.home(component = "home")
    to return R home environment
'''
os.environ["R_HOME"] = "/users/gregory/anaconda3/lib/R"

from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

print("TEST")

# import R's "base" package
base = importr('base')

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

packnames = ('survival', 'timeROC', 'ggplot2')

# R vector of strings
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import numpy2ri

# Selectively install what needs to be install.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

ROC = importr('timeROC')

def evaluate_AUCs_lifelines(model, x_dev, label_fit, labelv, losses, print_each_k_means = True):
    '''
        Evaluates for lifelines model (CoxPH):
            test loss
            test concordance index
            test inverse probability cersoring weight concordance
            test Area under ROC curve for visit 10
            test Area under ROC curve for visit 16
            test Area under ROC curve for visit 10 using a second algorithm to calculate evaluation
            test Area under ROC curve for visit 16 using a second algorithm to calculate evaluation
            test Area under Precision-Recall curve for visit 10
            test Area under Precision-Recall curve for visit 16

        input:
            model - deep survival model
            x_dex - test X features
            label_fit - training y features (time to event, precense of event)
            labelv - test y features (time to event, precense of event)
            losses - string of loss previously calculated
            print_each_k_means - boolean of if want to print out evaluation results
    '''
    outputv = model.predict_partial_hazard(x_dev)
    outputv = np.array(outputv)
    outputv = outputv.flatten()

    labelv = np.array(labelv).T
    label_fit = np.array(label_fit).T


    idxv = pandas.Series(labelv[0]).sort_values(ascending=False).index

    time_distv = labelv[0]
    event_occurv = labelv[1]
    time_distv = time_distv[idxv]
    event_occurv = event_occurv[idxv]
    outputv2 = outputv.copy()
    outputv = outputv[idxv]


    part_haz = outputv
    conc = concordance_index(time_distv, -part_haz, event_occurv)

    cutoff_visits = [10, 16]
    lab_train = []
    lab_val = []

    for q in range(label_fit[0].shape[0]):
        lab_train.append((bool(label_fit[1][q].item()), int(label_fit[0][q].item())))
    for p in range(labelv[0].shape[0]):
        lab_val.append((bool(labelv[1][p].item()), int(labelv[0][p].item())))

    lab_train = np.array(lab_train, dtype=[('death', '?'), ('futime', '<f8')])
    lab_val = np.array(lab_val, dtype=[('death', '?'), ('futime', '<f8')])

    auc = cumulative_dynamic_auc(lab_train, lab_val, outputv2, times=np.asarray(cutoff_visits))
    conc_ipcw = concordance_index_ipcw(lab_train, lab_val, outputv2)

    Surv = importr('survival')
    TP = []
    FP = []
    PPVres = []

    test = np.linspace(np.min(outputv) - .1,
                       np.max(outputv) + .1, outputv.shape[0], endpoint=False)

    for ii in np.flip(np.sort(test)):
        numpy2ri.activate()
        rc = ROC.timeROC(time_distv,
                         event_occurv,
                         outputv, cause=1, times=np.array(cutoff_visits))
        PPV = ROC.SeSpPPVNPV(cutpoint=ii, T=time_distv,
                             delta=event_occurv,
                             marker=outputv, cause=1, times=np.array(cutoff_visits))
        numpy2ri.deactivate()
        TP.append(PPV.rx2('TP'))
        PPVres.append(PPV.rx2('PPV'))
        FP.append(PPV.rx2('FP'))

    PPV10 = np.array(PPVres)[:, 0]
    PPV16 = np.array(PPVres)[:, 1]

    PPV10[np.isnan(PPV10)] = 0
    PPV16[np.isnan(PPV16)] = 0

    auc10 = sklearn.metrics.auc(np.array(FP)[:, 0], np.array(TP)[:, 0])
    auc16 = sklearn.metrics.auc(np.array(FP)[:, 1], np.array(TP)[:, 1])

    ppvauc10 = sklearn.metrics.auc(np.array(TP)[:, 0], PPV10)
    ppvauc16 = sklearn.metrics.auc(np.array(TP)[:, 1], PPV16)

    lossv = float(losses.split('\t')[-1].split(': ')[-1])

    if (print_each_k_means == True):
        print([f'val_loss: {round(lossv, 4)}, val_con: {round(conc, 4)}, val_con_ipcw: {round(conc_ipcw[0], 4)}, val_auc_10: {round(auc[0][0], 4)}, val_auc_16: {round(auc[0][1], 4)}, val_auc_10v: {round(auc10, 4)}, val_auc_16v: {round(auc16, 4)}, val_ppvauc_10v: {round(ppvauc10, 4)}, val_ppvauc_16v: {round(ppvauc16, 4)}'])

    return lossv, conc, conc_ipcw[0], auc[0][0], auc[0][1], auc10, auc16, ppvauc10, ppvauc16
