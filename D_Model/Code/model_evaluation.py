# -*- coding: utf-8 -*-
# Module for model evaluation
import os
import pandas as pd
import sys
import datetime as dt
import  numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

# Add absolute path of directory 
# 01_ML_QC_MARNET to sys.path
currentdir=os.path.dirname(__file__)
pathpieces = os.path.split(currentdir)
while pathpieces[-1]!='01_ML_QC_MARNET':
    currentdir= os.path.dirname(currentdir)
    pathpieces = os.path.split(currentdir)
sys.path.insert(0,currentdir)


from B_ExpDatAn.Code.utilities import get_filestring, read_station_data  # noqa: E402
from B_ExpDatAn.Code.common_variables import (  # noqa: E402
    stations,
    params,
    tlims,
    stationsdict,
    paramdict,
    layout,
    cm,
    fs,
    fontdict,
    bbox_inches,
)  
from D_Model.Code.detect_anomalies import methods  # noqa: E402

# from where to load results
log_file = 'log_training.txt'

# where to save dataframe of training results
savepath = os.path.join(currentdir,'D_Model', 'Trained_Models', 'Ocean_WNN')

# where to save results from evaluation
evalpath = os.path.join(currentdir, 'D_Model','Evaluation', 'Figures')

mthds = {'Median-Methode': 'ad_mm', 'Ocean_WNN': 'ad_ownn'}

def get_numerics(s)  -> list:
    replace = [
        "tensor(",
        ",",
        ":",
        "\t",
        "\n"
    ]
    for cc in replace:
        s = s.replace(cc, ' ')
    fvals = []
    for t in s.split():
        try:
            fvals.append(int(t))
        except ValueError:
            try:
                fvals.append(float(t))
            except ValueError:
                pass
            pass
    return fvals

def get_len_train(modelOutputDir) -> float:
    for f in os.listdir(modelOutputDir):
        if f!=log_file:
            elem = f.split('_')
            start = dt.datetime.strptime(elem[0]+elem[1], "%Y%m%d%H")
            end = dt.datetime.strptime(elem[2]+elem[3], "%Y%m%d%H")
            len_train = end-start
            len_train = len_train.days*24+len_train.seconds/3600
    return len_train

def get_fit_res(lgfile) -> dict:
    f = open(lgfile, "r")
    Lines = f.readlines()
    threshline = Lines[-1]
    epochline = Lines[-2]
    fvals = get_numerics(threshline)
    thresh = fvals[0]
    fvals = get_numerics(epochline)
    n_epochs = fvals[0]
    tloss = fvals[1]
    vloss = fvals[2]
    model_dict={'n_epochs':n_epochs, 't_loss':tloss, 'v_loss':vloss, 'thresh':thresh}
    return model_dict

def summarize_model_fitting():
    '''Summarizes the results of oceanwnn model fitting'''
    model_data = []
    
    for st in stations:
        for p in params:
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            data=read_station_data(filestr=filestr)
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            for d in unique_d:
                depth = abs(d)
                # directory of trained models
                modelOutputDir = os.path.join(currentdir,
                                              'D_Model',
                                           'Trained_Models', 
                                           'Ocean_WNN', 
                                           stationsdict[st],
                                           p, 
                                           str(depth)+'m')
                # necessary variables
                # length of training data in hours: len_train
                # number of training epochs: n_epochs
                # last epoch's training loss: t_loss
                # last epoch's validation loss: v_loss
                # threshold value for anomaly detection: thresh_anom
                # depth as index

                len_train = get_len_train(modelOutputDir)
                model_dict = get_fit_res(os.path.join(modelOutputDir, log_file))
                new_entry = [st, p, depth, len_train]
                for v in model_dict.values():
                    new_entry.append(v)

                model_data.append(new_entry)
    # fill dataframe
    model_fit = pd.DataFrame(
        model_data,
        columns=[
            "Station",
            "Parameter",
            "Depth",
            "len_train",
            "n_epochs",
            "t_loss",
            "v_loss",
            "thresh",
        ],
    )
   # save results of model fitting
    model_fit.to_csv(os.path.join(savepath, 'results_model_fitting.csv')) 

def calc_roc_metrics(ts, method):
    y_score = ts[method].values
    y_true = ts.QF3.values
    y_true[y_true==2]=0
    y_true[y_true>2]=1
    y_true_s = y_true[np.where(np.isnan(y_score)==False)]
    if all(np.iszero(y_true_s)):
        fpr=tpr=threshs=auc=np.nan
        return fpr,tpr,threshs,auc
    y_score_s = y_score[np.where(np.isnan(y_score)==False)]
    auc = roc_auc_score(y_true_s, y_score_s)
    fpr,tpr,threshs = roc_curve(y_true_s, y_score_s)
    return fpr,tpr,threshs, auc
def main():
    '''Plot ROC metrics'''
    
    for st in stations:
        for p in params:
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            filestr = filestr.replace('.csv','_'+ methods+'.csv')
            filestr = filestr.replace('A_Data', os.path.join('D_Model', 'Results'))
            data=pd.read_csv(filestr)
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            legendstrings = []
            savefigpath = os.path.join(evalpath, stationsdict[st].replace(" ", "_"), p)
            if not os.path.isdir(savefigpath):
                    os.makedirs(savefigpath) 
            for d in unique_d:
                depth = abs(d)
                figname = str(depth)+'m_ROCmetrics.png'
                fig = plt.figure(layout=layout)
                ts = data[data["Z_LOCATION"]==d] # entries corresponding to depth level d
                for m in mthds:
                    fpr, tpr, threshs,auc = calc_roc_metrics(ts, mthds[m])
                    if all(np.isnan([fpr, tpr, threshs,auc])):
                        
                        continue
                    plt.plot(fpr, tpr, '.')
                    legendstrings.append(m+', AUC='+str(round(auc,2)))
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.legend(legendstrings)
                titlestring = stationsdict[st]+', '+\
                    str(depth)+' m,'+\
                        paramdict[p].replace('[°C]', '')
                                    
                plt.title(titlestring, fontsize=fs)
                plt.legend(legendstrings)
                #plt.show()
                fig.savefig(os.path.join(savefigpath,figname), bbox_inches=bbox_inches)
                plt.close(fig) 

if __name__=='__main__':   
    main()
