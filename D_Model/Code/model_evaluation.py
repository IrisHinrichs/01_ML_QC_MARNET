# -*- coding: utf-8 -*-
# Module for model evaluation
import os
import pandas as pd
import sys
import datetime as dt

# Add absolute path of directory 
# 01_ML_QC_MARNET to sys.path
currentdir=os.path.dirname(__file__)
pathpieces = os.path.split(currentdir)
while pathpieces[-1]!='01_ML_QC_MARNET':
    currentdir= os.path.dirname(currentdir)
    pathpieces = os.path.split(currentdir)
sys.path.insert(0,currentdir)


from B_ExpDatAn.Code.utilities import get_filestring, read_station_data  # noqa: E402
from B_ExpDatAn.Code.common_variables import stations, params, tlims, stationsdict  # noqa: E402

# from where to load results
resultspath = os.path.join(currentdir,'D_Model', 'Results')
log_file = 'log_training.txt'

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

def get_len_train(modelOutputDir) -> dt.timedelta:
    for f in os.listdir(modelOutputDir):
        if f!=log_file:
            elem = f.split('_')
            start = dt.datetime.strptime(elem[0]+elem[1], "%Y%m%d%H")
            end = dt.datetime.strptime(elem[2]+elem[3], "%Y%m%d%H")
            len_train = end-start
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

def main():
    model_fit = pd.DataFrame()
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
main()
