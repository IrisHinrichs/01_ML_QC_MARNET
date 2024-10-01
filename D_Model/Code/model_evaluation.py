# -*- coding: utf-8 -*-
# Module for model evaluation
import os
import pandas as pd
import sys
import datetime as dt
import  numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

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
from B_ExpDatAn.Code.time_series_statistics import find_all_time_spans  # noqa: E402
from C_DataPreProc.Code.data_preprocessing import piecewise_interpolation  # noqa: E402  
from D_Model.Code.detect_anomalies import methods  # noqa: E402
from D_Model.Code.ocean_wnn.algorithm_iris import run_ownn_algorithm  # noqa: E402
from D_Model.Code.ocean_wnn.algorithm_iris import CustomParameters as ownn_custPar  # noqa: E402


# from where to load results
log_file = 'log_training.txt'

# where to save dataframe of training results
savepath = os.path.join(currentdir,'D_Model', 'Trained_Models', 'Ocean_WNN')
resultsfile = os.path.join(savepath, 'results_model_fitting.csv')

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
    return start.strftime("%Y%m%d%H"), end.strftime("%Y%m%d%H"), len_train

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

def summarize_model_fitting() -> pd.DataFrame:
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

                start_train, end_train,len_train = get_len_train(modelOutputDir)
                model_dict = get_fit_res(os.path.join(modelOutputDir, log_file))
                new_entry = [st, p, depth,start_train, end_train, len_train]
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
            "start_train",
            "end_train",
            "len_train",
            "n_epochs",
            "t_loss",
            "v_loss",
            "thresh",
        ],
    )
   # save results of model fitting
    model_fit.to_csv(resultsfile)
    return model_fit 

def calc_roc_metrics(ts, method):
    y_score = ts[method].values
    y_true = ts.QF3.values
    y_true[y_true==2]=0
    y_true[y_true>2]=1
    y_true_s = y_true[np.where(not np.isnan(y_score))]
    if all(y_true_s==0): # no bad data in current time_series
        fpr=tpr=threshs=auc=np.nan
        return fpr,tpr,threshs,auc
    y_score_s = y_score[np.where(not np.isnan(y_score))]
    auc = roc_auc_score(y_true_s, y_score_s)
    fpr,tpr,threshs = roc_curve(y_true_s, y_score_s)
    return fpr,tpr,threshs, auc

def non_NaN(ts):
    # choose evaluation data
    # needs be no-NaN values in both ad_mm- as well as ad_ownn-column
    # plus optionally data after training phase of Ocean_WNN prediciton model
    rowind = np.where(~np.isnan(ts['ad_mm']) & ~np.isnan(ts['ad_ownn'])) 
    ts_eval = ts.iloc[rowind[0]]
    return ts_eval

def after_training_phase(st, p, depth, ts_eval):
    # temporal restriction, only data after training phase
    if os.path.isfile(resultsfile):
        model_fit = pd.read_csv(resultsfile)
    else:
        model_fit = summarize_model_fitting()
                # find end of training phase for model for  
                # current station, parameter and depth level
    train_interval= model_fit.loc[
                    (model_fit.Station == st)
                    & (model_fit.Parameter == p)
                    & (model_fit.Depth == depth),
                    ['start_train', 'end_train']
                    ]
    ts_eval = ts_eval[ts_eval.index>dt.datetime.strptime(str(train_interval.end_train.iloc[0]),'%Y%m%d%H')]
    return ts_eval

def plot_roc_metrics():
    '''Plot ROC metrics
    plots ROC-Curves for all stations, parameters and depth levels
    presenting results of Ocean_WNN and Median-Method'''
    
    for st in stations:
        for p in params:
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            filestr = filestr.replace('.csv','_'+ methods+'.csv')
            filestr = filestr.replace('A_Data', os.path.join('D_Model', 'Results'))
            data=pd.read_csv(filestr, index_col='TIME_VALUE')
            data.index = pd.to_datetime(data.index)
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            legendstrings = []
            savefigpath = os.path.join(evalpath, 'ROC_metrics', stationsdict[st].replace(" ", "_"), p)
            if not os.path.isdir(savefigpath):
                    os.makedirs(savefigpath) 
            for d in unique_d:
                depth = abs(d)
                figname = str(depth)+'m_ROCmetrics.png'
                fig = plt.figure(layout=layout)
                ts = data[data["Z_LOCATION"]==d] # entries corresponding to depth level d
                # choose evaluation data
                # needs be no-NaN values in ad_mm- as well as ad_ownn-column
                # plus optionally data after training phase of Ocean_WNN prediciton model
                ts_eval = non_NaN(ts)
                # temporal restriction, only data after training phase
                ts_eval = after_training_phase(st, p, depth, ts_eval)
                for m in mthds:
                    fpr, tpr, threshs,auc = calc_roc_metrics(ts_eval, mthds[m])
                    if all([np.isnan(fpr).all(), np.isnan(tpr).all(), np.isnan(threshs).all()]): # no anomalies in ground truth data

                        continue
                    plt.plot(fpr, tpr, '.')
                    legendstrings.append(m+', AUC='+str(round(auc,2)))
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.legend(legendstrings)
                titlestring = stationsdict[st]+', '+\
                    str(depth)+' m,'+\
                        paramdict[p].replace('[° C]', '').replace('[]', '')
                                    
                plt.title(titlestring, fontsize=fs)
                plt.legend(legendstrings)
                #plt.show()
                #fig.savefig(os.path.join(savefigpath,figname), bbox_inches=bbox_inches)
                plt.close(fig)

def main():
    '''Compare Ocean_WNN-predicitions and actual observations'''
    # define figure height
    plt.rcParams['figure.figsize'][0]=16.5*cm
    plt.rcParams['figure.figsize'][1]=6*cm
    model_fit = pd.read_csv(resultsfile)
    for st in stations:
        for p in params:
           
            #define colormap
            if p == 'WT':
                col='blue'
                ylabelstr = '[° C]'
            else:
                col='purple'
                ylabelstr = '[]'
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            # replace following line with something like 
            # data = read_interp_data(filestr)
            data=read_station_data(filestr=filestr)
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            for d in unique_d:
                depth = abs(d)
                # define path to save figures
                savefigpath = os.path.join(evalpath, 'Predictions', stationsdict[st].replace(" ", "_"), p+'_'+str(depth))
                if not os.path.isdir(savefigpath):
                        os.makedirs(savefigpath) 

                ts = data[data["Z_LOCATION"]==d] # entries corresponding to depth level d
                # STEP I: piecewise interpolation of all time series
                ts_interp = piecewise_interpolation(ts, gap=1)

                # find all time spans longer than train window size
                time_spans = find_all_time_spans(time_vec=ts_interp.index, tdel=1)
            
                # ocean_wnn
                modelOutput = os.path.join(currentdir,
                                              'D_Model',
                                           'Trained_Models', 
                                           'Ocean_WNN', 
                                           stationsdict[st],
                                           p, 
                                           str(abs(d))+'m')
                train_interval= model_fit.loc[
                    (model_fit.Station == st)
                    & (model_fit.Parameter == p)
                    & (model_fit.Depth == depth),
                    ['start_train', 'end_train']
                    ]
                begin = str(train_interval.start_train.values[0])
                end = str(train_interval.end_train.values[0])
                begin = begin[0:8]+'_'+begin[8:11]
                end = end[0:8]+'_'+end[8:11]
                modelOutput=os.path.join(modelOutput,
                                begin+
                                '_'+end)
            

                for tp in time_spans.index:
                    # define time stamps for current part of
                    # time series
                    end = tp+time_spans[tp]
                    begin=pd.Timestamp(tp.year, tp.month, tp.day, tp.hour, 0)

                    # get existing time stamps for part of time series
                    inds = np.where(((ts_interp.index >= begin)&(ts_interp.index<=end)))
                    linds = len(inds[0])

                    if linds<=ownn_custPar.train_window_size: 
                        continue
                    else:
                        dat= ts_interp.iloc[inds[0]].DATA_VALUE.to_numpy().reshape(-1,1) 
                        # Prediction with Ocean_WNN model
                        ts_predict = run_ownn_algorithm(
                            dat,
                            modelOutput=modelOutput,
                            executionType="predict"
                        )  
                        time_vec = pd.date_range(start=begin, end=end, freq= 'h')
                        fig = plt.figure()
                        figname = (
                            str(depth)
                            + "m_Predictions_"
                            + begin.strftime("%Y-%m-%d_%H-%M-%S")
                            + "__"
                            + end.strftime("%Y-%m-%d_%H-%M-%S")
                            + ".png"
                        )
                        plt.plot(time_vec, ts_predict, 'go',alpha=0.2, markersize=3, linewidth=2)
                        plt.plot(time_vec, dat, '.',color = col, markersize=3, linewidth=2)
                        plt.grid()
                        pstring = paramdict[p].replace('[° C]', '').replace('[]', '')
                        titlestring = stationsdict[st]+', '+pstring+', '+\
                                        begin.strftime('%d.%m.%Y %H:%M:%S')+'-'+end.strftime('%d.%m.%Y %H:%M:%S')
                        plt.title(titlestring, fontsize=fs, wrap=True)
                        plt.ylabel(ylabelstr)
                        plt.annotate(str(abs(d))+' m', xy=(0.05, 0.05), xycoords='axes fraction')
                        
                        plt.gca().xaxis.set_major_formatter(
                        mdates.ConciseDateFormatter(plt.gca().xaxis.get_major_locator()))

                        plt.xlim(begin, end)

                        plt.legend(['Vorhersage', 'Beobachtung'])
                        
                        #plt.show()
                        fig.savefig(os.path.join(savefigpath,figname), bbox_inches=bbox_inches)
                        plt.close(fig)
                
if __name__=='__main__':   
    #test = summarize_model_fitting()
    main()
