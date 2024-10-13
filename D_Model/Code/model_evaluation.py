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
from matplotlib.legend_handler import HandlerTuple
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error as mase

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
from C_DataPreProc.Code.data_preprocessing import piecewise_interpolation, differencing, reverse_diff  # noqa: E402  
from D_Model.Code.detect_anomalies import methods  # noqa: E402
from D_Model.Code.ocean_wnn.algorithm_iris import run_ownn_algorithm  # noqa: E402
from D_Model.Code.ocean_wnn.algorithm_iris import CustomParameters as ownn_custPar  # noqa: E402


# from where to load results
log_file = 'log_training.txt'

# differencing parameter
ddiff = 2

# where to save dataframe of training results
savepath = os.path.join(
    currentdir,
    "D_Model",
    "Trained_Models",
    "Ocean_WNN",
    "Diff_"+str(ddiff)
)
resultsfile = os.path.join(savepath, 'results_model_fitting.csv')

# where to save results from evaluation
evalpath = os.path.join(currentdir, 'D_Model','Evaluation', 'Figures', "Diff_"+str(ddiff))

mthds = {'Median-Methode': 'ad_mm', 'Ocean_WNN': 'ad_ownn'}

def optimal_thresh(tpr: np.array, fpr: np.array, threshs: np.array) -> tuple:
    '''Find optimal threshold for classifier based on both hit
    and false alarm rate and the derived Younden's J Index'''
    YJ_index = tpr-fpr
    index = np.argmax(YJ_index)
    opt_thresh = threshs[index]
    return opt_thresh, index
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
                modelOutputDir = os.path.join(savepath,
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
    y_true_s = y_true[np.where(np.isnan(y_score)==False)]
    if all(y_true_s==0): # no bad data in current time_series
        fpr=tpr=threshs=auc=np.nan
        return fpr,tpr,threshs,auc
    y_score_s = y_score[np.where(np.isnan(y_score)==False)]
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
    # define figure height
    plt.rcParams['figure.figsize'][0]=8*cm
    plt.rcParams['figure.figsize'][1]=8*cm
    for st in stations:
        for p in params:
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            filestr = filestr.replace('.csv','_'+ methods+'.csv')
            filestr = filestr.replace('A_Data', os.path.join('D_Model', 'Results', 'Diff_'+str(ddiff)))
            data=pd.read_csv(filestr, index_col='TIME_VALUE')
            data.index = pd.to_datetime(data.index)
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            savefigpath = os.path.join(evalpath, 'ROC_metrics', stationsdict[st].replace(" ", "_"), p)
            if not os.path.isdir(savefigpath):
                    os.makedirs(savefigpath) 
            for d in unique_d:
                legendstrings = []
                legendhandles = []
                depth = abs(d)
                figname = str(depth)+'m_ROCmetrics.png'
                fig = plt.figure(layout=layout)
                ts = data[data["Z_LOCATION"]==d] # entries corresponding to depth level d
                # choose evaluation data
                # needs be no-NaN values in ad_mm- as well as ad_ownn-column
                # plus optionally data after training phase of Ocean_WNN prediciton model
                ts_eval = non_NaN(ts)
                # temporal restriction, only data after training phase
                # ts_eval = after_training_phase(st, p, depth, ts_eval)
                for m in mthds:
                    fpr, tpr, threshs,auc = calc_roc_metrics(ts_eval, mthds[m])
                    if all([np.isnan(fpr).all(), np.isnan(tpr).all(), np.isnan(threshs).all()]): # no anomalies in ground truth data
                        continue
                    opt_thresh, index = optimal_thresh(tpr, fpr, threshs)
                    l2, = plt.plot(fpr[index], tpr[index], 'ko',alpha=0.2, markersize=7, linewidth=2)
                    #l2.set_label('optimal thresh')
                    l1,  = plt.plot(fpr, tpr, '.')
                    #l1.set_label(m+', AUC='+str(round(auc,2)))
                    legendhandles.append(l1)
                    legendstrings.append(m+', AUC='+str(round(auc,2)))
                legendhandles.append(l2)    
                legendstrings.append('optimaler Schwellwert')
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.legend(legendhandles, legendstrings)
                titlestring = stationsdict[st]+', '+\
                    str(depth)+' m,'+\
                        paramdict[p].replace('[° C]', '').replace('[PSU]', '')
                                    
                plt.title(titlestring, fontsize=fs)
                #plt.legend()
                plt.grid()
                #plt.show()
                fig.savefig(os.path.join(savefigpath,figname), bbox_inches=bbox_inches)
                plt.close(fig)

def reverse_differencing(startpoints: np.array, ts_predict: np.array)->np.array:
    ''' Does not really make sense because deviations between predictions and observations
    sum up to considerable divergences of time series'''
    if len(startpoints)==2:
        first_value = np.diff(startpoints, axis=0)
        ts_predict = reverse_diff(first_value, ts_predict)
    first_value = startpoints[0]
    ts_predict = reverse_diff(first_value, ts_predict)
    return ts_predict

def predictions_observations():
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
                ylabelstr = '[PSU]'
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            # replace following line with something like 
            # data = read_interp_data(filestr)
            data=read_station_data(filestr=filestr)
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            for d in unique_d:
                depth = abs(d)
                # define path to save figures
                savefigpath = os.path.join(
                    evalpath,
                    "Predictions",
                    stationsdict[st].replace(" ", "_"),
                    p + "_" + str(depth),
                )
                if not os.path.isdir(savefigpath):
                        os.makedirs(savefigpath) 

                ts = data[data["Z_LOCATION"]==d] # entries corresponding to depth level d
                # STEP I: piecewise interpolation of all time series
                ts_interp = piecewise_interpolation(ts, gap=10)

                # find all time spans
                time_spans = find_all_time_spans(time_vec=ts_interp.index, tdel=10)
            
                # ocean_wnn
                modelOutput = os.path.join(savepath,
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

                    if linds<=ownn_custPar.train_window_size+ddiff: 
                        continue
                    else:
                        # difference time series ddiff times
                        dat = differencing(ts_interp.iloc[inds[0]].DATA_VALUE, ddiff)
                        # Prediction with Ocean_WNN model
                        ts_predict = run_ownn_algorithm(
                            dat,
                            modelOutput=modelOutput,
                            executionType="predict"
                        ) 
                        
                        ts_predict = np.append([np.nan]*ddiff, ts_predict)
                        dat = np.append([np.nan]*ddiff, dat)
                        
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
                        
                        plt.plot(time_vec, dat, '.',color = col, markersize=3, linewidth=2)
                        plt.plot(time_vec, ts_predict, 'go',alpha=0.2, markersize=3, linewidth=2)
                        plt.grid()
                        pstring = paramdict[p].replace(' [° C]', '').replace(' []', '')
                        titlestring = stationsdict[st]+', '+pstring+', '+\
                                        begin.strftime('%d.%m.%Y %H:%M:%S')+'-'+end.strftime('%d.%m.%Y %H:%M:%S')
                        plt.title(titlestring, fontsize=fs, wrap=True)
                        plt.ylabel(ylabelstr)
                        plt.annotate(str(abs(d))+' m', xy=(0.05, 0.05), xycoords='axes fraction')
                        
                        plt.gca().xaxis.set_major_formatter(
                        mdates.ConciseDateFormatter(plt.gca().xaxis.get_major_locator()))

                        plt.xlim(begin, end)

                        plt.legend(['Beobachtung','Vorhersage'])
                        
                        #plt.show()
                        fig.savefig(os.path.join(savefigpath,figname), bbox_inches=bbox_inches)
                        plt.close(fig)
def model_cross_validation():
    ''' Train model on single period of good data and 
    validate it comparing observations
    and predictions on on all other periods of good data
    '''
    min_day = 7
    # minimum number of days in training data 

    cross_val_data = []

    resultsfile = os.path.join(currentdir,
                                'D_Model',
                                'Cross_Validation',
                                'Ocean_WNN',
                                'Diff_'+str(ddiff),
                                'results_cross_val.csv'
    )   
    for st in stations:
        if st=='Fehmarn Belt Buoy':
            continue
        for p in params:
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            data=read_station_data(filestr=filestr)
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            for d in unique_d:
                depth=abs(d)
                # ocean_wnn, validation
                
                ts = data[data["Z_LOCATION"]==d] # entries corresponding to depth level d
                
                good_vals = ts[ts['QF3']==2] # only use good values for training based on original times series

                time_spans = find_all_time_spans(time_vec=good_vals.index, tdel=1)

                # # sort time spans by length, begining with longest
                time_spans_sort = time_spans.sort_values(ascending=False)

                # only keep those time spans longer than min_day days
                time_spans = time_spans_sort[time_spans_sort>=dt.timedelta(min_day)]

                # training with several parts of time series
                for pp in time_spans.index:
                    # initialize list for collection of all mase values of all time series parts
                    all_mase = [] 
                    
                    # define time stamps for current part of time series
                    end = pp+time_spans[pp]
                    begin=pd.Timestamp(pp.year, pp.month, pp.day, pp.hour, 0)

                    # get existing time stamps for part of time series
                    inds = np.where(((good_vals.index >= begin)&(good_vals.index<=end)))
                    
                    # difference time series ddiff times
                    dat = differencing(good_vals.iloc[inds[0]].DATA_VALUE, ddiff)
                    
                    model_name = (
                        dt.datetime.strftime(begin, "%Y%m%d_%H")
                        + "_"
                        + dt.datetime.strftime(end, "%Y%m%d_%H")
                    )
                    # define and create directory if necessary
                    modelOutputDir = os.path.join(currentdir,
                                              'D_Model',
                                           'Cross_Validation',
                                           'Ocean_WNN',
                                           'Diff_'+str(ddiff), 
                                           stationsdict[st],
                                           p, 
                                           str(abs(d))+'m', 
                                           model_name)
                    if not os.path.isdir(modelOutputDir):
                        os.makedirs(modelOutputDir)
                   
                    modelOutput=os.path.join(modelOutputDir,
                                            model_name)
                    log_file = os.path.join(modelOutputDir, 
                                            'log_file.txt')

                    # train model on current part of time series 
                    run_ownn_algorithm(
                        dat,
                        modelOutput=modelOutput,
                        executionType="train",
                        logfile=log_file,
                    )
                    y_train = dat
                    for qq in time_spans_sort.index:
                        # skip time interval corresponding to training data
                        if qq==pp:
                            continue
                        # define time stamps for current part of
                        # time series
                        end = qq+time_spans_sort[qq]
                        begin=pd.Timestamp(qq.year, qq.month, qq.day, qq.hour, 0)

                        # get existing time stamps for part of time series
                        inds = np.where(((good_vals.index >= begin)&(good_vals.index<=end)))
                        linds = len(inds[0])

                        if linds<=ownn_custPar.train_window_size+ddiff: 
                            continue
                        else:
                            # difference time series ddiff times
                            dat = differencing(good_vals.iloc[inds[0]].DATA_VALUE, ddiff)
                            # Prediction with Ocean_WNN model
                            ts_predict = run_ownn_algorithm(
                                dat,
                                modelOutput=modelOutput,
                                executionType="predict"
                            ) 
                            
                            # MASE of current time span
                            nnans = np.intersect1d(np.where(np.isnan(ts_predict)==False),np.where(np.isnan(dat)==False))
                            all_mase.append(mase(dat[nnans], np.array(ts_predict)[nnans], y_train=y_train))
                    # MASE averaged over all time spans except training intervall
                    mean_mase = np.array(all_mase).mean()
                    new_entry = [st, p, depth,model_name, len(y_train), mean_mase]
                    cross_val_data.append(new_entry)

# fill dataframe
    cross_val_data = pd.DataFrame(
        cross_val_data,
        columns=[
            "Station",
            "Parameter",
            "Depth",
            "model name",
            "len_train",
            "MASE"
        ],
    )
   # save results of model fitting
    cross_val_data.to_csv(resultsfile)

def plot_auc_roc_summary():
    ''' Visualize values for area under ROC curve for all
    time series, both methods (oceanwnn and median) and 
    differencing of time series'''
    # variables related to figure
    plt.rcParams['figure.figsize'][1]=5*cm
    
    savefigpath = os.path.join(evalpath, 'ROC_metrics')
    marker = [['1','v','P','s'], ['2', '^','*',  'D']]
    msize=7
    fillst= 'full'
    colors = ['blue', 'purple']
    titlestub=''
    if ddiff ==2:
        titlestub=', zweifache Differenzenbildung'    
    
    for m in mthds:
        fig = plt.figure(layout=layout)
        counter_p = -1
        all_axes = []
        if m=='Median-Methode' and ddiff!=0:
            continue # median method was not applied on differenced time series
        
        for p in params:
            all_p_aucs = []
            counter_p+=1
            counter_s = -1
            station_axes = []
            for s in stations:
                counter_s+=1
                filestring = get_filestring(s,p,tlims[0], tlims[1])
                filestr = filestring.replace('.csv','_'+ methods+'.csv')
                filestr = filestr.replace('A_Data', os.path.join('D_Model', 'Results', 'Diff_'+str(ddiff)))
                data=pd.read_csv(filestr, index_col='TIME_VALUE')
                data.index = pd.to_datetime(data.index)
                unique_d=list(set(data.Z_LOCATION))
                unique_d.sort(reverse=True)
                for d in unique_d:
                    ts = data[data["Z_LOCATION"]==d] # entries corresponding to depth level d
                    # choose evaluation data
                    # needs be no-NaN values in ad_mm- as well as ad_ownn-column
                    # plus optionally data after training phase of Ocean_WNN prediciton model
                    ts_eval = non_NaN(ts)
                
                    # Area under ROC Curve
                    _, _, _,auc = calc_roc_metrics(ts_eval, mthds[m])
                    l1 = plt.plot(auc, d ,marker[counter_p][counter_s], markersize=msize,
                        fillstyle=fillst, color=colors[counter_p])
                    all_p_aucs.append(auc)
                station_axes.append(l1[0])
            all_axes.append(tuple(station_axes))
            mean_p_auc = np.nanmean(np.array(all_p_aucs)) 
            plt.plot([mean_p_auc]*2, [-39,1],color= colors[counter_p])  
            
                
        plt.title(m+titlestub, fontsize=fs)
                  
    
        
        # set xlims, ylims, labels
    
        plt.ylabel('Wassertiefe [m]')
        plt.xlabel('Fläche unter ROC-Kurve')
        # get current yticklabel locations
        yticklocs = plt.gca().get_yticks()
        yticklabels = plt.gca().get_yticklabels()
        yticklabels = [ll._text.replace(chr(8722), '') for ll in yticklabels]
        plt.gca().set_yticks(yticklocs[1:-2], yticklabels[1:-2])
        plt.ylim((-39,1))
        plt.xlim((-0.01,1.01))
        plt.grid()
        
        if ddiff==2: # legend is only needed once
            all_axes = [(p1,p2) for p1,p2 in zip(all_axes[0], all_axes[1])]
            plt.legend(all_axes,list(stationsdict.values()),
                    handler_map={tuple: HandlerTuple(ndivide=None)})
        
        # customize axes labels etc.
        plt.yticks(fontsize= fs)
        plt.xticks(fontsize=fs)
        savefigstr = os.path.join(savefigpath,m+'_ROC_AUC_summary.png')
        fig.savefig(savefigstr, bbox_inches=bbox_inches)      

def plot_mase_summary():
    ''' Visualize values for mean absoute scales error
    for all time series and differencing of time series'''
    # variables related to figure
    plt.rcParams['figure.figsize'][1]=5*cm
    
    # model fitting results
    model_fit = pd.read_csv(resultsfile)

    savefigpath = os.path.join(evalpath, 'Predictions')
    marker = [['1','v','P','s'], ['2', '^','*',  'D']]
    msize=7
    fillst= 'full'
    colors = ['blue', 'purple']
    titlestub=''
    if ddiff ==2:
        titlestub=', zweifache Differenzenbildung'    
    
    
    fig = plt.figure(layout=layout)
    counter_p = -1
    all_axes = []
    all_mean_mase = [np.nan]*len(model_fit)
    for p in params:
        counter_p+=1
        counter_s = -1
        station_axes = []
        for s in stations:
            counter_s+=1
            filestring = get_filestring(s,p,tlims[0], tlims[1])
            data=read_station_data(filestring)
            
            # only good data
            data = data[data.QF3==2]

            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            for d in unique_d:
                all_mase = []
                depth = abs(d)
                ts = data[data["Z_LOCATION"]==d] # entries corresponding to depth level d

                # find all time spans
                time_spans = find_all_time_spans(time_vec=ts.index, tdel=1)
            
                # ocean_wnn
                modelOutput = os.path.join(savepath,
                                           stationsdict[s],
                                           p, 
                                           str(abs(d))+'m')
                ii = model_fit.index[
                    (model_fit.Station == s)
                    & (model_fit.Parameter == p)
                    & (model_fit.Depth == depth)
                ].to_list()
                train_interval = model_fit.loc[ii, ["start_train", "end_train"]]
                begin = str(train_interval.start_train.values[0])
                end = str(train_interval.end_train.values[0])
                begin = begin[0:8]+'_'+begin[8:11]
                end = end[0:8]+'_'+end[8:11]
                modelOutput=os.path.join(modelOutput,
                                begin+
                                '_'+end)
                tbegin = pd.Timestamp(dt.datetime.strptime(begin, '%Y%m%d_%H'))
                tend =pd.Timestamp(dt.datetime.strptime(end, '%Y%m%d_%H'))
                train_ind = np.where(((ts.index >= tbegin)&(ts.index<=tend)))
                y_train = ts.iloc[train_ind].DATA_VALUE.values
            
                # do all predictions
                for tp in time_spans.index:
                    # define time stamps for current part of
                    # time series
                    end = tp+time_spans[tp]
                    begin=pd.Timestamp(tp.year, tp.month, tp.day, tp.hour, 0)

                    # skip time interval corresponding to training data
                    if end ==tend and begin==tbegin:
                        continue

                    # get existing time stamps for part of time series
                    inds = np.where(((ts.index >= begin)&(ts.index<=end)))
                    linds = len(inds[0])

                    if linds<=ownn_custPar.train_window_size+ddiff: 
                        continue
                    else:
                        # difference time series ddiff times
                        dat = differencing(ts.iloc[inds[0]].DATA_VALUE, ddiff)
                        # Prediction with Ocean_WNN model
                        ts_predict = run_ownn_algorithm(
                            dat,
                            modelOutput=modelOutput,
                            executionType="predict"
                        ) 
                        
                        ts_predict = np.append([np.nan]*ddiff, ts_predict)
                        dat = np.append([np.nan]*ddiff, dat)
                        nnans = np.intersect1d(np.where(np.isnan(ts_predict)==False),np.where(np.isnan(dat)==False))


                        # MASE
                        all_mase.append(mase(dat[nnans], ts_predict[nnans], y_train=y_train))

                mean_mase = np.array(all_mase).mean()
                l1 = plt.plot(mean_mase, d ,marker[counter_p][counter_s], markersize=msize,
                    fillstyle=fillst, color=colors[counter_p])
                all_mean_mase[ii[0]]= mean_mase
            station_axes.append(l1[0])
        all_axes.append(tuple(station_axes))
        mm_mase = np.nanmean(np.array(all_mean_mase)) 
        plt.plot([mm_mase]*2, [-39,1],color= colors[counter_p])  
                
    plt.title(titlestub, fontsize=fs)
                
    # set xlims, ylims, labels
    plt.ylabel('Wassertiefe [m]')
    plt.xlabel('MASE')
    # get current yticklabel locations
    yticklocs = plt.gca().get_yticks()
    yticklabels = plt.gca().get_yticklabels()
    yticklabels = [ll._text.replace(chr(8722), '') for ll in yticklabels]
    plt.gca().set_yticks(yticklocs[1:-2], yticklabels[1:-2])
    plt.ylim((-39,1))
    #plt.xlim((-0.01,1.01))
    plt.grid()
    
    if ddiff==2: # legend is only needed once
        all_axes = [(p1,p2) for p1,p2 in zip(all_axes[0], all_axes[1])]
        plt.legend(all_axes,list(stationsdict.values()),
                handler_map={tuple: HandlerTuple(ndivide=None)})
    
    # customize axes labels etc.
    plt.yticks(fontsize= fs)
    plt.xticks(fontsize=fs)
    savefigstr = os.path.join(savefigpath,'MASE_summary.png')
    fig.savefig(savefigstr, bbox_inches=bbox_inches) 

        
                        
if __name__=='__main__':   
    #summarize_model_fitting()
    #predictions_observations()
    # plot_roc_metrics()
    #plot_auc_roc_summary()
    #plot_mase_summary()
    model_cross_validation()
    #main()
