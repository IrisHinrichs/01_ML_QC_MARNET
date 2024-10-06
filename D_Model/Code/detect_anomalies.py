# -*- coding: utf-8 -*-
# Module for anomaly detection
import os
import pandas as pd
import sys

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
from B_ExpDatAn.Code.time_series_statistics import find_all_time_spans  # noqa: E402
from C_DataPreProc.Code.data_preprocessing import piecewise_interpolation, differencing # noqa: E402
from D_Model.Code.median_method.algorithm_iris import run_mm_algorithm, mm_neigh  # noqa: E402
from D_Model.Code.ocean_wnn.algorithm_iris import run_ownn_algorithm  # noqa: E402
from D_Model.Code.ocean_wnn.algorithm_iris import CustomParameters as ownn_custPar  # noqa: E402
import numpy as np  # noqa: E402

# differencing parameter
ddiff = 2

# where to save results
resultspath = os.path.join(currentdir, "D_Model", "Results", "Diff_" + str(ddiff))
if not os.path.isdir(resultspath):
    os.makedirs(resultspath) 

# initiate string for used methods
methods = '_mm_ownn_' 

def ad_mm(ts):
    time_spans = find_all_time_spans(time_vec=ts.index, tdel=10)

    # initialize scores as array of nans
    scores = np.zeros((len(ts)))
    scores[:]=np.nan

    # iterate over all parts of the time series
    for p in time_spans.index:
        # define time stamps for current part of
        # time series
        end = p+time_spans[p]
        begin=pd.Timestamp(p.year, p.month, p.day, p.hour, 0)

        # get existing time stamps for part of time series
        inds = np.where(((ts.index >= begin)&(ts.index<=end)))

        # detect anomalies
        linds = len(inds[0])
        # current time series might be too short
        if linds>=2*mm_neigh+1: # refine values since it depends on defined neighbourhood for median-method
            scores[inds]=run_mm_algorithm(ts.iloc[inds]) # rethink method, rethink how first and last values
                                                    # of time series are handled
    return scores

def ad_ownn(ts,ts_interp,modelOutput, ddiff=ddiff):
    good_vals = ts[ts['QF3']==2] # only use good values for training based on original times series
    time_spans = find_all_time_spans(time_vec=good_vals.index, tdel=1)

    # # sort time spans by length, begining with longest
    time_spans_sort = time_spans.sort_values(ascending=False)

    # training should be done with longest time series part
    p = time_spans_sort.index[0]
    
    # define time stamps for current part of
    # time series
    end = p+time_spans[p]
    begin=pd.Timestamp(p.year, p.month, p.day, p.hour, 0)

    # get existing time stamps for part of time series
    inds = np.where(((good_vals.index >= begin)&(good_vals.index<=end)))

    # detect anomalies
    linds = len(inds[0])

    log_train = os.path.join(modelOutput, 'log_training.txt')
    modelOutput=os.path.join(modelOutput,
                                begin.strftime('%Y%m%d_%H')+
                                '_'+end.strftime('%Y%m%d_%H'))
    
    data = differencing(good_vals.iloc[inds].DATA_VALUE, ddiff)
    run_ownn_algorithm(
        data,
        modelOutput=modelOutput,
        executionType="train",
        logfile=log_train,
    )

    # initialize scores as array of nans
    scores = np.zeros((len(ts_interp)))
    scores[:]=np.nan

    # iterate over all parts of the interpolated time series
    time_spans = find_all_time_spans(time_vec=ts_interp.index, tdel=10)
    for p in time_spans.index:
        # define time stamps for current part of
        # time series
        end = p+time_spans[p]
        begin=pd.Timestamp(p.year, p.month, p.day, p.hour, 0)

        # get existing time stamps for part of time series
        inds = np.where(((ts_interp.index >= begin)&(ts_interp.index<=end)))

        # detect anomalies
        linds = len(inds[0])
        
        # current time series might be too short
        if linds<ownn_custPar.train_window_size+ddiff: 
            continue # part of time series is too short
        else:
            data = differencing(ts_interp.iloc[inds].DATA_VALUE, ddiff)
            scores[inds[0][ddiff::]]=run_ownn_algorithm(data, modelOutput=modelOutput, executionType='execute')             
    return scores

def main():
    # differencing parameter
    ddiff = 2
    for st in stations:
        for p in params:
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            # replace following line with something like 
            # data = read_interp_data(filestr)
            data=read_station_data(filestr=filestr)
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            for d in unique_d:
                ts = data[data["Z_LOCATION"]==d] # entries corresponding to depth level d
                # STEP I: piecewise interpolation of all time series
                ts_interp = piecewise_interpolation(ts)
            
                # STEP II: piecewise anomaly detection, 
                # several functions can be called
                # append scores to dataframe
                
                # median method
                scores = ad_mm(ts_interp.DATA_VALUE)
                ts_interp = ts_interp.assign(ad_mm=scores)
                

                # ocean_wnn
                modelOutputDir = os.path.join(currentdir,
                                              'D_Model',
                                           'Trained_Models', 
                                           'Ocean_WNN',
                                           'Diff_'+str(ddiff), 
                                           stationsdict[st],
                                           p, 
                                           str(abs(d))+'m')
                if not os.path.isdir(modelOutputDir):
                    os.makedirs(modelOutputDir)
                # Input for ocean_wnn is original time series
                scores = ad_ownn(ts,ts_interp, modelOutput=modelOutputDir, ddiff=ddiff)
                ts_interp = ts_interp.assign(ad_ownn=scores)

                # STEP III: Concatenate single time series pieces to dataframe again
                if 'df_results' not in locals():
                    df_results = pd.DataFrame()
               
                df_results = pd.concat([df_results,ts_interp])
            # Last STEP: Save dataframe with interpolated time series and anomaly score in results
            dummy = os.path.basename(filestr)
            filename = dummy.replace(".csv","_"+methods+".csv")
            savefile = os.path.join(resultspath,filename)
            df_results = df_results.sort_values(by = ['TIME_VALUE', 'Z_LOCATION'], ascending = [True, False])
            df_results.to_csv(savefile, sep=',', na_rep= "NaN") 
            del df_results    
if __name__ == "__main__":
    main()
