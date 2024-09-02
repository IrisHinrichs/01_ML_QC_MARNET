# -*- coding: utf-8 -*-
# Module for anomaly detection
import os
import sys
from pathlib import Path
import pandas as pd
ut_path = os.path.join("B_ExpDatAn", "Code") 
def get_path(cur_path = __file__, parents=2, dirname=ut_path):
    # construct  absolute path to directory given by
    # -dirname
    # which is located 
    # -parents
    # levels above __file__
    fpath = os.path.dirname(os.path.abspath(__file__))
    prepath =fpath
    for ll in range(0,parents):
        prepath = Path(prepath).parent.absolute()
    
    if isinstance(prepath,str):
        dirpath= os.path.join(prepath, dirname)
    else:
        dirpath = prepath / dirname
    return str(dirpath)

# set necessary paths
abspath = get_path()
sys.path.insert(0, abspath)
abspath = get_path(parents=2, dirname=os.path.join("C_DataPreProc","Code"))
sys.path.insert(0, abspath)
abspath = get_path(parents=0, dirname =os.path.join("median_method"))
sys.path.insert(0, abspath)
resultspath = get_path(parents=1, dirname =os.path.join("Results"))
from utilities import get_filestring, read_station_data
from pandas import DataFrame as DF
from common_variables import stations, params, tlims
from time_series_statistics import find_all_time_spans  # noqa: E402
from data_preprocessing import piecewise_interpolation
from algorithm_iris import run_mm_algorithm
import numpy as np


def ad_mm(ts):
    time_spans = find_all_time_spans(time_vec=ts.index, tdel=10)

    # initialize dataframe and list of scores
    scores = []

    # iterate over all parts of the time series
    for p in time_spans.index:
        # define time stamps for current part of
        # time series
        end = p+time_spans[p]
        start=pd.Timestamp(p.year, p.month, p.day, p.hour, 0)

        # get existing time stamps for part of time series
        inds = np.where(((ts.index >= start)&(ts.index<=end)))

        # detect anomalies
        linds = len(inds[0])
        # current time series might be too short
        if linds<201: # refine values since it depends on defined neighbourhood for median-method
            scores+=[np.nan]*linds
        else:
            scores+=run_mm_algorithm(ts.iloc[inds]) # rethink method, rethink hour first and last values
                                                    # of time series are handled
    return scores

def main():
    for st in stations:
        count_cols=1
        for p in params:
            filestr = get_filestring(st, p, tlims[0], tlims[1])
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
                scores = ad_mm(ts_interp.DATA_VALUE)
                ts_interp = ts_interp.assign(ad_mm=scores)

                # STEP III: append single time series pieces to dataframe again
                if 'df_results' not in locals():
                    df_results = pd.DataFrame()
                df_results = df_results.append(ts_interp, ignoreIndex=True)
    # Last STEP: Save dataframe with interpolated time series and anomaly score in results
    dummy = os.path.basename(filestr)
    filename = dummy.replace(".csv", "_mm.csv")
    savefile = os.path.join(resultspath,filename)
    df_results.to_csv(savefile, sep=',')      
main()
