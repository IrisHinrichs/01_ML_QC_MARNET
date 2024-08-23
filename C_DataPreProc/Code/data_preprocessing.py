# -*- coding: utf-8 -*-

import datetime as dt
import pandas as pd
import numpy as np
import sys
import os

abspath = os.path.abspath("B_ExpDatAn/Code")
sys.path.insert(0, abspath)
from time_series_statistics import find_all_time_spans


def piecewise_interpolation(ts, gap=10):
    '''
    Piecewise linear interpolation of MARNET time series data.
    Start and end point of time series' pieces are defined by 
    parameter 'gap': temporal gaps less or equal 'gap' are ignored

    Parameters
    ----------
    ts : pandas.core.DataFrame
        Time series data consisting of the datestamps as index and 
        the actual data values plus quality flags. 

    gap : int
        Integer defining the maximum length  of temporal gaps in hours
        that are to be filled by linear interpolation. Default is 10 hours. 

    Returns
    -------
    ts_interp : pandas.core.DataFrame
        Interpolated time series. Data gaps less or equal 'gap' are
        interpolated, others are void.

    '''
    # find all time spans that only have temporal gaps 
    # less or equal the integer stated by gap
    time_spans = find_all_time_spans(ts.index, gap)

    # initialize data frame for interpolated values
    ts_interp = pd.DataFrame(columns=ts.columns.to_list()+['TIME_VALUE', 'INTERP'])

    #initialize empty lists for single columns 
    for c in ts.columns:
        exec(c+"=[]")
    INTERP = []
    for p in time_spans.index[0:-2]: # iterate over all parts of the time series
        end = p+time_spans[p]
        start=pd.Timestamp(p.year, p.month, p.day, p.hour, 0)
        new_p_index =  pd.date_range(start, end, freq='h')
        old_inds = np.where(((ts.index >= start)&(ts.index<=end))==True)
        old_p_index=ts.index[old_inds[0]]
        if 'index_col' in locals():
            index_col=index_col.append(new_p_index)
        else:
            index_col= new_p_index
        for c in list(ts.columns):
            exec(c+"+=list(np.interp(new_p_index, ts.index, ts."+c+"))")
        interp_bool = [True]*len(new_p_index)
        int_inds = [list(new_p_index).index(oi) for oi in list(old_p_index) if oi in list(new_p_index)]
        if len(int_inds)!=0:
            for i in int_inds:
                interp_bool[i]=False 
        INTERP+=interp_bool
    # fill data frame
    ts_interp['TIME_VALUE']=index_col
    ts_interp=ts_interp.set_index(keys='TIME_VALUE', drop=True)
    
    for c in ts_interp.columns:
        exec("ts_interp."+c+"="+c)
    return ts_interp
from utilities import get_filestring, read_station_data
from common_variables import datapath
filestr = get_filestring()
data = read_station_data(datapath+filestr)
ts = data[data.Z_LOCATION==-3]
ts_interp = piecewise_interpolation(ts)