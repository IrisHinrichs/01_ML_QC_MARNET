# -*- coding: utf-8 -*-

#import datetime as dt
import pandas as pd
import numpy as np
import sys
import os

abspath = os.path.abspath("B_ExpDatAn/Code")
sys.path.insert(0, abspath)
from time_series_statistics import find_all_time_spans  # noqa: E402


def piecewise_interpolation(ts, gap=10):
    '''
    Piecewise linear interpolation of MARNET time series data.
    Start and end point of parts of the time series are defined by 
    parameter 'gap': temporal gaps less or equal 'gap' are ignored, 
    only those greater than 'gap' mark the end and the beginning of
    a piece of the time series. 

    Parameters
    ----------
    ts : pandas.core.DataFrame
        Time series data consisting of the datestamps as index, 
        the depth levels and the  actual data values plus quality flags . 

    gap : int
        Integer defining the maximum length  of temporal gaps in hours
        that are to be filled by linear interpolation. Default is 10 hours. 

    Returns
    -------
    ts_interp : pandas.core.DataFrame
        Interpolated time series. Data gaps less or equal 'gap' are
        interpolated, others are void. All columns of ts are in ts_interp
        plus an additional column, INTERP, showing if corresponding values
        are interpolated (INTERP=True) or the original ones (INTERP=False)

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

    # iterate over all parts of the time series
    for p in time_spans.index:
        # define time stamps for current part of
        # time series
        end = p+time_spans[p]
        start=pd.Timestamp(p.year, p.month, p.day, p.hour, 0)
        new_p_index =  pd.date_range(start, end, freq='h')

        # get existing time stamps for part of time series
        old_inds = np.where(((ts.index >= start)&(ts.index<=end)))
        old_p_index=ts.index[old_inds[0]]

        # concatenate all new time stamps to a datetimeindex
        if 'index_col' in locals(): # variable for index exists already
            index_col=index_col.append(new_p_index)  # noqa: F821
        else: # variable for index is created
            index_col= new_p_index

        # Interpolate values for all other columns of data frame
        for c in list(ts.columns):
            exec(c+"+=list(np.interp(new_p_index, ts.index, ts."+c+"))")
        
        # Create information about interpolated values of
        # current piece of time series and concat it to single list
        interp_bool = [True]*len(new_p_index)
        int_inds = [list(new_p_index).index(oi) for oi in list(old_p_index) if oi in list(new_p_index)]
        if len(int_inds)!=0:
            for i in int_inds:
                interp_bool[i]=False 
        INTERP+=interp_bool

    # fill data frame
    ts_interp['TIME_VALUE']=index_col
    ts_interp=ts_interp.set_index(keys='TIME_VALUE', drop=True) # set DatetimeIndex as index
    
    for c in ts_interp.columns:
        exec("ts_interp."+c+"="+c)
    return ts_interp