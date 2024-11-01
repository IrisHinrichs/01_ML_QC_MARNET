# -*- coding: utf-8 -*-

#import datetime as dt
import pandas as pd
import numpy as np
import os
import sys

# Add absolute path of directory 
# 01_ML_QC_MARNET to sys.path
currentdir=os.path.dirname(__file__)
pathpieces = os.path.split(currentdir)
while pathpieces[-1]!='01_ML_QC_MARNET':
    currentdir= os.path.dirname(currentdir)
    pathpieces = os.path.split(currentdir)
sys.path.insert(0,currentdir)

from B_ExpDatAn.Code.time_series_statistics import find_all_time_spans  # noqa: E402


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

    # set data types of single columns
    convert_dict = dict(ts.dtypes)
    convert_dict |= {'TIME_VALUE': ts.index.dtype, 'INTERP': 'boolean'}
    ts_interp = ts_interp.astype(convert_dict)


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
            coldata = ts[c][:]
            vals=np.interp(new_p_index, ts.index, coldata)
            vals = vals.astype(convert_dict[c])
            exec(c+"+=list(vals)")
        
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
 
def differencing(ts: pd.Series, n: int) -> np.array:
    '''n-times Differencing of time series in order to make time series stationary''' 
    for i in range(0,n):
        ts = np.diff(ts)
    if isinstance(ts, pd.Series):
        ts=ts.to_numpy().reshape(-1,1)
    else:
        ts = ts.reshape(-1,1)
    return ts

def reverse_diff(first_value: np.array, diff_ts: np.array) -> pd.Series:
    '''Reverse differencing of time series'''
    rev_diffts=np.cumsum(np.append(first_value,diff_ts))
    return rev_diffts

