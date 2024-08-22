# -*- coding: utf-8 -*-
import sys
import os
abspath = os.path.abspath("B_ExpDatAn/Code")
sys.path.insert(0, abspath)
from time_series_statistics import find_all_time_spans
import datetime as dt
import pandas as pd

def piecewise_interpolation(ts, gap=10):
    '''
    Piecewise linear interpolation of MARNET time series data.
    Start and end point of time series' pieces are defined by 
    parameter 'gap': temporal gaps less or equal 'gap' are ignored

    Parameters
    ----------
    ts : pandas.core.DataFrame
        Time series data consisting of the datestamps as index and 
        the actual data values. 

    gap : int
        Integer defining the maximum length  of temporal gaps in hours
        that are to be filled by linear interpolation. Default is 10 hours. 

    Returns
    -------
    ts_interp : pandas.core.DataFrame
        Interpolated time series. Data gaps less or equal 'gap' are
        interpolated, others are void.

    '''
    time_spans = find_all_time_spans(ts.index, gap)
    for p in time_spans.index[0:-2]:
        end = p+dt.timedelta(hours=time_spans[p])
        start=dt.datetime(p.year, p.month, p.day, p.hour, 0)
        new_p_index =  pd.date_range(start, end, freq='h')
