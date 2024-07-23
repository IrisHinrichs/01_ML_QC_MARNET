# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:57:32 2024

@author: bd2107
"""
import pandas as pd
import datetime as dt


def get_filestring(s='North Sea Buoy II',p='WT',start=dt.datetime(2020,1,1), end=dt.datetime(2024,6,30)):
    '''
    Make string of file name based on stationname, parameter name, start date
    and end date of data

    Parameters
    ----------
    s : str
        station name. Default is 'Nordsee II'
    p : str
        parameter name. Default is 'WT'
    start : datetime.datetime
        start of time series, default is dt.datetime(2020,1,1).
    end : datetime.datetime
        end of time series, default is end=dt.datetime(2024,6,30).

    Returns
    -------
    filestring : str
        file name .

    '''
    filestring = '_'.join([s.replace(' ','_'),start.strftime('%Y%m%d'), end.strftime('%Y%m%d'), p,'all_.csv'])
    return filestring

def read_station_data(filestr):
    usecols=['TIME_VALUE','Z_LOCATION','DATA_VALUE', 'QF1', 'QF3']
    data = pd.read_csv(filestr, index_col='TIME_VALUE', usecols=usecols)
    data.index = pd.to_datetime(data.index)
    return data

def diff_time_vec(dtindex):
    '''
    Differences between values in a series of time stamps
    Parameters
    ----------
    dtindex : pandas.core.indexes.datetimes.DatetimeIndex
        Datetime index containing the time stamps of the time series

    Returns
    -------
    diff_ts : pandas.core.series.Series
        Series containing the temporal differences in hours and the 
        corresponding index as time stamps

    '''
    tseries = pd.Series(dtindex, index=dtindex)
    diff_ts = tseries.diff()
    diff_ts = diff_ts.dt.days*24+diff_ts.dt.seconds/3600
    return diff_ts