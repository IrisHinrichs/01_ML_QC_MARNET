# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:57:32 2024

@author: bd2107
"""
import pandas as pd
import datetime as dt


def get_filestring(s='Nordsee II',p='WT',start=dt.datetime(2020,1,1), end=dt.datetime(2024,6,30)):
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