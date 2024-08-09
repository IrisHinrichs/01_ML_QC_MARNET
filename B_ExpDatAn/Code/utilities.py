# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:57:32 2024

@author: bd2107
"""
import pandas as pd
import datetime as dt
from pandas._libs.tslibs import timedeltas
import matplotlib.pyplot as plt
from common_variables import datapath, layout
import matplotlib as mtpl
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates



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

def convert_duration_string(dur_raw='263 days 21:00:00'):
    '''
    Convert string stating temporal duration to integer of hours

    Parameters
    ----------
    dur_raw : str, 
        String stating the days, hours, minutes and seconds of the duration. The default is '263 days 21:00:00'.

    Returns
    -------
    dur_hours : int
        Duration in hours.

    '''
    if isinstance(dur_raw,str):
        dur_vals = dur_raw.split(' ')
        days = int(dur_vals[0])
        hours = int(dur_vals[2][0:2])
    elif isinstance(dur_raw, timedeltas.Timedelta):
        days=dur_raw.days
        hours=dur_raw.seconds/3600
            
   
    dur_hours =days*24+hours
    return dur_hours

def plot_all_dl_time_series(station='North Sea Buoy III', p='WT', 
                            start=dt.datetime(2022,11,11,13), end=dt.datetime(2022,11,14,9)):
    '''
    Plot MARNET data of a certain parameter on all depth levels of a certain station in a certain 
    time period 

    Parameters
    ----------
    station : str
        Name of the station. Default is 'North Sea Buoy III' 
    p : str
        Observed parameter. Default is 'WT', water temperature
    start : datetime.datetime
        Beginning of time period.
    end : datetime.datetime
        End of time period.

    Returns
    -------
    None.

    '''
    
    # read data
    filestring = get_filestring(station, p)
    data = read_station_data(datapath+filestring)
    
    # temporal slice
    mask = (data.index>=start) & (data.index<=end)
    data_tslice = data.loc[mask]
   
    # plot parameters
    dy= 0.5
    ylim = [data_tslice.DATA_VALUE.min()-dy, data_tslice.DATA_VALUE.max()+dy]
    xlim = [start, end]
    plt.rcdefaults()
    fig = plt.figure(layout=layout)
    
    # depth levels
    depth_levels = [abs(d) for d in data.Z_LOCATION.unique()]
    depth_levels.sort(reverse=False)
    counter_d = 1
    for d in depth_levels:
        plt.subplot(len(depth_levels), 1, counter_d)
        plt.plot(data_tslice[data_tslice.Z_LOCATION==d*-1].DATA_VALUE, 'o', markersize=2)
        counter_d+=1
        plt.title(str(abs(d)))
        
        plt.gca().xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(plt.gca().xaxis.get_major_locator()))
        plt.grid()
        if d!=max(depth_levels):
            plt.gca().set_xticklabels([])
        plt.xlim(xlim)   
        plt.ylim(ylim)         
