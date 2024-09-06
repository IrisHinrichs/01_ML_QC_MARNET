# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:57:32 2024

@author: bd2107
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import datetime as dt
from pandas._libs.tslibs import timedeltas
import matplotlib.pyplot as plt
from B_ExpDatAn.Code.common_variables import (
    layout,
    stationsdict,
    fs,
    paramdict,
    cm,
    bbox_inches,
)
import matplotlib.dates as mdates

# Add absolute path of directory 
# 01_ML_QC_MARNET to sys.path
currentdir=os.path.dirname(__file__)
pathpieces = os.path.split(currentdir)
while pathpieces[-1]!='01_ML_QC_MARNET':
    currentdir= os.path.dirname(currentdir)
    pathpieces = os.path.split(currentdir)
sys.path.insert(0,currentdir)

def get_path(cur_path = __file__, parents=2, dirname="A_Data"):
    # construct  absolute path to directory given by
    # -dirname
    # which is located 
    # -parents
    # levels above __file__
    fpath = os.path.dirname(os.path.abspath(__file__))
    prepath =fpath
    for ll in range(0,parents):
        prepath = Path(prepath).parent.absolute()
    
    dirpath = prepath / dirname
    return str(dirpath)
    
datapath = get_path()



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
    abs_path_file : str
        absolute path and filename .

    '''
    filestring = '_'.join([s.replace(' ','_'),start.strftime('%Y%m%d'), end.strftime('%Y%m%d'), p,'all_.csv'])
    abs_path_file = os.path.join(datapath, filestring)
    return abs_path_file

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
                            start=dt.datetime(2022,11,11,13), end=dt.datetime(2022,11,14,9), dl='all'):
    '''
    Plot MARNET data of a certain parameter on definde depth levels of a certain station in a certain 
    time period and save figure in Figure-directory 

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
    dl: list or str
        list of depth levels that are to be plotted. Default is 'all', meaning
        that all depth levels correponding to station and observed parameter are 
        plotted

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
    #plt.rcdefaults()
    fig = plt.figure(layout=layout)
    plt.rcParams["figure.figsize"][1] = 14*cm 
    start_str = str(start).replace(' ', '_')
    start_str = start_str.replace(':', '-')
    end_str = str(end).replace(' ', '_')
    end_str = end_str.replace(':', '-')
    #define colormap
    if p == 'WT':
        col='blue'
        ylabelstr = '[° C]'
    else:
        col='purple'
        ylabelstr = '[]'
        
    if isinstance(dl, str) and dl =='all':
        # depth levels
        depth_levels = [abs(d) for d in data_tslice.Z_LOCATION.unique()]
        depth_levels.sort(reverse=False)
        depthstr = dl
    else:
        depth_levels = dl
        depthstr = '_'.join([str(d) for d in dl])
        
    savefigpath = '../Figures/'+station.replace(' ', '_')+\
                    '/_'+p+\
                    '_'+start_str+'_'+end_str+\
                    '_'+depthstr+'.png'
    counter_d = 1
    for d in depth_levels:
        plt.subplot(len(depth_levels), 1, counter_d)
        
        # anomlies Val 1
        data_d = data_tslice[data_tslice.Z_LOCATION==d*-1]
        plt.plot(data_d[data_d.QF1.isin([3,4])].DATA_VALUE, 'ko',alpha=0.2, markersize=7)
        
        # anomalies Val 2
        plt.plot(data_d[data_d.QF3.isin([3,4])].DATA_VALUE, 'ro',alpha=0.2, markersize=7)
        
        # DATA_VALUES
        plt.plot(data_tslice[data_tslice.Z_LOCATION==d*-1].DATA_VALUE, 'o',
                 color=col,markersize=2)
        
        plt.annotate(str(abs(d))+' m', xy=(0.05, 0.05), xycoords='axes fraction')
        
        plt.gca().xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(plt.gca().xaxis.get_major_locator()))
        plt.grid()
        if d!=max(depth_levels):
            plt.gca().set_xticklabels([])
        plt.xlim(xlim) 
        
        pstring = paramdict[p].replace(' '+ylabelstr, '')
        titlestring = stationsdict[station].replace('[° C]', '')+', '+pstring+', '+\
                       start.strftime('%d.%m.%Y %H:%M:%S')+'-'+end.strftime('%d.%m.%Y %H:%M:%S')
        if counter_d==1:
            plt.title(titlestring, fontsize=fs, wrap=True)
        plt.ylabel(ylabelstr)
        plt.ylim(ylim) 
        counter_d+=1       
        fig.savefig(savefigpath, bbox_inches=bbox_inches)
        
def plot_anomaly_legend():
    yvals = [0.3,0.5,0.7]
    fig = plt.figure(layout=layout)
    plt.rcParams["figure.figsize"][0] = 7*cm 
    plt.rcParams["figure.figsize"][1] = 3*cm 
    
    #define colormap
    colT='blue'
    colS='purple'
   
    # anomlies Val 1
    plt.plot([0.2, 0.3],[yvals[2]]*2, 'ko',alpha=0.2, markersize=7)
    plt.plot([0.2, 0.3],[yvals[1]]*2, 'ko',alpha=0.2, markersize=7)
    
    # anomalies Val 2
    plt.plot([0.2,0.3],[yvals[1]]*2, 'ro',alpha=0.2, markersize=7)
    plt.plot([0.2,0.3],[yvals[0]]*2, 'ro',alpha=0.2, markersize=7)
    
    # DATA_VALUES
    plt.plot([0.2]*3,yvals, 'o',
             color=colT,markersize=2)
    plt.plot([0.3]*3,yvals, 'o',
             color=colS,markersize=2)

    plt.annotate('T', xy=(0.2, 0.9),xytext=(0.18, 0.85))#, xycoords='axes fraction')
    plt.annotate('S', xy=(0.3, 0.9),xytext=(0.28, 0.85))#, xycoords='axes fraction')
    plt.annotate('markiert in Validierungsstufe', xy=(0.4, 0.9), xytext=(0.38, 0.85))#, xycoords='axes fraction')
    plt.annotate('1', xy=(0.4, 0.7), xytext=(0.38, 0.65))#, xycoords='axes fraction')
    plt.annotate('1 und 3', xy=(0.4, 0.5), xytext=(0.38, 0.45))#, xycoords='axes fraction')
    plt.annotate('3', xy=(0.4, 0.3), xytext=(0.38, 0.25))#, xycoords='axes fraction')
    plt.xlim([0.16,1.9])
    plt.ylim([0.2,1])
    plt.axis('off')
    fig.savefig('../Figures(legend_anomalies.png')

def read_json_file(method = 'median_method'):
     #read json file and append necessary attributes
    abspath = os.path.join(sys.path[0], 'D_Model', 'Code', method)
    jsonfile= os.path.join(abspath ,"manifest.json")
    f = open(jsonfile)
    jsondict = json.load(f)
    f.close()
    jsondict['executionType']='execute'
    return jsondict
   


    