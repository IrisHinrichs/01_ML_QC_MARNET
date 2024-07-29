# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:32:37 2024

@author: bd2107
"""

import datetime as dt
import time
import numpy as np
import pandas as pd
from utilities import read_station_data, get_filestring, diff_time_vec
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from common_variables import datapath, layout, cm, stations, stationsdict,\
    params, paramdict, tlims, fs, fontdict, bbox_inches
    
import os

def statistics(stname='North Sea Buoy II', paracode='WT', dlevels='all', 
            start=tlims[0], end=tlims[1]):
    '''
    Gain statistics of parameter time series and print them.

    Parameters
    ----------
    stname : str, optional
        Name of the station. 
        The default is 'North Sea Buoy II'.
    paracode : str, optional
        Code of observed parameter. The default is 'WT'.
    dlevels : str of list, optional
        string or tuple of depth levels. The default is 'all'.
    start : datetime.datetime, optional
        Start of time series. The default is dt.datetime(2020,1,1).
    end : datetime.datetime, optional
        End of time series. The default is dt.datetime(2024,4,30).

    Returns
    -------
    None.

    '''
    
    # number of possible quality flags
    n_qfs = 4
     # %% define path and filename
    strings = get_filestring(stname)
    file = datapath+strings
    df = read_station_data(file)
    
   
    n_dp=len(df)
    print(stname+', '+paracode+'\n'+
          '----------------------------------\n'+
         'Anzahl Datenpunkte ('+start.strftime('%Y%m%d')+'-'+end.strftime('%Y%m%d')+'): '+str(n_dp)+'\n')
    for flag in range(0,n_qfs+1):
        absval=df[df.QF1==flag].QF1.count()
        frac = absval/n_dp
        print('Validation level 1, quality flag '+str(flag)+': '+str(absval)+' ({:.4f}'.format(frac)+')')
        absval=df[df.QF3==flag].QF3.count()
        frac = absval/n_dp
        print('Validation level 3, quality flag '+str(flag)+': '+str(absval)+' ({:.4f}'.format(frac)+')')
    print('----------------------------------\n')
    return

def anomaly_exploration(stname='North Sea Buoy II', paracode='WT', dlevels='all', 
            start=tlims[0], end=tlims[1]):
    '''
    Explore anomalies: 
        - how many point anomalies, how many sequentiel?
        - anomaly after missing value?
        - anomalies on all depth levels?
        - how often difference in qf between val 1 and val 3?.

    Parameters
    ----------
    stname : str, optional
        Name of the station. 
        The default is 'North Sea Buoy II'.
    paracode : str, optional
        Code of observed parameter. The default is 'WT'.
    dlevels : str of list, optional
        string or tuple of depth levels. The default is 'all'.
    start : datetime.datetime, optional
        Start of time series. The default is dt.datetime(2020,1,1).
    end : datetime.datetime, optional
        End of time series. The default is dt.datetime(2024,4,30).

    Returns
    -------
    None.

    '''
    
    # number of possible quality flags
    n_qfs = 4
     # %% define path and filename
    strings = get_filestring(stname)
    file = datapath+strings
    df = read_station_data(file)
    
   
    n_dp=len(df)
    
    # Find data where quality flag in validation level 3 is set to 4 or 3
    val3_qf4 = df[df.QF3==4]+df[df.QF3==3]
    print(stname+', '+paracode+'\n'+
          '----------------------------------\n'+
         'Anzahl Datenpunkte ('+start.strftime('%Y%m%d')+'-'+end.strftime('%Y%m%d')+'): '+str(n_dp)+'\n')
    for flag in range(0,n_qfs+1):
        absval=df[df.QF1==flag].QF1.count()
        frac = absval/n_dp
        print('Validation level 1, quality flag '+str(flag)+': '+str(absval)+' ({:.4f}'.format(frac)+')')
        absval=df[df.QF3==flag].QF3.count()
        frac = absval/n_dp
        print('Validation level 3, quality flag '+str(flag)+': '+str(absval)+' ({:.4f}'.format(frac)+')')
    print('----------------------------------\n')
    return
