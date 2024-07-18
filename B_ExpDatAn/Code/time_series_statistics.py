# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:33:40 2024

@author: bd2107
"""
import datetime as dt
import time
import numpy as np
import pandas as pd
from utilities import read_station_data

datapath = '../../A_Data/'
stations = ['Fehmarn Belt Buoy', 'Kiel Lighthouse', 
            'North Sea Buoy II', 'North Sea Buoy III']
stationsdict = {'Fehmarn Belt Buoy': 'Fehmarn', 
                'Kiel Lighthouse': "Leuchtturm Kiel", 
                'North Sea Buoy II': 'Nordsee II', 
                'North Sea Buoy III': 'Nordsee III'}
pcodes=['WT','SZ']

def statistics(stname='North Sea Buoy II', paracode='WT', dlevels='all', 
            start=dt.datetime(2020,1,1), end=dt.datetime(2024,6,30)):
    '''
    Gain statistics of parameter time series and save them to .csv file

    Parameters
    ----------
    stname : str, optional
        Name of the station to be plotted, as in table column insituadm.time_series_point.name. 
        The default is 'North Sea Buoy II'.
    paracode : str, optional
        Code of observed parameter, as in table column insituadm.parameter.code The default is 'WT'.
    dlevels : str of list, optional
        string or tuple of depth levels. The default is 'all'.
    start : datetime.datetime, optional
        Start of time series. The default is dt.datetime(2021,1,1).
    end : datetime.datetime, optional
        End of time series. The default is dt.datetime(2024,4,30).

    Returns
    -------
    None.

    '''
    
    # number of possible quality flags
    n_qfs = 4
     # %% define path and filename
    strings = '_'.join([stname.replace(' ','_'),start.strftime('%Y%m%d'), end.strftime('%Y%m%d'), paracode, str(dlevels), '.csv'])
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


for s in stations:
    for p in pcodes:
        #plot_ts(stname=s, paracode=p)   
        #scatter_TS()
        statistics(stname=s, paracode=p)
    
