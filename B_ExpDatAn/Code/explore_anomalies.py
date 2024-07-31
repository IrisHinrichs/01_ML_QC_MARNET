# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:32:37 2024

@author: bd2107
"""

import datetime as dt
import time
import numpy as np
import pandas as pd # version 2.1.4 auf dem Laptop zu Hause, 1.4.4 bei der Arbeit
from utilities import read_station_data, get_filestring, diff_time_vec, convert_duration_string
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import matplotlib as mtpl

from common_variables import datapath, layout, cm, stations, stationsdict,\
    params, paramdict, tlims, fs, fontdict, bbox_inches
    
import os

# define figure height
#plt.rcParams['figure.figsize'][1]=14*cm
#fig = plt.figure(layout=layout)

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

def anomaly_exploration():
    '''
    Explore anomalies: 
        - how many point anomalies, how many sequential?
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
    
    for st in stations:
        # path for data storage
        savepath = '../Results/'+'_'.join(st.split(' '))+'/Anomalies/'
        
        time_series_without_anomalies=[]
        savestring_no_anomalies = savepath+'no_anomalies.txt'
        for p in params:
            filestr = get_filestring(st, p)
            print(filestr)
            data=read_station_data(filestr=datapath+filestr)
    
            # Find data with quality flag set to 4 or 3 in validation level 3
            val3_qf3_or_4 = data[data.QF3.isin([3,4])]
            
            # number of anomalies grouped by depth levels
            n_anom= val3_qf3_or_4.groupby('Z_LOCATION').size()
            
            # unique depth levels of current station
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            
            for d in unique_d:
                 
                # default values of status of missing value before and after
                # anomaly, 1 corresponds to "start of anomaly comes after a missing value"
                # and "end of anomaly comes before missing value" respectively
                missbef=1
                missaft=1
                
                danom = val3_qf3_or_4[val3_qf3_or_4['Z_LOCATION']==d] # anomalies on current depth level d
                if len(danom)==0: # current time series has no anomalies
                    time_series_without_anomalies.append('_'.join([st.replace(' ', '_'),p, str(abs(d))]))
                    continue # to next time series
                tstamps = pd.Series(data[data['Z_LOCATION']==d].index) # time stamps of all observations on current depth level
                
                savestring =savepath +p+'_'+str(abs(d))+'_anomalies_'+'_'.join(st.split(' '))+'.csv'
                
                # timestamp of first anomalie in current time series
                start_anom=danom.index[0]
                
                time_stamp_before =start_anom-dt.timedelta(hours=1)
                time_stamp_after = start_anom+dt.timedelta(hours=1)

                # check for missing values
                if tstamps.isin([time_stamp_before]).any():
                    missbef=0
                if tstamps.isin([time_stamp_after]).any():
                    missaft=0
                
               
                if len(danom)==1: # only a single anomaly in current time series
                    # initialize DataFrame for data about the anomalies
                    anomalies=pd.DataFrame(columns= ['LENGTH','MISSING_BEFORE', 'MISSING_AFTER'], index=danom.index)
                    
                    # save data in dataframe 
                    anomalies.loc[danom.index[0]] = [1, #LENGTH
                                              missbef,# MISSING_BEFORE
                                              missaft]# MISSING_AFTER
                    continue
                
                # differences of the time stamps of the anomalie
                # rethink time_gaps! The way they are calculated now, they also take into account the observational 
                # as timedelta between one anomalie and the other
                time_gaps = diff_time_vec(danom.index).dropna()
                
                # time gaps greater than 1 mark beginning of anomaly
                # turn them into string of the timestamp
                start_of_anomalies = [s for s in time_gaps[time_gaps >1].index]
                
                # initialize DataFrame for saving data about the anomalies
                # here: index is different
                other_index = [danom.index[0]]
                [other_index.append(ind) for ind in start_of_anomalies]
                anomalies=pd.DataFrame(columns= ['LENGTH','MISSING_BEFORE', 'MISSING_AFTER'], index=other_index)
                
                # initiate index for time_gaps 
                i=0
                if time_gaps.iloc[i]>1: # first datestamp of anomaly time series (danom) marks an anomaly of length 1
                    time_stamp_after = start_anom+dt.timedelta(hours=1)

                    # check for missing values
                    if tstamps.isin([time_stamp_before]).any() and time_stamp_before>tlims[0]:
                        missbef=0
                    if tstamps.isin([time_stamp_after]).any() and time_stamp_before>tlims[1]:
                        missaft=0
                        
                    # save data in dataframe 
                    anomalies.loc[danom.index[0]] = [1, #LENGTH
                                              missbef,# MISSING_BEFORE
                                              missaft]# MISSING_AFTER
                    i+=1
                else: # first datestamp of anomaly time series (danom) marks an anomaly of length >1
                    # find out length of anomaly
                    lgth=1
                    while  i<len(time_gaps) and time_gaps.iloc[i]<=1:
                        i+=1
                        lgth+=1
                    i+=1    
                    # check for missing values
                    time_stamp_after = start_anom+dt.timedelta(hours=lgth)
                    if tstamps.isin([time_stamp_before]).any() and time_stamp_before>tlims[0]:
                        missbef=0
                    if tstamps.isin([time_stamp_after]).any() and time_stamp_before>tlims[1]:
                        missaft=0
                        
                    # save data in dataframe
                    anomalies.loc[start_anom] = [lgth, #LENGTH
                                              missbef,# MISSING_BEFORE
                                              missaft]# MISSING_AFTER
                    
                    
                # all in-between anomalies
                start_anom_old = start_anom
                for sta in start_of_anomalies:
                    start_anom = sta
                    print(start_anom)
                    
                    # time stamp before anomaly
                    time_stamp_before =start_anom-dt.timedelta(hours=1)
                    # find out length of anomaly
                    lgth=1
                    while  i<len(time_gaps) and time_gaps.iloc[i]<=1:
                        i+=1
                        lgth+=1
                    
                    # check for missing values
                    time_stamp_after = start_anom+dt.timedelta(hours=lgth)
                    if tstamps.isin([time_stamp_before]).any() and time_stamp_before>tlims[0]:
                        missbef=0
                    if tstamps.isin([time_stamp_after]).any() and time_stamp_before>tlims[1]:
                        missaft=0
                        
                    # save data in dataframe
                    anomalies.loc[start_anom] = [lgth, #LENGTH
                                              missbef,# MISSING_BEFORE
                                              missaft]# MISSING_AFTER
    
                    start_anom_old = start_anom
                # save data
                anomalies.to_csv(savestring, sep=';', index_label='time_stamp')
        
        if len(time_series_without_anomalies)>0:
            with open(savestring_no_anomalies, 'w') as outfile:
                outfile.write('\n'.join(i for i in time_series_without_anomalies))
            outfile.close()
    return

anomaly_exploration()
