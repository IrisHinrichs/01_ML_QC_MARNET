# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:32:37 2024

@author: bd2107
"""
import math
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
            
            # unique depth levels of current station
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            
            for d in unique_d:
                
                danom = val3_qf3_or_4[val3_qf3_or_4['Z_LOCATION']==d] # anomalies on current depth level d
                if len(danom)==0: # current time series has no anomalies
                    time_series_without_anomalies.append('_'.join([st.replace(' ', '_'),p, str(abs(d))]))
                    continue # to next time series
                tstamps = pd.Series(data[data['Z_LOCATION']==d].index) # time stamps of all observations on current depth level
                
                savestring =savepath +p+'_'+str(abs(d))+'_anomalies_'+'_'.join(st.split(' '))+'.csv'
                
                # timestamp of first anomalie in current time series
                start_anom=danom.index[0]
               
                if len(danom)==1: # only a single anomaly in current time series
                    # initialize DataFrame for data about the anomalies
                    anomalies=pd.DataFrame(columns= ['LENGTH','N_MISSING_BEFORE', 'N_MISSING_AFTER'], index=danom.index)
                   
                    
                    #check missing values before and after anomaly
                    start_anom = danom.index[0]
                    missbef, missaft = check_missval_around_anomaly(start_anom, tstamps)
                    # save data in dataframe 
                    anomalies.loc[danom.index[0]] = [1, #LENGTH
                                              missbef,# N_MISSING_BEFORE
                                              missaft]# N_MISSING_AFTER
                   
                    continue
                
                # differences of the time stamps of the anomalies
                time_gaps = diff_time_vec(danom.index).dropna()
                
                # time gaps greater than 1 mark beginning of anomaly
                # turn them into string of the timestamp
                start_of_anomalies = [s for s in time_gaps[time_gaps >1].index]
                
                # initialize DataFrame for saving data about the anomalies
                # here: index is different
                other_index = [danom.index[0]]
                [other_index.append(ind) for ind in start_of_anomalies]
                anomalies=pd.DataFrame(columns= ['LENGTH','N_MISSING_BEFORE', 'N_MISSING_AFTER'], index=other_index)
                
                # initiate index for time_gaps 
                i=0
                if time_gaps.iloc[i]>1: # first datestamp of anomaly time series (danom) marks an anomaly of length 1
                    
                    #check missing values before and after anomaly
                    missbef, missaft = check_missval_around_anomaly(start_anom, tstamps)
                    # save data in dataframe 
                    anomalies.loc[danom.index[0]] = [1, #LENGTH
                                              missbef,# N_MISSING_BEFORE
                                              missaft]# N_MISSING_AFTER
                else: # first datestamp of anomaly time series (danom) marks an anomaly of length >1
                    # find out length of anomaly
                    lgth=1
                    while  i<len(time_gaps) and time_gaps.iloc[i]<=1:
                        i+=1
                        lgth+=1    
                    #check missing values before and after anomaly
                    missbef, missaft = check_missval_around_anomaly(start_anom, tstamps, lgth)
                        
                    # save data in dataframe
                    anomalies.loc[start_anom] = [lgth, #LENGTH
                                              missbef,# N_MISSING_BEFORE
                                              missaft]# N_MISSING_AFTER
                   
                # all following anomalies
                for sta in start_of_anomalies:
                    start_anom = sta
                    print(start_anom)
                    
                    # index of time_gap corresponding next time step after start_anom
                    i = list(time_gaps.index).index(sta)+1
                    
                    # find out length of anomaly
                    lgth=1
                    while  i<len(time_gaps) and time_gaps.iloc[i]<=1:
                        i+=1
                        lgth+=1
                    
                    #check missing values before and after anomaly
                    missbef, missaft = check_missval_around_anomaly(start_anom, tstamps, lgth)
                    # save data in dataframe
                    anomalies.loc[start_anom] = [lgth, #LENGTH
                                              missbef,# N_MISSING_BEFORE
                                              missaft]# N_MISSING_AFTER
                   
                # save data
                anomalies.to_csv(savestring, sep=';', index_label='time_stamp')
        
        if len(time_series_without_anomalies)>0:
            with open(savestring_no_anomalies, 'w') as outfile:
                outfile.write('\n'.join(i for i in time_series_without_anomalies))
            outfile.close()
    return

def check_missval_around_anomaly(start_anom, tstamps, lgth=1):
    '''
    Checks for any missing values in time series before and after detected 
    anomaly and counts the number of them

    Parameters
    ----------
    start_anom : TimeStamp
        Beginning of the anomaly.
    tstamps : DateTimeIndex
        TimeStamps of observational data.
    lgth : int, optional
        Length of anomaly. The default is 1.

    Returns
    -------
    missbef : int
        number of missing values before anomaly.
    missaft : TYPE
        number of missing values after anomaly.

    '''
    
    # initialize missing values before and after anomaly with 0
    missbef=tgap_before=0
    missaft= tgap_after=0

    
    obsindex_start = list(tstamps).index(start_anom)
    obsindex_end = obsindex_start+lgth-1
    
    
    if obsindex_start>0:
        tgap_before=tstamps.iloc[obsindex_start-1:obsindex_start+1].diff()
        tgap_before = convert_duration_string(tgap_before.iloc[1]) # time gap in hours
    
    if obsindex_end<(len(tstamps)-1):
        tgap_after=tstamps.iloc[obsindex_end:obsindex_end+2].diff()
        tgap_after = convert_duration_string(tgap_after.iloc[1]) # time gap in hours
    
    if tgap_before>1:
        missbef=tgap_before
        
    if tgap_after>1:
        missaft=tgap_after
    
    return int(math.ceil(missbef)), int(math.ceil(missaft))

def plot_anomaly_mdata():
    count_rows=0
    nrows = len(stations)
    ncols=len(params)
    fstring = 'anomalies'
    for st in stations:
        stname = '_'.join(st.split(' '))
        count_cols=1
        curr_dir = '../Results/'+stname+'/Anomalies/'
        for p in params:
            for f in os.listdir(curr_dir):
                if f[0:2]==p and fstring in f:
                    print(f)
                # get depth level
                ff = f.split('_')
                d = float(ff[1])*-1
                data = pd.read_csv(curr_dir+f, sep=';', index_col='time_stamp')
                # convert string stating temporal duration to integer of hours
                dur_hours = data.LENGTH
                
               
                ylabelstr = 'Länge [Stunden]'
                
                #define colormap
                if p == 'WT':
                    col='blue'
                else:
                    col='purple'
                
                
                ax=plt.subplot(nrows, ncols, count_rows*2+count_cols)
                plt.bar(d, dur_hours, orientation='horizontal', color=col)            
                
                if st==stations[0]:
                    plt.title(paramdict[p], fontsize=fs)
                if p!='SZ':
                    plt.text(1.05, 0.5, stationsdict[st], 
                        horizontalalignment='center',
                        verticalalignment='center', 
                        transform=plt.gca().transAxes, 
                        rotation=90, **fontdict)
                
                # keep current ylims
                ylims = plt.ylim()
                
                
                # set xlims, ylims
                plt.ylim(ylims)
                plt.xlim(tlims)
                plt.grid()
                
                # make legend
                # legstring= [str(d) for d in unique_d]
                # legstring.append('flag=3,4')
                # plt.legend(legstring)
                
                
                
                # customize axes labels etc.
                plt.yticks(fontsize= fs)
                if count_rows*2+count_cols not in [nrows*ncols-1, nrows*ncols]:
                    ax.set_xticklabels([])
                else:
                    # labels = ax.get_xticklabels()
                    # label_locs = ax.get_xticks()
                    plt.xticks(rotation=45, fontsize=fs)
            count_cols+=1 
        count_rows+=1 
    #fig.savefig(savefigpath, bbox_inches=bbox_inches)
    return

#anomaly_exploration()
plot_anomaly_mdata()


