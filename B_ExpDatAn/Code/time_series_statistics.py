# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:33:40 2024

@author: bd2107
"""
import datetime as dt
import time
import numpy as np
import pandas as pd
from utilities import read_station_data, get_filestring
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from common_variables import datapath, layout, cm, stations, stationsdict,\
    params, paramdict, tlims, fs, fontdict, bbox_inches
    

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

def plot_coverage():
    # variables related to figure
    plt.rcParams['figure.figsize'][1]=10*cm
    fig = plt.figure(layout=layout)
    savefigpath = '../Figures/temporal_coverage.png'
    marker = [['1','v','P','s'], ['2', '^','*',  'D']]
    msize=7
    fillst= 'full'
    colors = ['blue', 'purple']
    
    # maximum possible length of time series
    startts = tlims[0]
    stopts = tlims[1]
    max_ts_length = len(pd.date_range(start=startts, end=stopts, freq='H'))
    counter_s = -1
    all_axes = []
    for s in stations:
        station_axes = []
        counter_s+=1
        counter_p = -1
        for p in params:
            counter_p+=1
            filestring = get_filestring(s,p,)
            file = datapath+filestring
            data = read_station_data(file)
            
            # fraction of time series len
            ts_frac = data.groupby('Z_LOCATION').count()/max_ts_length*100
            ax = plt.plot(ts_frac['DATA_VALUE'], ts_frac.index, marker[counter_s], markersize=msize,
                     fillstyle=fillst, color=colors[counter_p])
            station_axes.append(ax[0])
        all_axes.append(tuple(station_axes))
            
    plt.title('Zeitliche Abdeckung', fontsize=fs)
    
    # get current yticklabel locations
    yticklocs = plt.gca().get_yticks()
    yticklabels = plt.gca().get_yticklabels()
    yticklabels = [l._text.replace(chr(8722), '') for l in yticklabels]
 
    
    # set xlims, ylims, labels
    plt.ylim((-39,1))
    plt.ylabel('Wassertiefe [m]')
    plt.xlabel('Abdeckung [%]')
    plt.gca().set_yticks(yticklocs[1:-2], yticklabels[1:-2])

    plt.grid()
    plt.show()
    
    plt.legend(all_axes,list(stationsdict.values()),
               handler_map={tuple: HandlerTuple(ndivide=None)})
    
    # customize axes labels etc.
    plt.yticks(fontsize= fs)
    plt.xticks(fontsize=fs)
    fig.savefig(savefigpath, bbox_inches=bbox_inches)        
    return

def plot_gaps():
    # variables related to figure
    plt.rcParams['figure.figsize'][1]=10*cm
    fig = plt.figure(layout=layout)
    savefigpath = '../Figures/data_gaps.png'
    marker = [['1','v','P','s'], ['2', '^','*',  'D']]
    msize=7
    fillst= 'full'
    colors = ['blue', 'purple']
    
    # maximum possible length of time series
    startts = tlims[0]
    stopts = tlims[1]
    max_ts_length = len(pd.date_range(start=startts, end=stopts, freq='H'))
    counter_s = -1
    all_axes = []
    for s in stations:
        station_axes = []
        counter_s+=1
        counter_p = -1
        for p in params:
            counter_p+=1
            filestring = get_filestring(s,p,)
            file = datapath+filestring
            print(filestring)
            file = datapath+filestring
            data = read_station_data(file)
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            for d in unique_d:
                # get vector with time stamps
                time_vec = pd.Series(data[data['Z_LOCATION']==d].index)
                diff_vec = time_vec.diff()
                # time difference between neighbouring observations in hours
                diff_vec_hrs = diff_vec.dt.days*24+diff_vec.dt.seconds/3600
                
                # what minutes are recorded in time stamps?
                minutes = time_vec.dt.minute
                if len(set(minutes))>1:
                    print('Wechsel in Aufnahmeminute: \n')
                    n_minutes= minutes.to_frame().groupby('TIME_VALUE').size()
                    print(n_minutes/len(time_vec))
                
                # maximum time difference
                max_time_diff = np.nanmax(diff_vec_hrs)
                print('Maximale Zeitdifferenz: '+"%0.2f" %max_time_diff+' Stunden')
                # plot series of time differences
                plt.plot(diff_vec, '+')
                plt.gca().set_yscale('log')
                
                #make boxplots
                
                #Lagemaße Verteilung der zeitl. Differenzen
        all_axes.append(tuple(station_axes))
            
    plt.title('Zeitliche Abdeckung', fontsize=fs)
    
    # get current yticklabel locations
    yticklocs = plt.gca().get_yticks()
    yticklabels = plt.gca().get_yticklabels()
    yticklabels = [l._text.replace(chr(8722), '') for l in yticklabels]
 
    
    # set xlims, ylims, labels
    plt.ylim((-39,1))
    plt.ylabel('Wassertiefe [m]')
    plt.xlabel('Abdeckung [%]')
    plt.gca().set_yticks(yticklocs[1:-2], yticklabels[1:-2])

    plt.grid()
    plt.show()
    
    plt.legend(all_axes,list(stationsdict.values()),
               handler_map={tuple: HandlerTuple(ndivide=None)})
    
    # customize axes labels etc.
    plt.yticks(fontsize= fs)
    plt.xticks(fontsize=fs)
    fig.savefig(savefigpath, bbox_inches=bbox_inches)        

def main():
    for s in stations:
        for p in params:
            statistics(stname=s, paracode=p)
plot_gaps()
