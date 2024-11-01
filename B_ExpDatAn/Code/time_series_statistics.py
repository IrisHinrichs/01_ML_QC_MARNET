# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:33:40 2024

@author: bd2107
"""
import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple

# Add absolute path of directory 
# 01_ML_QC_MARNET to sys.path
currentdir=os.path.dirname(__file__)
pathpieces = os.path.split(currentdir)
while pathpieces[-1]!='01_ML_QC_MARNET':
    currentdir= os.path.dirname(currentdir)
    pathpieces = os.path.split(currentdir)
sys.path.insert(0,currentdir)


from B_ExpDatAn.Code.utilities import get_filestring, read_station_data, diff_time_vec, convert_duration_string  # noqa: E402
from B_ExpDatAn.Code.common_variables import (  # noqa: E402
    stations,
    params,
    tlims,
    stationsdict,
    layout,
    cm,
    fs,
    bbox_inches,
)


def plot_coverage():
    # variables related to figure
    plt.rcParams['figure.figsize'][1]=10*cm
    fig = plt.figure(layout=layout)
    savefigpath = os.path.join(
        currentdir, "B_ExpDatAn", "Figures", "temporal_coverage.png"
    )
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
            data = read_station_data(filestring)
            
            # fraction of time series len
            ts_frac = data.groupby('Z_LOCATION').count()/max_ts_length*100
            ax = plt.plot(ts_frac['DATA_VALUE'], ts_frac.index,marker[counter_p][counter_s], markersize=msize,
                     fillstyle=fillst, color=colors[counter_p])
            station_axes.append(ax[0])
        all_axes.append(tuple(station_axes))
            
    plt.title('Zeitliche Abdeckung', fontsize=fs)
    
    # get current yticklabel locations
    yticklocs = plt.gca().get_yticks()
    yticklabels = plt.gca().get_yticklabels()
    yticklabels = [ll._text.replace(chr(8722), '') for ll in yticklabels]
 
    
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

def analyze_gaps():
    '''
    Analyzes temporal gaps in the MARNET time series data

    Returns
    -------
    None.

    '''
    
    # General collection of maximum time gap length 
    # of all time series
    all_max_time_diff= []
    all_min_time_diff = []
    
    
   
    for s in stations:
        # path for data storage
        savepath = os.path.join(
            currentdir, "B_ExpDatAn", "Results", "_".join(s.split(" "))
        )
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        for p in params:
            
            # savestrings for data storage
            
            # max time spans as function of 
            # time delta defining data gaps to be ingored
            savestringtstd = os.path.join(
                savepath,
                "Time_Spans",
                p + "_time_spans_deltas_" + "_".join(s.split(" ")) + ".csv",
            )
            
            # change in minutes of sampling scheme
            savestringmints = os.path.join(
                savepath, p + "_sampling_scheme_" + "_".join(s.split(" ")) + ".csv"
            )
            
            # time delta corresponding t 99.9% of cumulative distribution
            # savestringtd99p9= savepath+'Quantiles/' +p+'_td99.9_'+'_'.join(s.split(' '))+'.csv'
            # time delta corresponding t 50% of cumulative distribution
            savestringtd50 = os.path.join(
                savepath, "Quantiles", p + "_td50_" + "_".join(s.split(" ")) + ".csv"
            )
            
            filestring = get_filestring(s,p,)
            print(filestring)
            
            # read data
            data = read_station_data(filestring)
            
            # get list of unique depth levels
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            depth_index = [abs(d) for d in unique_d]
            
            # get unique time stamps
            unique_t =  pd.Series(data.index.unique())
            
            # what minutes are recorded in time stamps?
            minutes = unique_t.dt.minute
            if len(set(minutes))>1:
                #print('Wechsel in Aufnahmeminute: \n')
                n_minutes= minutes.to_frame().groupby('TIME_VALUE').size()
                #print(n_minutes/len(unique_t))
                mints_fracs = (n_minutes/len(unique_t))
                mints_fracs.name='temporal fraction'
                mints_fracs.to_csv(savestringmints, sep= ';', index_label= 'minute')
                
                # find out when the change in minutes happens
                all_minutes = pd.Series(data.index, index=data.index).dt.minute
                diff_minutes = all_minutes.diff()
                change_ind = diff_minutes[diff_minutes!=0].index
                second_ind = change_ind[1]
                first_ind = list(diff_minutes.index).index(second_ind)-1
                first_ind = diff_minutes.index[first_ind]
                change_ind = [first_ind, second_ind]
                add_line = 'Wechsel zwischen '+' und '.join(str(cind) for cind in change_ind)
                with open(savestringmints, 'a') as f:
                    f.write(add_line)
                
                
            # initialize DataFrame for saving the trade-off between maximum time span
            # and minimum length of gaps being ignored
            depth_max_tspans=pd.DataFrame(columns= ['MIN_TIME_DELTA','DURATION', 'START'], index=depth_index)
            
            # initialize DataFrame for saving time delta corresponding to 99.9% of 
            # cumulative distribution of temporal differences in time series
            # tdelta_99p9=pd.DataFrame(columns= ['TIME_DELTA_99p9'], index=depth_index)
            tdelta_50=pd.DataFrame(columns= ['TIME_DELTA_0p5'], index=depth_index)
            for d in unique_d:
                # filestring for data storage
                savestringts = os.path.join(
                    savepath,
                    "Time_Spans",
                    p
                    + "_"
                    + str(abs(d))
                    + "_max_time_spans_"
                    + "_".join(s.split(" "))
                    + ".csv",
                )
                
                
                #print('Tiefenstufe '+ str(abs(d))+' m')
                
                # get vector with time stamps
                time_vec = time_vec = data[data['Z_LOCATION']==d].index
                
                # time difference between neighbouring observations in hours
                diff_vec_hrs = diff_time_vec(time_vec)
                
                # maximum time difference
                max_time_diff = np.nanmax(diff_vec_hrs)
                #print('Maximale Zeitdifferenz: '+"%0.2f" %max_time_diff+' Stunden')
                
                # minimum time difference
                min_time_diff = np.nanmin(diff_vec_hrs)
                # print('Minimale Zeitdifferenz: '+"%0.2f" %min_time_diff+' Stunden')
               
                # cumulative distribution of time differences
                # keep only values greater than 1 hour time difference
                vg1 = diff_vec_hrs[diff_vec_hrs>1]
                dummy = vg1.to_frame(name='delta_t')
                cumdistr = dummy.groupby('delta_t').size().cumsum()/len(dummy)
                # time difference marking 99.9% of cumulative distribution
                hr_delta_50 = cumdistr[cumdistr>0.5].index[0]
                tdelta_50.loc[abs(d)]=hr_delta_50
                
                
                all_max_time_diff.append(max_time_diff)
                all_min_time_diff.append(min_time_diff)
                
                # maximum time span with observations
                # depends on tolerance of time delta
                timedelta = range(1,25)
                
                # list of all maximum time spans together with 
                # start date of current time series
                # depending on timedelta
                max_tspans = pd.DataFrame(columns=[ 'DURATION', 'START'], index=timedelta) 
                                         
                for tdel in timedelta:
                    # find time all time intervalls between temporal gaps defined by tdel
                    tspans = find_all_time_spans(time_vec,tdel) 
                   
                    maxval= max(tspans) 
                    
                    # max time spans for all time deltas, converges at specific time delta                              
                    max_tspans.loc[tdel][:]=[
                                                maxval, # DURATION
                                                tspans[tspans==maxval].index[0]#START
                                             ]
                # save data, maximum time spans as function of timedelta
                max_tspans.to_csv(savestringts,sep=';', index_label='time_delta')
                
                
                    
                # find out minimum time delta between 0 and 24 hours
                # corresponding to maximum time span
                ind = max_tspans[max_tspans.DURATION==max(max_tspans.DURATION)].index[0]
                depth_max_tspans.loc[abs(d)] = [ind, #MIN_TIME_DELTA
                                           max_tspans.DURATION.loc[ind],# DURATION
                                           max_tspans.START.loc[ind]]# START
            # save data, trade-off between maximum time span
            # and minimum gap length of gaps being ignored 
            depth_max_tspans.to_csv(savestringtstd, sep=';', index_label='depth level')
            tdelta_50.to_csv(savestringtd50, sep=';', index_label='depth level')
                                           
    print('Maximale Zeitdifferenzen in Stunden:')
    print(sorted(list(set(all_max_time_diff)),reverse=True))
    print(' ')
    print('Maximale Zeitdifferenzen in Tagen:')
    print([m/24 for m in sorted(list(set(all_max_time_diff)),reverse=True)])
    print(' ')
    print('Minimale Zeitdifferenzen in Stunden:')
    print(sorted(list(set(all_min_time_diff)),reverse=True))

def plot_max_time_spans(time_delta=False, frac=False):
    '''
    Makes figure of results of function 'analyze_gaps'

    Parameters
    ----------
    time_delta : boolean, optional
        If set to True results of maximum time interval with consecutive 
        observations as function of tolerated time delta are plotted
        If set to False, tolerated time delta corresponds to one
        and maximum time intervals on all depth levels are presented.
        The default is False.
    frac : boolean, optional
        If set to True maximum time intervals are plotted as fractional 
        value in %. If set to false, intervals are plotted in absolute values
        of days. The default is False.

    Returns
    -------
    None.

    '''
    # variables related to figure
    plt.rcParams['figure.figsize'][1]=10*cm
    fig = plt.figure(layout=layout)
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
    
    # path to files with results from analyze data_gaps
    resultpath = os.path.join(currentdir, 'B_ExpDatAn','Results')
    fstring = 'max_time_spans'
    if time_delta:
        savefigpath = os.path.join(
            currentdir, "B_ExpDatAn", "Figures", "max_time_spans_deltas.png"
        )
    else:
        savefigpath = os.path.join(
            currentdir, "B_ExpDatAn", "Figures", "max_time_spans.png"
        )
    for s in stations:
        station_axes = []
        counter_s+=1
        counter_p = -1
        stname = '_'.join(s.split(' '))
        for p in params:
            counter_p+=1
            curr_dir = os.path.join(resultpath,stname)
            for f in os.listdir(curr_dir):
                if f[0:2]==p and fstring in f:
                    print(f)
                    
                    if time_delta:
                        data = pd.read_csv(curr_dir+f, sep=';', index_col='time_delta')
                        if frac:
                            dur_hours_frac= [convert_duration_string(dh)/max_ts_length*100 for dh in data.DURATION]
                            ylabelstr = 'Anteilige Länge [%]'
                        else:
                            dur_hours_frac= [convert_duration_string(dh)/24 for dh in data.DURATION]
                            ylabelstr = 'Länge [Tage]'
                            
                        #plt.subplot(1,2,1)
                        ax = plt.plot(data.index,dur_hours_frac, '-'+marker[counter_p][counter_s], markersize=5,
                                 fillstyle=fillst, color=colors[counter_p], linewidth=0.5)
                        # plt.subplot(1,2,2)
                        # ax= plt.plot(data.MIN_TIME_DELTA, depth_levels, marker[counter_p][counter_s], markersize=msize,
                        #          fillstyle=fillst, color=colors[counter_p])
                    else:
                        # get depth level
                        ff = f.split('_')
                        d = float(ff[1])*-1
                        data = pd.read_csv(curr_dir+f, sep=';', index_col='time_delta')
                        # convert string stating temporal duration to integer of hours
                        dur_raw = data.loc[1].DURATION
                        
                        dur_hours=convert_duration_string(dur_raw)
                        if frac:
                            dur_hours_frac= dur_hours/max_ts_length*100
                            ylabelstr = 'Anteilige Länge [%]'
                        else:
                            dur_hours_frac= dur_hours/24
                            ylabelstr = 'Länge [Tage]'
                        
                        
                        ax = plt.plot(dur_hours_frac, d, marker[counter_p][counter_s], markersize=msize,
                                 fillstyle=fillst, color=colors[counter_p])
                        
            station_axes.append(ax[0])
        all_axes.append(tuple(station_axes))
        
    
    if time_delta:
        
        # subplot #1
        #plt.subplot(1,2,1)
        plt.title('Längste Zeitspanne konsekutiver Beobachtungen', fontsize=fs)
        
        # get current yticklabel locations
        yticklocs = plt.gca().get_yticks()
        yticklabels = plt.gca().get_yticklabels()
        yticklabels = [ll._text.replace(chr(8722), '') for ll in yticklabels]
     
        
        # set xlims, ylims, labels
        #plt.ylim((-39,1))
        plt.ylabel(ylabelstr)
        plt.xlabel(r'$\Delta$t$_{tol}$ [Stunden]')
        #plt.gca().set_yticks(yticklocs[1:-2], yticklabels[1:-2])
    
        plt.grid()
        
        plt.legend(all_axes,list(stationsdict.values()),
                   handler_map={tuple: HandlerTuple(ndivide=None)})
        
        # customize axes labels etc.
        plt.yticks(fontsize= fs)
        plt.xticks(fontsize=fs)
        
        # # subplot #2
        # plt.subplot(1,2,2)
        # plt.title('Minimale Länge ignorierter zeitl. Lücken', fontsize=fs)
        
       
        # # set xlims, ylims, labels
        # plt.ylim((-39,1))
        # plt.ylabel('')
        # plt.xlabel('Länge [Stunden]')
        # plt.gca().set_yticks([])
    
        # plt.grid()
        
        # # customize axes labels etc.
        # plt.yticks(fontsize= fs)
    else:
        
        plt.title('Längste Zeitspanne konsekutiver Beobachtungen', fontsize=fs)
        
        # get current yticklabel locations
        yticklocs = plt.gca().get_yticks()
        yticklabels = plt.gca().get_yticklabels()
        yticklabels = [ll._text.replace(chr(8722), '') for ll in yticklabels]
     
        
        # set xlims, ylims, labels
        plt.ylim((-39,1))
        plt.ylabel('Wassertiefe [m]')
        plt.xlabel(ylabelstr)
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

def plot_td(quant='50'):
    '''
    Plots the results of analyze_gaps, in this case the 0.999 quantile
    of the temporal differences of all time series
    
    Parameters
    ----------
    quant : str, optional
            defines the quantile to be plotted, default is 50, meaning the median 
            of the distribution of the 

    Returns
    -------
    None.

    '''
    # variables related to figure
    plt.rcParams['figure.figsize'][1]=10*cm
    fig = plt.figure(layout=layout)
    marker = [['1','v','P','s'], ['2', '^','*',  'D']]
    msize=7
    fillst= 'full'
    colors = ['blue', 'purple']
    all_axes = []
    savefigpath = os.path.join(
        currentdir, "B_ExpDatAn", "Figures", "td" + quant + ".png"
    )
   
    
    # path to files with results from analyze data_gaps
    resultpath = os.path.join(currentdir, 'B_ExpDatAn','Results')
    fstring = 'td'+quant
    
    
    counter_s = -1
    for s in stations:
        station_axes = []
        counter_s+=1
        counter_p = -1
        stname = '_'.join(s.split(' '))
        for p in params:
            counter_p+=1
            curr_dir = os.path.join(resultpath,stname)
            for f in os.listdir(curr_dir):
                if f[0:2]==p and fstring in f:
                    print(f)
                    data = pd.read_csv(curr_dir+f, sep=';', index_col='depth level')
                   
                    ax = plt.plot(data,data.index*-1, marker[counter_p][counter_s], markersize=msize,
                             fillstyle=fillst, color=colors[counter_p])
                        
            station_axes.append(ax[0])
        all_axes.append(tuple(station_axes))
  
    plt.title(r'Median der Zeitdifferenzen $\Delta$t > 1', fontsize=fs)
    
    # get current yticklabel locations
    yticklocs = plt.gca().get_yticks()
    yticklabels = plt.gca().get_yticklabels()
    yticklabels = [ll._text.replace(chr(8722), '') for ll in yticklabels]
 
    
    # set xlims, ylims, labels
    plt.ylim((-39,1))
    plt.ylabel('Wassertiefe [m]')
    plt.xlabel(r'$\Delta$t$_{0.5}$ [Stunden]')
    plt.gca().set_yticks(yticklocs[1:-2], yticklabels[1:-2])
    plt.gca().set_xticks(range(0,21,5))

    plt.grid()
    plt.show()
    
    plt.legend(all_axes,list(stationsdict.values()),
               handler_map={tuple: HandlerTuple(ndivide=None)})
    
    # customize axes labels etc.
    plt.yticks(fontsize= fs)
    plt.xticks(fontsize=fs)
    fig.savefig(savefigpath, bbox_inches=bbox_inches)        
    return
    

def find_all_time_spans(time_vec, tdel):
    '''
    Finds all time intervals of consecutive time steps that are not greater
    than a defined time delta

    Parameters
    ----------
    time_vec : pandas.core.indexes.datetimes.DatetimeIndex
        time vector containing time stamps .
    tdel : int
        time delta defining limit for tolerable time steps with respect to
        time intervall with consecutive observations.

    Returns
    -------
    tspans : pandas.core.series.Series
        pandas Series containing the detected time intervalls. Index corresponds
        to the beginning of the time interval, length of the intervals are provided
        as timedelta

    '''
    diff_vec_hrs = diff_time_vec(time_vec)
    time_gaps=diff_vec_hrs[diff_vec_hrs>tdel] # all time gaps greater than tdel
    
    # get time spans, iterate over all gaps
    tspans = pd.Series(dtype='float64') # list of duration of all time spans 
                                        # for current definition of time gap
                                        # and current time series, index: beginning of time span
    old_end_gap = time_vec[0]
    for counter_tg in range(0, len(time_gaps)+1):
        
        # distinguish between first, last and inbetween gap
        if counter_tg==0: # first gap
            # Index marking end of big gap in time vector
            ii = list(time_vec).index(time_gaps.index[counter_tg])
            
            # time stamp of beginning and end of big gap in time vector
            start_gap=time_vec[ii-1]
            end_gap = time_vec[ii]
            tspan= start_gap-time_vec[0]
        elif counter_tg==len(time_gaps): # last gap
            tspan= time_vec[-1]-old_end_gap
        else: # inbetween gap
            # Index marking end of big gap in time vector
            ii = list(time_vec).index(time_gaps.index[counter_tg])
            
            # time stamp of beginning and end of big gap in time vector
            start_gap=time_vec[ii-1]
            end_gap = time_vec[ii]
            tspan = start_gap-old_end_gap
           
        tspans[old_end_gap]=tspan
        old_end_gap = end_gap 
    return tspans

if __name__=="__main__":
    plot_coverage()
    #analyze_gaps()
    #plot_max_time_spans(time_delta=False, frac=False)
    #plot_td()