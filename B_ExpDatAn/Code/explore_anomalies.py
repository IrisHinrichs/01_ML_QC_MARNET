# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:32:37 2024

@author: bd2107
"""
import math
import datetime as dt
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple 
import matplotlib.dates as mdates
import os
import sys

# Add absolute path of directory 
# 01_ML_QC_MARNET to sys.path
currentdir=os.path.dirname(__file__)
pathpieces = os.path.split(currentdir)
while pathpieces[-1]!='01_ML_QC_MARNET':
    currentdir= os.path.dirname(currentdir)
    pathpieces = os.path.split(currentdir)
sys.path.insert(0,currentdir)


from B_ExpDatAn.Code.common_variables import (  # noqa: E402
    layout,
    cm,
    stations,
    stationsdict,
    params,
    paramdict,
    tlims,
    fs,
    fontdict,
    bbox_inches,
)  # noqa: E402
from B_ExpDatAn.Code.utilities import (  # noqa: E402
    datapath,
    read_station_data,
    get_filestring,
    diff_time_vec,
    convert_duration_string,
)


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
     # define path and filename
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
    
            # Find data with quality flag set to 4 or 3 in both validation levels
            mask = (data.QF3.isin([3,4])) | (data.QF1.isin([3,4]))
            val3_qf3_or_4 = data.loc[mask]
            
            
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

def plot_fraction_seq():
    '''Analyse fraction of anomalous point that are part of sequential anomalies
      of length greater than [seq_len] in single
    time series and plot results'''
   
    fstring = '_anomalies_'
    seq_len = 5
   
    # variables related to figure
    plt.rcParams['figure.figsize'][0]=16*cm
    plt.rcParams['figure.figsize'][1]=10.0*cm
    
    savefigpath = os.path.join('B_ExpDatAn', 'Figures')
    resultsfile = os.path.join('B_ExpDatAn','Results', 'fraction_anom_points_in_seq_glen_'+str(seq_len)+'.csv')
    results = []
    markerT=["1", "v", "P", "s"]
    markerT.reverse()
    markerS=["2", "^", "*", "D"]
    markerS.reverse()
    marker = [markerT, markerS]
    msize=7
    fillst= 'full'
    colors = ['blue', 'purple']
    fig = plt.figure(layout=layout)
    
    all_axes = []
    counter_s = -1
    stations.reverse()
    for st in stations:
        station_axes = []
        counter_s+=1
        counter_p = -1
        stname = '_'.join(st.split(' '))
        curr_dir = os.path.join("B_ExpDatAn", "Results", stname, "Anomalies")
        for p in params:
            counter_p+=1
            for f in os.listdir(curr_dir):
                if f[0:2]==p and fstring in f:
                    print(f)
                else:
                    continue
                # get depth level
                ff = f.split('_')
                d = float(ff[1])*-1
                data = pd.read_csv(os.path.join(curr_dir,f), sep=';', index_col='time_stamp')
                data_c = combine_anomalies(data)
                # calculate fraction of anomalies that are longer than 1
                longer_1 = data_c[data_c.LENGTH>seq_len]
                frac_anom = (longer_1.LENGTH.sum()-longer_1.N_MISSING_IN_SEQ.sum())/data.LENGTH.sum()
                l1 = plt.plot(frac_anom, d ,marker[counter_p][counter_s], markersize=msize,
                        fillstyle=fillst, color=colors[counter_p])
                # save data in dataframe
                new_entry = [st, p, d, frac_anom]
                results.append(new_entry)
            station_axes.append(l1[0])
        all_axes.append(tuple(station_axes))

    # save fractions data
    resdf = pd.DataFrame(
        results,
        columns=[
            "Station",
            "Parameter",
            "Depth",
            "Fraction of anomalous values being part of of sequences longer than "
            + str(seq_len),
        ]
    ) 
    # mean fraction temperature
    col = 'Fraction of anomalous values being part of of sequences longer than '+str(seq_len)
    avgs= resdf.groupby('Parameter')[col].mean()
    mean_temp = avgs.WT
    # mean fraction salinity 
    mean_sal = avgs.SZ 

    plt.plot([mean_temp]*2, [-39, 1], '-', color=colors[0])
    plt.plot([mean_sal]*2, [-39, 1], '-', color=colors[1])       
    # set xlims, ylims, labels
    plt.ylabel('Wassertiefe [m]', fontsize=fs)
    plt.xlabel('Anteil', fontsize=fs)
    plt.title('Markierte Werte als Sequenz', fontsize=fs)
    # get current yticklabel locations
    yticklocs = plt.gca().get_yticks()
    yticklabels = plt.gca().get_yticklabels()
    yticklabels = [ll._text.replace(chr(8722), '') for ll in yticklabels]
    plt.gca().set_yticks(yticklocs[1:-2], yticklabels[1:-2])
    plt.ylim((-39,1))
    plt.xlim((-0.01,1.01))
    plt.grid()
    
    legendstrings = list(stationsdict.values())
    legendstrings.reverse()
    plt.legend(all_axes,legendstrings,
            handler_map={tuple: HandlerTuple(ndivide=None)})
    
    # customize axes labels etc.
    plt.yticks(fontsize= fs)
    plt.xticks(fontsize=fs)
    savefigstr = os.path.join(savefigpath,'fraction_of_values_in_seq_anomalies_l'+str(seq_len)+'.png')
    fig.savefig(savefigstr, bbox_inches=bbox_inches)  
     
    resdf.to_csv(resultsfile, index=False)           
    return
def visualize_anomalies():
    fstring = '_anomalies_'
   
    # define figure height
    plt.rcParams['figure.figsize'][0]=16.5*cm
    plt.rcParams['figure.figsize'][1]=6*cm
    #plt.rcdefaults()
    
    for st in stations:
        stname = '_'.join(st.split(' '))
        curr_dir = '../Results/'+stname+'/Anomalies/'
        for p in params:
            
            #define colormap
            if p == 'WT':
                col='blue'
                ylabelstr = '[° C]'
            else:
                col='purple'
                ylabelstr = '[PSU]'
                
                
            # read station data
            station_data = read_station_data(datapath+get_filestring(st, p))
            
           
            for f in os.listdir(curr_dir):
                if f[0:2]==p and fstring in f:
                    print(f)
                else:
                    continue
                # get depth level
                ff = f.split('_')
                d = float(ff[1])*-1
                data = pd.read_csv(curr_dir+f, sep=';', index_col='time_stamp')
                
                # create path to save figures
                savefigpath = '../Figures/'+ stname+'/Anomalies/'+p+'_'+str(abs(d))+'/'
                if not os.path.exists(savefigpath):
                    os.makedirs(savefigpath)
                
                # station data on specific depth level
                stdata_d = station_data[station_data.Z_LOCATION==d]
               
                
                # combine anomalies that are only connected by a sequence of 
                # missing values
                len_data_old = 0
                while len(data)!= len_data_old: # repeat procedure until there 
                                                # are no pairwise anomlies that 
                                                # need to be combined
                                                # and input and output of function
                                                # combine_anomalies have equal length
                    len_data_old = len(data)
                    data = combine_anomalies(data)
                    
                
                # iterate over all anomalies and viusalize the data
                extension = 25 # also visualize t25 hours before and after
                                # start and end of anomaly

                for ind in range(0,len(data)):

                    fig = plt.figure()
                    # time stamp of starting point of current anomaly
                    start_anom =dt.datetime.strptime(data.index[ind], '%Y-%m-%d %H:%M:%S')
                    # length of current anomaly
                    len_anom = data.LENGTH.iloc[ind]
                    
                    start_str = data.index[ind].replace(' ', '_')
                    start_str = start_str.replace(':', '-')
                    end_str = str(start_anom+dt.timedelta(hours=int(len_anom))).replace(' ', '_')
                    end_str = end_str.replace(':', '-')
                    figname = start_str+'__'+end_str+'.png'
                    # start of visualization
                    start_vis = start_anom-dt.timedelta(hours=extension)
                    # end of visualization
                    end_vis = start_anom+dt.timedelta(hours=(int(len_anom)+extension))
                    # data to visualize
                    vis_mask = (stdata_d.index>=start_vis) & (stdata_d.index <=end_vis)
                    vis_data = stdata_d.loc[vis_mask]
                    vis_anom_QF3 = vis_data.DATA_VALUE[vis_data.QF3.isin([3,4])]
                    vis_anom_QF1 = vis_data.DATA_VALUE[vis_data.QF1.isin([3,4])]
                    vis_obs = vis_data.DATA_VALUE
                    
                    # plot time series
                    plt.plot(vis_anom_QF3, 'ro',alpha=0.2, markersize=7, linewidth=2)
                    plt.plot(vis_anom_QF1, 'ko',alpha=0.2, markersize=7, linewidth=2)
                    plt.plot(vis_obs, '.', color=col, markersize=3, linewidth=2)
                    plt.grid()
                    pstring = paramdict[p].replace(' '+ylabelstr, '')
                    titlestring = stationsdict[st].replace('[° C]', '')+', '+pstring+', '+\
                                    start_vis.strftime('%d.%m.%Y %H:%M:%S')+'-'+end_vis.strftime('%d.%m.%Y %H:%M:%S')
                    plt.title(titlestring, fontsize=fs, wrap=True)
                    plt.ylabel(ylabelstr)
                    plt.annotate(str(abs(d))+' m', xy=(0.05, 0.05), xycoords='axes fraction')
                    
                    #
                    # date_form = DateFormatter("%b-%d")
                    # plt.gca().xaxis.set_major_formatter(date_form)
                    
                    plt.gca().xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(plt.gca().xaxis.get_major_locator()))

                    plt.xlim(start_vis, end_vis)
                    
                    plt.show()
                    fig.savefig(savefigpath+figname, bbox_inches=bbox_inches)
                    plt.close(fig)
                   
                   
    
def combine_anomalies(anom_meta, gap_thresh=12):
    '''
    Combines anomalies that are connected by a sequence of missing data points

    Parameters
    ----------
    anom_meta : pandas.core.frame.DataFrame
        Dataframe containing meta data about anomalies.
        Index corresponds to time stamp of beginnning of anomaly,
        columns are "LENGTH", "N_MISSING_BEFORE", "N_MISSING_AFTER"
    gap_thresh : int, optional
        Threshold defining the length to which a sequence of missing values 
        between two anomalies is accepted as connecting. The default is 12.

    Returns
    -------
    anomalies : pandas.core.frame.DataFrame
        same as anom_meta but with combined anomalies.

    '''
    # initiate dataframe for new anomaly meta data
    anomalies=pd.DataFrame(columns= list(anom_meta.columns))
    anomalies.loc[:,'N_MISSING_IN_SEQ']= []
    cont=False
    for ind in range(0,len(anom_meta)):
        if cont is True:
            cont=False
            continue
        # length of current anomaly
        anom_len = anom_meta.LENGTH.iloc[ind]
        # number of missing values before and after current anomaly
        missbef= anom_meta.N_MISSING_BEFORE.iloc[ind]
        missaft= anom_meta.N_MISSING_AFTER.iloc[ind]
        # start of current anomaly
        start = dt.datetime.strptime(anom_meta.index[ind], '%Y-%m-%d %H:%M:%S')
        
        if missaft>0 and missaft <= gap_thresh and ind+1<len(anom_meta):
            # number of missing values before next anomaly
            next_missbef = anom_meta.N_MISSING_BEFORE.iloc[ind+1]
            # start of next anomaly
            start_next= dt.datetime.strptime(anom_meta.index[ind+1], '%Y-%m-%d %H:%M:%S')
            len_next = anom_meta.LENGTH.iloc[ind+1]
            next_missaft=anom_meta.N_MISSING_AFTER.iloc[ind+1]
            end_anom = start+dt.timedelta(hours=int(anom_len+missaft))
            if next_missbef==missaft and start_next-end_anom<=dt.timedelta(seconds=3600): # timedelta in seconds
                # both anomalies are connected by a sequence of missing values
                # and can therefore be considered as a single anomaly
                new_current_anomaly = [anom_len+missaft+len_next,missbef, next_missaft, missaft ]
                anomalies.loc[start] = new_current_anomaly
                cont=True
            else:# anomaly is not combined with another one
                anomalies.loc[start]= [anom_len, missbef, missaft, 0] 
                cont=False
        else: # anomaly is not combined with another one
            anomalies.loc[start]= [anom_len, missbef, missaft, 0]
            cont=False
            continue
    # reset index
    anomalies=anomalies.set_index(anomalies.index.strftime('%Y-%m-%d %H:%M:%S'))
    return anomalies        
    
if __name__=='__main__':
    # anomaly_exploration()
    # plot_anomaly_mdata()
    # visualize_anomalies()
    plot_fraction_seq()


