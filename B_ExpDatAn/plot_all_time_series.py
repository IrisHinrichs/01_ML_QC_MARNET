# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:13:09 2024

@author: Iris
"""
from matplotlib import pyplot as plt
import matplotlib as mtpl
import pandas as pd
from datetime import datetime as dt
import numpy as np
from cycler import cycler
import seaborn as sns

datapath = '../A_Data/'
stations = ['Fehmarn Belt Buoy', 'Kiel Lighthouse', 
            'North Sea Buoy II', 'North Sea Buoy III']

params = ['WT', 'SZ']
paramdict = {'WT': 'Wassertemperatur [° C]', 'SZ':'Salzgehalt [PSU]'}
time_period = '20200101_20240630'
xlims = [dt(2020,1,1,0,0),dt(2024,6,3,23,59)]

cm = 1/2.54  # centimeters in inches
figsize= (25*cm, 17*cm)
fig = plt.figure(figsize=figsize, layout='constrained')
# fontdict for text in figure
fontdict = {'family': ['sans-serif'],
                     'variant': 'normal',
                     'weight': 'normal',
                     'stretch': 'normal',
                     'size': 12.0,
                     'math_fontfamily': 'dejavusans'}
savefigpath = 'C:/Users/Iris/Documents/IU-Studium/Masterarbeit/Code_Stuf/temp/all_time_series.png'

def make_figure():
    count_rows=0
    nrows = len(stations)
    ncols=len(params)
    for st in stations:
        count_cols=1
        for p in params:
            filestr = datapath+st.replace(' ', '_')+'_'+time_period+'_'+p+'_all_.csv'
            data=read_station_data(filestr=filestr)
            # unique depth levels of current station
            unique_d=list(set(data.Z_LOCATION))
            unique_d.sort(reverse=True)
            #define colormap
            if p == 'WT':
                cmp =mtpl.colormaps['Blues'](np.linspace(0,1,len(unique_d)+3))
            else:
                cmp = mtpl.colormaps['Purples'](np.linspace(0,1,len(unique_d)+3))
            
            # initiate counter for color values
            d_counter=2
            for d in unique_d:
               
                ddata = data[data['Z_LOCATION']==d] # entries corresponding to depth level d
                
                ax=plt.subplot(nrows, ncols, count_rows*2+count_cols)
                            
                # data with qf=2
                ddata_good = ddata[ddata['QF3']==2]
                plt.plot(ddata_good.DATA_VALUE, '+', color=cmp[d_counter])
                
                if st==stations[0]:
                    plt.title(paramdict[p])
                if p!='SZ':
                    plt.text(1.05, 0.5, st, 
                        horizontalalignment='center',
                        verticalalignment='center', 
                        transform=plt.gca().transAxes, 
                        rotation=90, **fontdict)
                d_counter+=1
            # keep current ylims
            ylims = plt.ylim()
            
            # data with quality flag 3 or 4
            ddata_bad = data[data['QF3']!=2]
            bad = plt.plot(ddata_bad.DATA_VALUE, '+', color='r')
            
            # set xlims, ylims
            plt.ylim(ylims)
            plt.xlim(xlims)
            plt.grid()
            
            # make legend
            # legstring= [str(d) for d in unique_d]
            # legstring.append('flag=3,4')
            # plt.legend(legstring)
            
            
            
            # customize axes labels etc.
            if count_rows*2+count_cols not in [nrows*ncols-1, nrows*ncols]:
                ax.set_xticklabels([])
            else:
                labels = ax.get_xticklabels()
                ax.set_xticklabels(labels,rotation=25)
            count_cols+=1 
        count_rows+=1 
    plt.savefig('../../00_Masterarbeit/Abbildungen/all_time_series.png', bbox_inches='tight')
        

def read_station_data(filestr):
    usecols=['TIME_VALUE','Z_LOCATION','DATA_VALUE', 'QF1', 'QF3']
    data = pd.read_csv(filestr, index_col='TIME_VALUE', usecols=usecols)
    data.index = pd.to_datetime(data.index)
    return data

def main():
    make_figure()
    
main()