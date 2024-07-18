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

datapath = '../../A_Data/'
stations = ['Fehmarn Belt Buoy', 'Kiel Lighthouse', 
            'North Sea Buoy II', 'North Sea Buoy III']
stationsdict = {'Fehmarn Belt Buoy': 'Fehmarn', 
                'Kiel Lighthouse': "Leuchtturm Kiel", 
                'North Sea Buoy II': 'Nordsee II', 
                'North Sea Buoy III': 'Nordsee III'}

params = ['WT', 'SZ']
paramdict = {'WT': 'Wassertemperatur [° C]', 'SZ':'Salzgehalt []'}
time_period = '20200101_20240630'
xlims = [dt(2020,1,1,0,0),dt(2024,6,3,23,59)]

cm = 1/2.54  # conversion factor centimeters=>inches
figsize= (16.5*cm, 14*cm)
dpi=600
fig = plt.figure(figsize=figsize, layout='constrained',dpi=dpi)
#fig = plt.figure()
# fontdict for text in figure
fs = 10
fontdict = {'family': ['sans-serif'],
                     'variant': 'normal',
                     'weight': 'normal',
                     'stretch': 'normal',
                     'size': fs,
                     'math_fontfamily': 'dejavusans'}
savefigpath = '../Figures/all_time_series.png'
marker = '.'
msize=1
fillst= 'full'


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
                plt.plot(ddata_good.DATA_VALUE,marker, markersize=msize,fillstyle=fillst, color=cmp[d_counter])
                
                if st==stations[0]:
                    plt.title(paramdict[p], fontsize=fs)
                if p!='SZ':
                    plt.text(1.05, 0.5, stationsdict[st], 
                        horizontalalignment='center',
                        verticalalignment='center', 
                        transform=plt.gca().transAxes, 
                        rotation=90, **fontdict)
                d_counter+=1
            # keep current ylims
            ylims = plt.ylim()
            
            # data with quality flag 3 or 4
            ddata_bad = data[data['QF3']!=2]
            bad = plt.plot(ddata_bad.DATA_VALUE, marker, markersize=msize,markerfacecolor='r', color='r')
            
            # set xlims, ylims
            plt.ylim(ylims)
            plt.xlim(xlims)
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
    plt.savefig(savefigpath, bbox_inches='tight', dpi=dpi)
        

def read_station_data(filestr):
    usecols=['TIME_VALUE','Z_LOCATION','DATA_VALUE', 'QF1', 'QF3']
    data = pd.read_csv(filestr, index_col='TIME_VALUE', usecols=usecols)
    data.index = pd.to_datetime(data.index)
    return data

def main():
    make_figure()
    
main()