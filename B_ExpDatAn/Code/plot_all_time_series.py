# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:13:09 2024

@author: Iris
"""

import matplotlib as mtpl
import numpy as np
from common_variables import (
    bbox_inches,
    cm,
    datapath,
    fontdict,
    fs,
    layout,
    paramdict,
    params,
    stations,
    stationsdict,
    tlims,
)
from matplotlib import pyplot as plt
from utilities import get_filestring, read_station_data

# define figure height
plt.rcParams['figure.figsize'][1]=14*cm
fig = plt.figure(layout=layout)

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
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            data=read_station_data(filestr=datapath+filestr)
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
                ddata_good = ddata[ddata["QF3"] == 2]
                plt.plot(
                    ddata_good.DATA_VALUE,
                    marker,
                    markersize=msize,
                    fillstyle=fillst,
                    color=cmp[d_counter],
                )

                if st == stations[0]:
                    plt.title(paramdict[p], fontsize=fs)
                if p != "SZ":
                    plt.text(
                        1.05,
                        0.5,
                        stationsdict[st],
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=plt.gca().transAxes,
                        rotation=90,
                        **fontdict,
                    )
                d_counter += 1
            # keep current ylims
            ylims = plt.ylim()
            
            # data with quality flag 3 or 4
            ddata_bad = data[data['QF3']!=2]
            plt.plot(ddata_bad.DATA_VALUE, marker, markersize=msize,markerfacecolor='r', color='r')
            
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
    fig.savefig(savefigpath, bbox_inches=bbox_inches)
        

def main():
    make_figure()
    
if __name__ == '__main__':
    main()