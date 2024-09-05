# -*- coding: utf-8 -*-
import matplotlib as mtpl
import numpy as np
from scipy.signal import lombscargle
from B_ExpDatAn.Code.common_variables import (
    bbox_inches,
    cm,
    
    fs,
    layout,
    fontdict,
    paramdict,
    params,
    stations,
    stationsdict,
    tlims,
)
from matplotlib import pyplot as plt
from B_ExpDatAn.Code.utilities import datapath, get_filestring, read_station_data

import os
import sys

os.chdir(sys.path[0]) # change to modules parent directory

nlags = 96 # for autocorrelation
def plot_periodograms(res='h'):
    '''
    Plots Peridograms for all time series (all stations, parameters, depth levels)
    for periods in hours or days

    Parameters
    ----------
    res : str
        String stating the temporal resolution of the time periods to analyze.
        Default is 'h' for hourly resolution.

    Returns
    -------
    None.

    '''
    # define figure height
    plt.rcParams['figure.figsize'][1]=16*cm
    fig = plt.figure(layout=layout)

    # for Lomb-Scargle Periodogram
    if res=='h': # hourly resolution of periods
        periods = np.linspace(1,30, 30)
        pstring = '30hrs'
    elif res=='d':
        periods = np.linspace(1,380, 380)
        pstring='380days'

    freqs=[np.pi*2/p for p in periods]
    savefigpath = '../Figures/periodograms_'+pstring+'.png'


    count_rows=0
    nrows = len(stations)
    ncols=len(params)
    ax_1_list = [] # for WT
    ax_2_list = [] # for SZ
    #ax_invs= plt.subplots(1,2)
    for st in stations:
            count_cols=1
            for p in params:

                #define colormap according to parameter
                if p == 'WT':
                    cmp =mtpl.colormaps['Blues']
                else:
                    cmp = mtpl.colormaps['Purples']
                # read data
                filestr = get_filestring(st, p, tlims[0], tlims[1])
                data=read_station_data(filestr=datapath+filestr)
                # unique depth levels of current station
                unique_d=list(set(data.Z_LOCATION))
                unique_d.sort(reverse=True)
                #acf_array = np.zeros((len(unique_d),nlags+1))
                prdgram_array = np.zeros((len(unique_d), len(periods)))
                extent = [0.5, len(periods)+0.5, 0.5, len(unique_d)+0.5]
                counter_d = 0
                for d in unique_d:

                    ddata = data[(data['Z_LOCATION']==d)&(data['QF3']==2)] # entries corresponding to depth level d and good data
                    if res=='d': # daily resolution of periods
                      #resampling from hourly to daily resolution is necessary
                        ddata = ddata.resample('D').mean()  


                    #ddata = ddata.asfreq('h') # change to hourly frequency
                    # not recommended, replaces all values after minutes shift with NaNs
                    # resampling and forward filling
                    #ddata_res = ddata.resample('h').ffill()

                    # attempt, autocorrelation function
                    #acf_array[counter_d, :] = acf(ddata_res.DATA_VALUE, nlags=nlags)


                    # attempt, Lomb-Scargle Periodogram
                    vals = lombscargle(ddata.index, ddata.DATA_VALUE, freqs=freqs, normalize=True, precenter=True)
                    prdgram_array[counter_d, :]=[v/max(vals) for v in vals]
                    counter_d+=1
                # plot current prdgram_array
                ax=plt.subplot(nrows, ncols, count_rows*2+count_cols)
                im=plt.imshow(prdgram_array, cmap=cmp,extent= extent, aspect='auto')
                plt.grid()
                unique_d.reverse()
                yticks= [yt+1 for yt in list(range(0, len(unique_d)))]
                plt.gca().set_yticks(
                    yticks
                )
                plt.gca().set_yticklabels(
                    [str(d).replace("-", "") for d in unique_d],
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
                # save current axis
                exec('ax_'+str(count_cols)+'_list.append(ax)')
                # customize axes labels etc.
                plt.yticks(fontsize= fs)
                if count_rows*2+count_cols not in [nrows*ncols-1, nrows*ncols]:
                    ax.set_xticklabels([])
                else:
                    plt.xticks(fontsize=fs)
                    plt.colorbar(im,  label = 'normierte Energiedichte', orientation='horizontal')
                    plt.xlabel('Periode [h]')
                    exec('axes=ax_'+str(count_cols)+'_list')
                    # if p == 'WT':
                    #     cblabel = list(stationsdict.values())
                    #     cblabel.reverse()
                    #     cblabel='      '.join(cblabel)
                    #     plt.colorbar(im,ax=axes, pad=0.025, label=cblabel)   # noqa: F821
                    # else:
                    #     plt.colorbar(im,ax=axes, pad=0.025) # noqa: F821
                    
                    fig.text(x=-0.05, y=0.6, s='Wassertiefe [m]', va='center', rotation='vertical')
                count_cols+=1 
            count_rows+=1
    fig.savefig(savefigpath, bbox_inches=bbox_inches)
    return


plot_periodograms()