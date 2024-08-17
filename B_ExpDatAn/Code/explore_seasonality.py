# -*- coding: utf-8 -*-
import matplotlib as mtpl
import numpy as np
from scipy.signal import lombscargle
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
from statsmodels.tsa.stattools import acf

import os
import sys

os.chdir(sys.path[0]) # change to modules parent directory

nlags = 96 # for autocorrelation

# for Lomb-Scargle Periodogram
periods = np.linspace(1,30, 30)
freqs=[np.pi*2/p for p in periods]


count_rows=0
nrows = len(stations)
ncols=len(params)
for st in stations:
        count_cols=1
        for p in params:

            #define colormap
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
               ddata = data[data['Z_LOCATION']==d] # entries corresponding to depth level d
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
            # plot current acf_array
            plt.figure()
            plt.imshow(prdgram_array, cmap=cmp,extent= extent, aspect='auto')
            plt.colorbar()
            plt.title(st+', '+p)
            print('ende')