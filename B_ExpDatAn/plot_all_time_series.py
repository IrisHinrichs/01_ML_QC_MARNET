# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:13:09 2024

@author: Iris
"""
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime as dt
datapath = '../A_Data/'
stations = ['Fehmarn Belt Buoy', 'Kiel Lighthouse', 
            'North Sea Buoy II', 'North Sea Buoy III']

params = ['WT', 'SZ']
time_period = '20200101_20240630'
xlims = [dt(2020,1,1,0,0),dt(2024,6,3,23,59)]

cm = 1/2.54  # centimeters in inches
figsize= (18*cm,27*cm)
fig = plt.figure(figsize=figsize)

def make_figure():
    count_rows=0
    nrows = len(stations)
    ncols=len(params)
    for st in stations:
        count_cols=1
        for p in params:
            filestr = datapath+st.replace(' ', '_')+'_'+time_period+'_'+p+'_all_.csv'
            data=read_station_data(filestr=filestr)
            for d in set(data['Z_LOCATION']):
                ddata = data[data['Z_LOCATION']==d] # entries corresponding to depth leve d
                plt.subplot(nrows, ncols, count_rows*2+count_cols)
                plt.plot(ddata, '+')
                plt.title(st)
                plt.legend(str(d))
            count_cols+=1 
        count_rows+=1    
            
def read_station_data(filestr):
    usecols=['TIME_VALUE','Z_LOCATION','DATA_VALUE', 'QF1', 'QF3']
    data = pd.read_csv(filestr, usecols=usecols)
    data['TIME_VALUE'] = pd.to_datetime(data['TIME_VALUE'], format="%Y-%m-%d %H:%M:%S")
    data = data.set_index('TIME_VALUE')
    return data
def main():
    make_figure()
    
main()