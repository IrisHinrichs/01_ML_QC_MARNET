# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:01:42 2024

@author: bd2107
"""




import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# from common_variables import  layout, cm, stationsdict,\
     # fs, fontdict, bbox_inches
     



def map_stations(lat= [54.5981666667,54.4995, 54.9983333333, 54.6827833333],\
                 lon=[11.1491666667,10.2736666667, 6.3515, 6.75445]):
    
    # define everything that has to do with the resulting figure
    # plt.rcParams['figure.figsize'][1]=10*cm
    # fig = plt.figure(layout=layout)
    # savefigpath = '../Figures/temporal_coverage.png'
    dlat = 1
    dlon = 1

    # load GEBCO

    # Add station positions
    plt.plot(lon, lat, '.')

    
    
    plt.show()
    
    return 
map_stations()
