# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:01:42 2024

@author: bd2107
"""


#import geopandas as gpd
#from shapely.geometry import LineString, Point
import cartopy.crs as ccrs
#import pandas as pd
import matplotlib.pyplot as plt
# from common_variables import  layout, cm, stationsdict,\
     # fs, fontdict, bbox_inches
     
#https://scitools.org.uk/cartopy/docs/latest/gallery/lines_and_polygons/ocean_bathymetry.html


def map_stations(lat= [54.5981666667,54.4995, 54.9983333333, 54.6827833333],\
                 lon=[11.1491666667,10.2736666667, 6.3515, 6.75445]):
    
    # define everything that has to do with the resulting figure
    # plt.rcParams['figure.figsize'][1]=10*cm
    # fig = plt.figure(layout=layout)
    # savefigpath = '../Figures/temporal_coverage.png'
    dlat = 1
    dlon = 1
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
   

    # # turn positions into shapely.LineString object
    # if len(lat) >1 and len(lon)>1:
    #     shobj = LineString(list(zip(lon, lat)))
    # else:
    #     shobj = Point((lon[0],lat[0]))

    # plot map with regions and position values
    # ax = world.plot(
        # color=[0.8,0.8,0.8])
    plt.plot(lon, lat, '.')
    plt.xlim([min(lon)-dlon, max(lon)+dlon])
    plt.ylim([min(lat)-dlat, max(lat)+dlat])
    plt.grid()
    plt.show()
    
    return 

map_stations()
