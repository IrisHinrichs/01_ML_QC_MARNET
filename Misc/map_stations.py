# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:01:42 2024

@author: bd2107
"""


import geopandas as gpd
from shapely.geometry import LineString, Point
import pandas as pd
import matplotlib.pyplot as plt
from common_variables import  layout, cm, stationsdict,\
     fs, fontdict, bbox_inches


def map_stations(lat= [54.5981666667,54.4995, 54.9983333333, 54.6827833333],\
                 lon=[11.1491666667,10.2736666667, 6.3515, 6.75445]):
    
    # define everything that has to do with the resulting figure
    plt.rcParams['figure.figsize'][1]=10*cm
    fig = plt.figure(layout=layout)
    savefigpath = '../Figures/temporal_coverage.png'
    
    
    # load data for world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
   

    # turn positions into shapely.LineString object
    if len(lat) >1 and len(lon)>1:
        shobj = LineString(list(zip(lon, lat)))
    else:
        shobj = Point((lon[0],lat[0]))


    # check intersection between linestring and Copernicus Regions polygons
    check_table = pd.DataFrame(columns = ['check'], index = copernicus_regions.Name)
    for i in range(0, len(copernicus_regions)):
        polygon = copernicus_regions.geometry.loc[i]
        check_table['check'][i]= polygon.intersects(shobj)

    # get region abbrevation of intersecting region
    ind = list(check_table[check_table['check']==True].index)
    region = []
    for i in ind:
        reg = list(copernicus_regions.Region[copernicus_regions.Name==i])
        region.append(reg[0])
        
    if show_map:
        # plot map with regions and position values
        ax = world.plot(
            color=[0.8,0.8,0.8])
        copernicus_regions.plot(column = 'Name', facecolor = 'None',
                                     legend = list(copernicus_regions.Name),
                                     linewidth = 3, ax = ax)
        plt.plot(lon, lat, '.')
        plt.grid()
        plt.show()
    return list(set(region))
