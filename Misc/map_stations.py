# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:01:42 2024

@author: bd2107
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

import netCDF4
import sys

abspath = os.path.abspath("B_ExpDatAn/Code")
sys.path.insert(0, abspath)
from common_variables import cm, fs, bbox_inches, layout  # noqa: E402

def map_stations(
    lat=[54.5981666667, 54.4995, 54.9983333333, 54.6827833333],
    lon=[11.1491666667, 10.2736666667, 6.3515, 6.75445],
):
    # define everything that has to do with the resulting figure
    plt.rcParams["figure.figsize"][1] = 10 * cm
    fig= plt.figure(layout=layout)
    savefigpath = "Misc/map_stations.png"
    dlon = 0.5
    dlat = 1
    cmap = matplotlib.cm.bone
    cmap.set_bad("gray", 1.0)

    # extent of map
    extent = [min(lon) - dlon, max(lon) + dlon, min(lat) - dlat, max(lat) + dlat]

    # load GEBCO
    gebcofile = (
        "Misc\GEBCO_14_Aug_2024_1671b01b7c83\gebco_2024_n67.0_s50.0_w10.0_e30.0.nc"
    )
    rg = netCDF4.Dataset(gebcofile, "r", format="NETCDF4")
    elev = rg["elevation"][:]
    elev = elev.astype(float)
    glat = rg["lat"][:]
    glon = rg["lon"][:]
    rg.close()

    # reduce size of GEBCO elevation field to area surrounding the station's positions
    indlon = np.where((glon >= extent[0]) & (glon <= extent[1]))
    indlat = np.where((glat >= extent[2]) & (glat <= extent[3]))
    elev = elev[np.ix_(indlat[0], indlon[0])]

    # create landmask
    landmask = elev > 0
    # set landvalues to nan
    elev[landmask] = np.nan

    # plot elevation
    ax = plt.axes()
    im=plt.imshow(np.flipud(elev), interpolation=None, extent=extent, cmap=cmap)

    # Add station positions
    plt.plot(lon, lat, "rx")
    plt.xlabel("Geographische Länge [°]", fontsize=fs)
    plt.ylabel("Geographische Breite [°]", fontsize=fs)
    plt.clim(-60, 0)
    plt.grid()
    
    # colorbar
    #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    #cb = plt.colorbar(im, cax=cax, label = "Wassertiefe [m]") 
    cb = plt.colorbar(im, label = "Wassertiefe [m]", shrink=0.5) 
    cb.ax.tick_params(labelsize=fs)
    labels = [str(ll) for ll in range(60, -10, -10)]
    cb.ax.set_yticks(cb.get_ticks(), labels, fontsize=fs)
    
    plt.savefig(savefigpath, bbox_inches=bbox_inches)
    plt.show()
    return 
map_stations()
