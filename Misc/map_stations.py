# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:01:42 2024

@author: bd2107
"""

import matplotlib
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import os

import netCDF4
import sys

abspath = os.path.abspath(os.path.join("B_ExpDatAn", "Code"))
sys.path.insert(0, abspath)
from common_variables import cm, fs, bbox_inches  # noqa: E402

def map_stations(
    lat=[54.5981666667, 54.4995, 54.9983333333, 54.6827833333],
    lon=[11.1491666667, 10.2736666667, 6.3515, 6.75445],
):
    # define everything that has to do with the resulting figure
    plt.rcParams["figure.figsize"][1] = 10 * cm
    plt.figure()#layout=layout)
    savefigpath = os.path.join("Misc", "map_stations.png")
    dlon = 0.5
    dlat = 1
    cmap = matplotlib.cm.bone
    cmap.set_bad("white", 1.0)

    # extent of map
    extent = [min(lon) - dlon, max(lon) + dlon, min(lat) - dlat, max(lat) + dlat]

    # load GEBCO
    gebcofile = os.path.join(
        "Misc",
        "GEBCO_14_Aug_2024_1671b01b7c83",
        "gebco_2024_n67.0_s50.0_w10.0_e30.0.nc",
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
    axx = plt.axes(projection=ccrs.PlateCarree())
    im = plt.imshow(
        np.flipud(elev), interpolation=None, extent=extent, cmap=cmap, aspect="auto"
    )

    # labels etc-
    gridlines = axx.gridlines(draw_labels=True)
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    axx.set_xlabel("Geographische Länge [°]", fontsize=fs)
    axx.set_ylabel("Geographische Breite [°]", fontsize=fs)
    plt.clim(-60, 0)


    # Add station positions
    plt.plot(lon, lat, "rx")

    # Add coastline    
    axx.add_feature(cfeature.COASTLINE)
    
    #  colorbar
    cb = plt.colorbar(im, label = "Wassertiefe [m]", shrink=1.0, ax=axx) 
    cb.ax.tick_params(labelsize=fs)
    labels = [str(ll).replace('-', '') for ll in cb.get_ticks()]
    cb.ax.set_yticks(cb.get_ticks(), labels, fontsize=fs)
    
    plt.savefig(savefigpath, bbox_inches=bbox_inches)
    plt.show()
    return 
map_stations()
