# -*- coding: utf-8 -*-
# Module for anomaly detection
import os
import sys
from utilities import get_filestring, read_station_data, get_path
from common_variables import stations, params, tlims

# set necessary paths
abspath = get_path(parents=2, dirname=os.path.join("B_ExpDatAn","Code"))
sys.path.insert(0, abspath)
abspath = get_path(parents=2, dirname=os.path.join("C_DataPreProc","Code"))
sys.path.insert(0, abspath)
from time_series_statistics import find_all_time_spans  # noqa: E402
from data_preprocessing import piecewise_linear_interpolation

def main():
    for st in stations:
        count_cols=1
        for p in params:
            filestr = get_filestring(st, p, tlims[0], tlims[1])
            data=read_station_data(filestr=filestr)
            # STEP I: piecewise interpolation of all time series
            data_interp = piecewise_linear_interpolation(data)
            
            # STEP II: piecewise anomaly detection
            # IDEE: piecewise_linear_interpolation anpassen und als Funktion definieren
            # unique depth levels of current station
            unique_d=list(set(data_interp.Z_LOCATION))
            unique_d.sort(reverse=True)
            for d in unique_d:
                ts = data_interp[data_interp["Z_LOCATION"]==d] # entries corresponding to depth level d
                time_spans = find_all_time_spans(time_vec=ts.index, tdel=10)

