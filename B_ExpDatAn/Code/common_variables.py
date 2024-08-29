# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:11:41 2024

@author: Iris
"""
'''
Variables common for several scripts
'''

import datetime as dt  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


# station names
stations = ['Fehmarn Belt Buoy', 'Kiel Lighthouse', 
            'North Sea Buoy II', 'North Sea Buoy III']
stationsdict = {'Fehmarn Belt Buoy': 'Fehmarn', 
                'Kiel Lighthouse': "Leuchtturm Kiel", 
                'North Sea Buoy II': 'Nordsee II', 
                'North Sea Buoy III': 'Nordsee III'}

# parameter names
params = ['WT', 'SZ']
paramdict = {'WT': 'Wassertemperatur [° C]', 'SZ':'Salzgehalt []'}

# temporal limits
tlims = [dt.datetime(2020,1,1,0,0),dt.datetime(2024,6,30,23,59)]


# variables for customizing figures 
cm = 1/2.54  # conversion factor centimeters=>inches

# set figwidth and dpi for all figures
plt.rcParams["figure.figsize"][0] = 16.5*cm # figure width
plt.rcParams["figure.dpi"] = 300
layout = 'constrained'
bbox_inches = 'tight'

# fontdict for text in figure
fs = 10
fontdict = {'family': ['sans-serif'],
                     'variant': 'normal',
                     'weight': 'normal',
                     'stretch': 'normal',
                     'size': fs,
                     'math_fontfamily': 'dejavusans'}