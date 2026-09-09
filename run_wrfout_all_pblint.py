#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:17:28 2026

@author: jordaneissner
"""
import glob
import re
from netCDF4 import Dataset
from get_wrfout_all_pblint import get_all_pblint
    

vyear = "2022"
keep_hours = {'00', '06', '12', '18'}

path = "wrfout_d03_"+vyear+"*"
dr = 1000. #grid spacing
filename = 'wrfout_d03_hw1_pblint_6h.nc' # output file name

#create dataset
files = sorted(glob.glob(path))
# filter based on the hour in the filename
# matches "wrfout_d01_YYYY-MM-DD_HH:MM:SS"
subset = [
    f for f in files
    if re.search(r"_(\d{2}):\d{2}:\d{2}", f) and
       re.search(r"_(\d{2}):\d{2}:\d{2}", f).group(1) in keep_hours
]

#print(subset)
ds = [Dataset(f) for f in subset] # or "for f in files"

get_all_pblint(ds, dr, filename)