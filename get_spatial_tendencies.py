
# put dataset together that is just LGA, MAN, JFK, and rural time series of all flux budget variables
# run after run_wrfout_all_pblint.py 

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import wrf
import xwrf
import matplotlib.dates as mdates

def spatial_integral(var, cpx, cpy, nx1,nx2,ny1,ny2):
    return(np.sum(var[:,cpx-nx1:cpx+nx2, cpy-ny1:cpy+ny2], axis=(1,2)))

def spatial_integraln(var, cpx, cpy, nx1,nx2,ny1,ny2):
    n = var[:,cpx-nx1:cpx+nx2, cpy-ny1:cpy+ny2].count(dim=("south_north", "west_east"))
    return([np.sum(var[:,cpx-nx1:cpx+nx2, cpy-ny1:cpy+ny2]*1*1, axis=(1,2)), n.values[0]])

def get_all(ds, mask, cpx, cpy, nx1, nx2, ny1, ny2):
    bl,n = spatial_integraln(ds.bl_tend.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)
    bl = bl/n
    sw = spatial_integral(ds.sw_tend.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)/n
    lw = spatial_integral(ds.lw_tend.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)/n
    mp = spatial_integral(ds.mp_tend.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)/n
    adv = spatial_integral(ds.adv_tend.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)/n
    wadv = spatial_integral(ds.wadv_tend.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)/n
    dT = spatial_integral(ds.T_tend.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)/n
    T = spatial_integral(ds.t_av.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)/n
    ws = spatial_integral(ds.ws_av.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)/n
    ds['wd_av'] = ds.wd_av % 360
    wd = spatial_integral(ds.wd_av.where(mask,drop=True), cpx, cpy, nx1, nx2, ny1, ny2)/n
    return([bl,sw,lw,mp,adv,wadv,dT,ws,wd,T])



file1 = "/Volumes/T9/WRFout/tendencies/wrfout_d03_hw1_pblint.nc"
ds1 = xr.open_dataset(file1)

mask1 = (ds1.LU_INDEX != 17).compute()#.squeeze(['Time'], drop=True)
lats = ds1.XLAT.values
lons = ds1.XLONG.values


#*** get analysis boxes
#JFK
dist = (lats - 40.64)**2 + (lons + 73.78)**2
#LGA 
dist2 = (lats - 40.78)**2 + (lons + 73.87)**2
#ESS 40.87522, -74.28135
dist3 = (lats - 40.88)**2 + (lons + 74.28)**2
#RUR 41.01, -74.55
dist4 = (lats - 41.01)**2 + (lons + 74.55)**2

j, i = np.unravel_index(dist.argmin(), dist.shape)
j2, i2 = np.unravel_index(dist2.argmin(), dist2.shape)
j3, i3 = np.unravel_index(dist3.argmin(), dist3.shape)
j4, i4 = np.unravel_index(dist4.argmin(), dist4.shape)

JFK = (i, j)
MAN = (72, 72) # manually did this because unravel is screwing up the points 
LGA = (i2, j2)
ESS = (i3, j3)
RUR = (i4, j4)

#JFK averages JFK + 9
bl_jfk1, sw_jfk1, lw_jfk1, mp_jfk1, adv_jfk1, wadv_jfk1, T_jfk1, ws_jfk1, wd_jfk1, th_jfk1 = get_all(ds1, mask1, JFK[1], JFK[0], 0, 9, 0, 9)

#MANH averages MAN +/- 4
bl_man1, sw_man1, lw_man1, mp_man1, adv_man1, wadv_man1, T_man1, ws_man1, wd_man1, th_man1 = get_all(ds1, mask1, MAN[1], MAN[0], 5, 4, 5, 4)

#LGA 
bl_lga1, sw_lga1, lw_lga1, mp_lga1, adv_lga1, wadv_lga1, T_lga1, ws_lga1, wd_lga1, th_lga1 = get_all(ds1, mask1, LGA[1], LGA[0], 9, 0, 0, 9)

#RUR
bl_rur1, sw_rur1, lw_rur1, mp_rur1, adv_rur1, wadv_rur1, T_rur1, ws_rur1, wd_rur1, th_rur1 = get_all(ds1, mask1, RUR[1], RUR[0], 9, 0, 0, 9)

sites = ["JFK", "LGA", "MAN", "RURAL"]

pbl = np.column_stack([
    bl_jfk1,
    bl_lga1,
    bl_man1,
    bl_rur1,
])  

sw = np.column_stack([
    sw_jfk1,
    sw_lga1,
    sw_man1,
    sw_rur1,
])  

lw = np.column_stack([
    lw_jfk1,
    lw_lga1,
    lw_man1,
    lw_rur1,
])  

mp = np.column_stack([
    mp_jfk1,
    mp_lga1,
    mp_man1,
    mp_rur1,
]) 

adv = np.column_stack([
    adv_jfk1,
    adv_lga1,
    adv_man1,
    adv_rur1,
]) 

wadv = np.column_stack([
    wadv_jfk1,
    wadv_lga1,
    wadv_man1,
    wadv_rur1,
]) 

dT = np.column_stack([
    T_jfk1,
    T_lga1,
    T_man1,
    T_rur1,
]) 

ws = np.column_stack([
    ws_jfk1,
    ws_lga1,
    ws_man1,
    ws_rur1,
]) 

wd = np.column_stack([
    wd_jfk1,
    wd_lga1,
    wd_man1,
    wd_rur1,
]) 

TH = np.column_stack([
    th_jfk1,
    th_lga1,
    th_man1,
    th_rur1,
])     
    
    
ds = xr.Dataset(
    data_vars={
        "RTHBLTEN": (("time", "site"), pbl, {
            "long_name": "Theta tendency from the PBL parameterization",
            "units": "W/m2",
        }),
        "RTHSWTEN": (("time", "site"), sw, {
           "long_name": "Theta tendency from the SW radiation parameterization",
           "units": "W/m2",
        }),
        "RTHLWTEN": (("time", "site"), lw, {
           "long_name": "Theta tendency from the LW radiation parameterization",
           "units": "W/m2",
        }),
        "ATHMPTEN": (("time", "site"), mp, {
           "long_name": "Theta tendency from the microphysics parameterization",
           "units": "W/m2",
       }),
        "RTHADVTEN": (("time", "site"), adv, {
           "long_name": "Theta tendency due to horizontal advection",
           "units": "W/m2",
       }),
        "RTHWADVTEN": (("time", "site"), wadv, {
           "long_name": "Theta tendency due to vertical advection",
           "units": "W/m2",
       }),
        "RTHTEN": (("time", "site"), dT, {
           "long_name": "Total theta tendency",
           "units": "W/m2",
       }),
        "PBL_SPD": (("time", "site"), ws, {
           "long_name": "Average PBL wind speed",
           "units": "m/s",
       }),
        "PBL_DIR": (("time", "site"), wd, {
           "long_name": "Average PBL wind direction",
           "units": "m/s",
       }),
        "PBL_TH": (("time", "site"), TH, {
           "long_name": "Average PBL potential temperature",
           "units": "K",
       }),
        
    
    },
    coords={
        "time": pd.to_datetime(ds1.Time),
        "site": sites,
    },
)

ds.to_netcdf("Tendencies_timeseries_hw1.nc")


