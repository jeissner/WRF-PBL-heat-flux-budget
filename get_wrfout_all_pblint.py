import xarray as xr
import numpy as  np
import wrf


def density(TH, P):
    Rd = 287.0
    cp = 1004.0
    T = TH * (P / 100000.0)**(Rd / cp)
    return(P / (Rd * T))

def pbl_integral(da, rho, z, pblh, zdim):
    cp = 1004 #J kg-1 K-1
    
    mask = z <= pblh
    
    da_masked = da.where(mask)
    rho_masked = rho.where(mask)
    
    dz = z.diff(zdim)
    
    da_mid = da_masked.isel({zdim: slice(0, -1)})
    rho_mid = rho_masked.isel({zdim: slice(0, -1)})
    
    integral = (cp * rho_mid * da_mid * dz).sum(zdim)
    return(integral)

def pbl_mean(da, rho, z, pblh, zdim):
    mask = z <= pblh

    da_masked = da.where(mask)
    rho_masked = rho.where(mask)

    dz = z.diff(zdim)

    da_mid = da_masked.isel({zdim: slice(0, -1)})
    rho_mid = rho_masked.isel({zdim: slice(0, -1)})

    num = (rho_mid * da_mid * dz).sum(zdim)
    den = (rho_mid * dz).sum(zdim)
    return(num/den)

def T_adv(T, u, v, dx, dy, dim1, dim2):

    dTdx = T.differentiate(dim1) / dx
    dTdy = T.differentiate(dim2) / dy

    adv_T = -(u * dTdx + v * dTdy)
    return(adv_T)


def T_adv_w(T, w, z, dim):  
    dz = z.diff(dim)
    T_mid = T.isel({dim: slice(0, -1)})
    w_mid = w.isel({dim: slice(0, -1)})
    
    dTdz = T_mid.differentiate(dim) / dz  
    adv_T = -w_mid * dTdz
    return(adv_T)


def get_all_pblint(ds, dr, filename):

    z = wrf.getvar(ds, "z", msl=False, timeidx=wrf.ALL_TIMES) 
    pblh = wrf.getvar(ds, "PBLH", timeidx=wrf.ALL_TIMES)
    k_pbl = abs(z-pblh).argmin(dim="bottom_top")
    
    U = wrf.getvar(ds, "ua", timeidx=wrf.ALL_TIMES)
    V = wrf.getvar(ds, "va", timeidx=wrf.ALL_TIMES)
    W = wrf.getvar(ds, "wa", timeidx=wrf.ALL_TIMES)
    T = wrf.getvar(ds, "theta", timeidx=wrf.ALL_TIMES)
    P = wrf.getvar(ds, "pressure", timeidx=wrf.ALL_TIMES)
    rho = density(T, P*100.)
    
    ## tendencies, cu, shcu, and mp are accumulated "A.." and have units K/hr; instantaneous "R.." variables have units K/s
    blten = wrf.getvar(ds, "RTHBLTEN", timeidx=wrf.ALL_TIMES)
    cuten = wrf.getvar(ds, "ATHCUTEN", timeidx=wrf.ALL_TIMES)
    shten = wrf.getvar(ds, "ATHSHTEN", timeidx=wrf.ALL_TIMES)
    mpten = wrf.getvar(ds, "ATHMPTEN", timeidx=wrf.ALL_TIMES)
    lwten = wrf.getvar(ds, "RTHRATLW", timeidx=wrf.ALL_TIMES)
    swten = wrf.getvar(ds, "RTHRATSW", timeidx=wrf.ALL_TIMES)
    # advten = wrf.getvar(ds, "RTHFTEN", timeidx=wrf.ALL_TIMES) # this is a sum of horizontal and vertical advections & onyl non-zero in d01 
    dthetadt = T.differentiate("Time", datetime_unit="s") #K/s
    
    ws=np.sqrt(U**2 + V**2)
    wd=np.degrees(np.arctan2(V, U))
    
    u_av = pbl_mean(U, rho, z, pblh, 'bottom_top')
    v_av = pbl_mean(V, rho, z, pblh, 'bottom_top')
    t_av = pbl_mean(T, rho, z, pblh, 'bottom_top')
    ws_av = pbl_mean(ws, rho, z, pblh, 'bottom_top')
    wd_av = pbl_mean(wd, rho, z, pblh, 'bottom_top')
    
    blten_int = pbl_integral(blten, rho, z, pblh, 'bottom_top') #K/s
    cuten_int = pbl_integral(cuten, rho, z, pblh, 'bottom_top')/3600. #K/hr to K/s
    shten_int = pbl_integral(shten, rho, z, pblh, 'bottom_top')/3600.  
    mpten_int = pbl_integral(mpten, rho, z, pblh, 'bottom_top')/3600.
    lwten_int = pbl_integral(lwten, rho, z, pblh, 'bottom_top')
    swten_int = pbl_integral(swten, rho, z, pblh, 'bottom_top')
    # advten_int = pbl_integral(advten, rho, z, pblh, 'bottom_top') 
    Ttend_int = pbl_integral(dthetadt, rho, z, pblh, 'bottom_top')
    
    # horizontal advection
    tadv2 = T_adv(T, U, V, dr, dr, "west_east", "south_north") #K/s #* 3600.
    tadv_int = pbl_integral(tadv2, rho, z, pblh, 'bottom_top')
    
    # vertical advection, use midpoints of each sigma layer
    tadvw = T_adv_w(T, W, z, 'bottom_top') #* 3600.
    z_mid = 0.5 * (z.isel(bottom_top=slice(0, -1)) +
                   z.isel(bottom_top=slice(1, None)))
    rho_mid = 0.5 * (rho.isel(bottom_top=slice(0, -1)) +
                   rho.isel(bottom_top=slice(1, None)))
    advw_int = pbl_mean(tadvw, rho_mid, z_mid, pblh, 'bottom_top')
    
    ds_out = xr.Dataset({
        "lw_tend": lwten_int,
        "sw_tend": swten_int,
        "bl_tend": blten_int,
        "mp_tend": mpten_int,
        "cu_tend": cuten_int,
        "shcu_tend": shten_int,
        # "adv_tend": advten_int,
        "adv_tend": tadv_int,
        "wadv_tend": advw_int,
        "T_tend": Ttend_int,
        "u_av": u_av,
        "v_av": v_av,
        "t_av": t_av,
        "ws_av": ws_av,
        "wd_av": wd_av
        
    })
    ds_out = ds_out.compute()
    ds_out.to_netcdf(filename)
    
    
    
    

