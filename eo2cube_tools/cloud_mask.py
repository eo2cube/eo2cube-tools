## cloud_mask.py

"""
This file includes python functions for masking clouds from Sentinel-2 data for Bukina Faso in the Open Data Cube environment.

Available functions:

    ssmask
    s2cmask

Last modified: September 2021
Author: KaHeiChow

"""

## Comment: remaining bug need to be fixed -
## 1) A partially defect scene (eg. a corner is missing) will threw error in running the function
## 2) strength parameter adjustment is not properly working yet (for now stick with the default option)
## 3) NDVI and MNDWI will be integrated into the function to improve accuracy for partial clouds condition

# Import required libraries
import datacube
import xarray as xr
import pandas as pd
import geopandas as gpd
from datetime import datetime
import warnings; warnings.simplefilter('ignore')
import imp
from time import time
import matplotlib.pyplot as plt
import numpy as np
from odc.ui import with_ui_cbk
import seaborn as sns
from dask.distributed import Client, LocalCluster
import sys

# import specific libraries for NDCI analysis
from scipy.stats import skew
from scipy import stats
from scipy.stats import norm, kurtosis
from scipy.signal import find_peaks
from scipy.signal import argrelmin

def ssmask_snow(dataset,strength=0.4):
    """
    Adding cloud variable to detect cloud cover in the single scene.
    Description
    ----------
    Generating cloud mask.
    Parameters
    ----------
    dataset: xr.Dataset
        dataset with a single time step, including bands "water_vapour","red","green","blue","nir", and "swir1". 
        For multiple time steps, use s2cmask() instead.
    stength: float
        A number from 0 to 1 to determine the intensity for cloud masking. A higher number allows masking of less probable cloud pixels.
    Returns
    -------
    dataset: xr.Dataset
        The input xr.Dataset with a new variable "cloud" of values 0, 1, or nan (0 = Not cloud, 1 = cloud, nan = defect pixel).
    """
    
    # error catching
    assert strength >= 0 and strength <=1,"strength argument should be in range 0 to 1"
    
    assert isinstance(dataset, xr.Dataset),"Input has to be a xarray.Dataset."
    
    try:
        dataset.red
        dataset.green
        dataset.blue
        dataset.nir
        dataset.swir1
    except Exception:
        print("RGB/NIR/SWIR1 bands not found.")
    
    try:
        dataset.water_vapour
    except Exception:
        print("water_vapour band not found.")
    
    # storing variable for defect bands
    # if half or more of all pixels are zeros or missing values, the scene is treated as defect
    numr = np.nan_to_num(dataset.red.values.flatten(), nan = 0.0)
    numg = np.nan_to_num(dataset.green.values.flatten(), nan = 0.0)
    numb = np.nan_to_num(dataset.blue.values.flatten(), nan = 0.0)
    defectr = (np.median(numr) == 0)
    defectg = (np.median(numg) == 0)
    defectb = (np.median(numb) == 0)
    
    # for defect scene, return nan for cloud and indices
    if any([defectr,defectg,defectb]) == True:
        dataset = dataset.assign(cloud = (dataset.blue*0)*np.nan)
        dataset = dataset.assign(MNDWI = (dataset.blue*0)*np.nan)
        dataset = dataset.assign(NDVI = (dataset.blue*0)*np.nan)
        dataset = dataset.assign(NDCI = (dataset.blue*0)*np.nan)
        return dataset
    
    # for non-defect scene
    
    # calculating MNDWI, NDVI and NDCI spectral indices for cloud detection
    dataset = dataset.assign(
        MNDWI = (dataset["green"]/10000 - dataset["swir1"]/10000)/(dataset["green"]/10000 + dataset["swir1"]/10000),
        NDVI = (dataset["nir"]/10000 - dataset["red"]/10000)/(dataset["nir"]/10000 + dataset["red"]/10000),
        NDCI = (dataset["blue"]/10000 - dataset["nir"]/10000)/(dataset["blue"]/10000 + dataset["nir"]/10000)
    )
    
    dataset = dataset.assign(
        NDSI_exp = dataset["MNDWI"] - (0.0652 * np.exp(1.8069 * dataset["NDVI"])),
        NDSI_div = dataset["MNDWI"] - ((dataset["NDVI"] - 0.2883) / (-0.4828)),
        NDWI = (dataset["green"]/10000 - dataset["nir"]/10000)/(dataset["green"]/10000 + dataset["nir"]/10000)
    )
    
    # get snow classification
    
    snow_condition1 = np.logical_and(dataset["green"]/10000 > 0.11, dataset["nir"]/10000 > 0.10)
    snow_condition2 = (dataset["MNDWI"] > 0.4)
    snow_condition3 = np.logical_and(dataset["NDVI"] > 0.25, dataset["NDSI_exp"] >= 0)
    snow_condition4 = np.logical_and(dataset["NDVI"] >= 0.2, 
                                     np.logical_and(dataset["NDVI"] < 0.25, dataset["NDSI_div"] >= 0))
    
    snow_condition234 = np.logical_or(np.logical_or(snow_condition2, snow_condition3), snow_condition4)
    snow_condition = np.logical_and(snow_condition1, snow_condition234)
    
    dataset = dataset.assign(snow = xr.where(snow_condition, 1.0, 0.0))
    
    # get statistics for NDCI and water vapour distribution
    
    # remove nan values from array to avoid errors from calculating statistics
    ndci_arr = dataset.NDCI.values.flatten()
    ndci_arr = ndci_arr[~np.isnan(ndci_arr)]
    vapour_arr = dataset.water_vapour.values.flatten()
    vapour_arr = vapour_arr[~np.isnan(vapour_arr)]
    
    # store statistics to variables
    NDCI_min = np.quantile(ndci_arr,0.05) # minimum NDCI allowing 5% outliners
    NDCI_max = np.quantile(ndci_arr,0.95) # maximum NDCI allowing 5% outliners
    NDCI_std = dataset.NDCI.values.std() # NDCI standard deviation
    vapour_median = np.median(vapour_arr) # median for water vapour
    NDCI_skew = skew(ndci_arr) # NDCI skewness: how far is NDCI away from normal distribution
    NDCI_median = np.median(ndci_arr) # NDCI median
    NDCI_kur = kurtosis(ndci_arr) # NDCI kurtosis: depends on the shape of distribution
    
    # calculating the peak of NDCI histogram and detect number of peaks
    counts, bin_edges = np.histogram(ndci_arr) # get frequency count of every NDCI range groups
    peaks,_ = find_peaks(counts, prominence=len(ndci_arr)/50) # get all peaks (between increasing and decreasing counts)
    npeak = len(peaks.flatten()) # get the number of peaks
    
    # Classified Scene: All Clouds
    if NDCI_min >= -0.1:
        dataset = dataset.assign(cloud = (dataset.blue*0 + 1))
    
    # Classified Scene: Mostly clouds
    elif NDCI_min < -0.1 and NDCI_min >= -0.3: # high (> -0.3) NDCI for all pixels
        if bin_edges[peaks[0]] < -0.3:
            thres = bin_edges[peaks[0]]
        else:
            thres = -0.3

        condition = dataset.NDCI >= thres # criteria for cloud classification
        dataset = dataset.assign(cloud = xr.where(condition, 1.0, 0.0)) # assign cloud variable in the dataset
        
    # Classified Scene: Mostly clouds
    # other condition for scene to be classify as mostly clouds: lower NDCI but histogram highly skewed towards high values
    # kurtosis values restricted to peak to be sharp for cloudy conditions
    # if clouds are dominant in scene, there should be only one peak in the histogram
    elif NDCI_min < -0.3 and NDCI_median > -0.35 and NDCI_kur > -1 and NDCI_skew < 0 and npeak == 1:
        if bin_edges[peaks[0]] < -0.3:
            thres = bin_edges[peaks[0]]
        else:
            thres = -0.3
        condition = dataset.NDCI >= thres
        dataset = dataset.assign(cloud = xr.where(condition, 1.0, 0.0))
    
    # Classified Scene: Clear sky
    # either low NDCI for all pixels or low median with really sharp peak for sunny conditionsf
    elif NDCI_max < -0.3 or (NDCI_kur > 1.5 and NDCI_median < -0.4):
        condition = np.logical_and(dataset.NDCI >= -0.3, dataset.water_vapour > vapour_median*1.1)
        dataset = dataset.assign(cloud = xr.where(condition, 1.0, 0.0))
    
    # Classified Scene: Partly clouds
    else:
        counts, bin_edges = np.histogram(dataset.NDCI.values.flatten(), bins=30)
        if npeak <= 1:
            peak = np.argmax(counts)
            mode = (bin_edges[peak] + bin_edges[peak+1])/2 # check dominant NDCI value

            par = stats.percentileofscore(dataset.NDCI.values.flatten(), mode) # get quantile for dominant NDCI value

            factor = 1 - NDCI_std # define fector to adjust NDCI threshold 
            # the more variant NDCI, the more conservative to define the end of NDCI cluster
            adj_factor = (100 - par)*(factor/50) # tune the value to make sure percentile lies within 0% and 100%
            refine = (par + adj_factor)/100
            NDCI_thres = np.quantile(dataset.NDCI.values.flatten(),refine) # calculate the NDCI threshold

        else:
            min_indices = argrelmin(counts)[0]
            ind_min = np.argmin(counts[min_indices])
            ind_hist = min_indices[ind_min]
            NDCI_thres = (bin_edges[ind_hist] + (-0.4))/2
            strength = 0.5

        # default values false for neg: indicate negative strength
        neg = False
        
        # adjust strength depends on NDCI skewness (the proxy for cloud proportion)
        # to make sure consistent masking independent of weather condition
        if NDCI_skew > 1:
            strength = strength + 0.3
            if strength > 1:
                strength = 1 # make sure strength lies within 0 and 1
        elif NDCI_skew > 0 and NDCI_skew < 0.2: # less strength when there are few clouds
            strength = strength - 0.2
            if strength < 0:
                neg = True
                negative_adj = strength
                strength = 0
        elif NDCI_skew < -0.4: # increase strength when there are lots of clouds
            strength = strength + 0.3
            if strength > 1:
                strength = 1
            
        # apply strength to adjust the NDCI threshold
        if NDCI_thres <= 0:
            NDCI_thres_adjust = (NDCI_thres - 1) * strength * 2
            NDCI_thres_adjust = NDCI_thres_adjust + (strength * 2)
        elif NDCI_thres > 0:
            NDCI_thres_adjust = (NDCI_thres + 1) * strength * -2
            NDCI_thres_adjust = NDCI_thres_adjust - (strength * -2)
        if neg == True:
            NDCI_thres_adjust = NDCI_thres_adjust + negative_adj/5 
        
        # use NDCI sknewness as the proxy for cloudyness of the scene
        # which is used for second condition depending on the water vapour value
        if NDCI_skew >= -0.5 and NDCI_skew < -0.3:
            condition1 = np.logical_and(dataset.NDCI > NDCI_thres_adjust,
                                        dataset.water_vapour > np.quantile(dataset.water_vapour.values,0.45))
        elif NDCI_skew < -0.5: # lower water vapour threshold for relatively cloudy condition
            condition1 = np.logical_and(dataset.NDCI > NDCI_thres_adjust,
                                        dataset.water_vapour > np.quantile(dataset.water_vapour.values,0.3))
        else: # higher water vapour threshold for relatively less cloudy condition
            condition1 = np.logical_and(dataset.NDCI > NDCI_thres_adjust,
                                        dataset.water_vapour > np.quantile(dataset.water_vapour.values,0.5))
        condition2 = (dataset.NDVI < 0.2) # remove cloud pixels if vegetation presents in the pixel
        condition = np.logical_and(condition1, condition2)
        dataset = dataset.assign(cloud = xr.where(condition, 1.0, 0.0)) # add cloud variable in the dataset
    
    dataset["snow"] = dataset["snow"].rolling(latitude=3, longitude=3, center=True).min()
    dataset["snow"] = dataset["snow"].rolling(latitude=3, longitude=3, center=True).max()
    dataset["snow_buffer"] = dataset["snow"].rolling(latitude=3, longitude=3, center=True).max()
    final_adjust = np.logical_and(dataset.cloud == 1, dataset.snow_buffer == 0)
    dataset = dataset.assign(cloud2 = xr.where(final_adjust, 1.0, 0.0))
        
    # use rolling minimum to remove cloud noise of size smaller than four pixels
    # shrink every cloud object
    dataset["cloud_fill"] = dataset["cloud2"].rolling(latitude=2, longitude=2, center=True).min()
    # For the remaining large cloud object, expand them again
    dataset["cloud_fill"] = dataset["cloud_fill"].rolling(latitude=2, longitude=2, center=True).max()
    
    #dataset["cloud_fill"] = dataset["cloud_fill"].rolling(latitude=5, longitude=5, center=True).min()
    #dataset["cloud_fill"] = dataset["cloud_fill"].rolling(latitude=5, longitude=5, center=True).max()
    
    # Fill the empty edge with original cloud variable
    dataset["cloud_fill"] = dataset["cloud_fill"].fillna(value = dataset.cloud)
    # Set the updated value in cloud variable and delete unneeded variable
    dataset = dataset.drop("cloud")
    dataset = dataset.drop("cloud2")
    dataset["cloud"] = dataset.cloud_fill
    dataset = dataset.drop("cloud_fill")
    dataset = dataset.drop("snow_buffer")
        
    # drop other unneeded variables
    #dataset = dataset.drop(["NDVI","MNDWI","NDCI"],errors='ignore')
        
    return dataset

def ssmask(dataset,strength=0.4):
    """
    Adding cloud variable to detect cloud cover in the single scene.
    Description
    ----------
    Generating cloud mask.
    Parameters
    ----------
    dataset: xr.Dataset
        dataset with a single time step, including bands "water_vapour","red","green","blue","nir", and "swir1". 
        For multiple time steps, use s2cmask() instead.
    stength: float
        A number from 0 to 1 to determine the intensity for cloud masking. A higher number allows masking of less probable cloud pixels.
    Returns
    -------
    dataset: xr.Dataset
        The input xr.Dataset with a new variable "cloud" of values 0, 1, or nan (0 = Not cloud, 1 = cloud, nan = defect pixel).
    """
    
    # error catching
    assert strength >= 0 and strength <=1,"strength argument should be in range 0 to 1"
    
    assert isinstance(dataset, xr.Dataset),"Input has to be a xarray.Dataset."
    
    try:
        dataset.red
        dataset.green
        dataset.blue
        dataset.nir
        dataset.swir1
    except Exception:
        print("RGB/NIR/SWIR1 bands not found.")
    
    try:
        dataset.water_vapour
    except Exception:
        print("water_vapour band not found.")
    
    # storing variable for defect bands
    # if half or more of all pixels are zeros or missing values, the scene is treated as defect
    numr = np.nan_to_num(dataset.red.values.flatten(), nan = 0.0)
    numg = np.nan_to_num(dataset.green.values.flatten(), nan = 0.0)
    numb = np.nan_to_num(dataset.blue.values.flatten(), nan = 0.0)
    defectr = (np.median(numr) == 0)
    defectg = (np.median(numg) == 0)
    defectb = (np.median(numb) == 0)
    
    # for defect scene, return nan for cloud and indices
    if any([defectr,defectg,defectb]) == True:
        dataset = dataset.assign(cloud = (dataset.blue*0)*np.nan)
        dataset = dataset.assign(MNDWI = (dataset.blue*0)*np.nan)
        dataset = dataset.assign(NDVI = (dataset.blue*0)*np.nan)
        dataset = dataset.assign(NDCI = (dataset.blue*0)*np.nan)
        return dataset
    
    # for non-defect scene
    
    # calculating MNDWI, NDVI and NDCI spectral indices for cloud detection
    dataset = dataset.assign(
        MNDWI = (dataset["green"]/10000 - dataset["swir1"]/10000)/(dataset["green"]/10000 + dataset["swir1"]/10000),
        NDVI = (dataset["nir"]/10000 - dataset["red"]/10000)/(dataset["nir"]/10000 + dataset["red"]/10000),
        NDCI = (dataset["blue"]/10000 - dataset["nir"]/10000)/(dataset["blue"]/10000 + dataset["nir"]/10000)
    )
    
    # get statistics for NDCI and water vapour distribution
    
    # remove nan values from array to avoid errors from calculating statistics
    ndci_arr = dataset.NDCI.values.flatten()
    ndci_arr = ndci_arr[~np.isnan(ndci_arr)]
    vapour_arr = dataset.water_vapour.values.flatten()
    vapour_arr = vapour_arr[~np.isnan(vapour_arr)]
    
    # store statistics to variables
    NDCI_min = np.quantile(ndci_arr,0.05) # minimum NDCI allowing 5% outliners
    NDCI_max = np.quantile(ndci_arr,0.95) # maximum NDCI allowing 5% outliners
    NDCI_std = dataset.NDCI.values.std() # NDCI standard deviation
    vapour_median = np.median(vapour_arr) # median for water vapour
    NDCI_skew = skew(ndci_arr) # NDCI skewness: how far is NDCI away from normal distribution
    NDCI_median = np.median(ndci_arr) # NDCI median
    NDCI_kur = kurtosis(ndci_arr) # NDCI kurtosis: depends on the shape of distribution
    
    # calculating the peak of NDCI histogram and detect number of peaks
    counts, bin_edges = np.histogram(ndci_arr) # get frequency count of every NDCI range groups
    peaks,_ = find_peaks(counts, prominence=len(ndci_arr)/50) # get all peaks (between increasing and decreasing counts)
    npeak = len(peaks.flatten()) # get the number of peaks
    
    # Classified Scene: All Clouds
    if NDCI_min >= -0.1:
        dataset = dataset.assign(cloud = (dataset.blue*0 + 1))
    
    # Classified Scene: Mostly clouds
    elif NDCI_min < -0.1 and NDCI_min >= -0.3: # high (> -0.3) NDCI for all pixels
        if bin_edges[peaks[0]] < -0.3:
            thres = bin_edges[peaks[0]]
        else:
            thres = -0.3

        condition = dataset.NDCI >= thres # criteria for cloud classification
        dataset = dataset.assign(cloud = xr.where(condition, 1.0, 0.0)) # assign cloud variable in the dataset
        
    # Classified Scene: Mostly clouds
    # other condition for scene to be classify as mostly clouds: lower NDCI but histogram highly skewed towards high values
    # kurtosis values restricted to peak to be sharp for cloudy conditions
    # if clouds are dominant in scene, there should be only one peak in the histogram
    elif NDCI_min < -0.3 and NDCI_median > -0.35 and NDCI_kur > -1 and NDCI_skew < 0 and npeak == 1:
        if bin_edges[peaks[0]] < -0.3:
            thres = bin_edges[peaks[0]]
        else:
            thres = -0.3
        condition = dataset.NDCI >= thres
        dataset = dataset.assign(cloud = xr.where(condition, 1.0, 0.0))
    
    # Classified Scene: Clear sky
    # either low NDCI for all pixels or low median with really sharp peak for sunny conditionsf
    elif NDCI_max < -0.3 or (NDCI_kur > 1.5 and NDCI_median < -0.4):
        condition = np.logical_and(
            dataset.NDCI >= -0.3, dataset.water_vapour > vapour_median*1.1
        )
        dataset = dataset.assign(cloud = xr.where(condition, 1.0, 0.0))
    
    # Classified Scene: Partly clouds
    else:
        counts, bin_edges = np.histogram(ndci_arr, bins=30)
        if npeak <= 1:
            peak = np.argmax(counts)
            mode = (bin_edges[peak] + bin_edges[peak+1])/2 # check dominant NDCI value

            par = stats.percentileofscore(ndci_arr, mode) # get quantile for dominant NDCI value

            factor = 1 - NDCI_std # define fector to adjust NDCI threshold 
            # the more variant NDCI, the more conservative to define the end of NDCI cluster
            adj_factor = (100 - par)*(factor/50) # tune the value to make sure percentile lies within 0% and 100%
            refine = (par + adj_factor)/100
            NDCI_thres = np.quantile(ndci_arr,refine) # calculate the NDCI threshold

        else:
            min_indices = argrelmin(counts)[0]
            ind_min = np.argmin(counts[min_indices])
            ind_hist = min_indices[ind_min]
            NDCI_thres = (bin_edges[ind_hist] + (-0.4))/2
            strength = 0.5

        # default values false for neg: indicate negative strength
        neg = False
        
        # adjust strength depends on NDCI skewness (the proxy for cloud proportion)
        # to make sure consistent masking independent of weather condition
        if NDCI_skew > 1:
            strength = strength + 0.3
            if strength > 1:
                strength = 1 # make sure strength lies within 0 and 1
        elif NDCI_skew > 0 and NDCI_skew < 0.2: # less strength when there are few clouds
            strength = strength - 0.2
            if strength < 0:
                neg = True
                negative_adj = strength
                strength = 0
        elif NDCI_skew < -0.4: # increase strength when there are lots of clouds
            strength = strength + 0.3
            if strength > 1:
                strength = 1
            
        # apply strength to adjust the NDCI threshold
        if NDCI_thres <= 0:
            NDCI_thres_adjust = (NDCI_thres - 1) * strength * 2
            NDCI_thres_adjust = NDCI_thres_adjust + (strength * 2)
        elif NDCI_thres > 0:
            NDCI_thres_adjust = (NDCI_thres + 1) * strength * -2
            NDCI_thres_adjust = NDCI_thres_adjust - (strength * -2)
        if neg == True:
            NDCI_thres_adjust = NDCI_thres_adjust + negative_adj/5 
        
        # use NDCI sknewness as the proxy for cloudyness of the scene
        # which is used for second condition depending on the water vapour value
        if NDCI_skew >= -0.5 and NDCI_skew < -0.3:
            condition1 = np.logical_and(dataset.NDCI > NDCI_thres_adjust,
                                        dataset.water_vapour > np.quantile(dataset.water_vapour.values,0.45))
        elif NDCI_skew < -0.5: # lower water vapour threshold for relatively cloudy condition
            condition1 = np.logical_and(dataset.NDCI > NDCI_thres_adjust,
                                        dataset.water_vapour > np.quantile(dataset.water_vapour.values,0.3))
        else: # higher water vapour threshold for relatively less cloudy condition
            condition1 = np.logical_and(dataset.NDCI > NDCI_thres_adjust,
                                        dataset.water_vapour > np.quantile(dataset.water_vapour.values,0.5))
        condition2 = (dataset.NDVI < 0.2) # remove cloud pixels if vegetation presents in the pixel
        condition = np.logical_and(condition1, condition2)
        dataset = dataset.assign(cloud = xr.where(condition, 1.0, 0.0)) # add cloud variable in the dataset
        
        # use rolling minimum to remove cloud noise of size smaller than four pixels
        # shrink every cloud object
        dataset["cloud_fill"] = dataset["cloud"].rolling(latitude=2, longitude=2, center=True).min()
        # For the remaining large cloud object, expand them again
        dataset["cloud_fill"] = dataset["cloud_fill"].rolling(latitude=2, longitude=2, center=True).max()
        # Fill the empty edge with original cloud variable
        dataset["cloud_fill"] = dataset["cloud_fill"].fillna(value = dataset.cloud)
        # Set the updated value in cloud variable and delete unneeded variable
        dataset = dataset.drop("cloud")
        dataset["cloud"] = dataset.cloud_fill
        dataset = dataset.drop("cloud_fill")
        
        # drop other unneeded variables
        #dataset = dataset.drop(["NDVI","MNDWI","NDCI"],errors='ignore')
        
    return dataset


def s2cmask(ds, strength=0.4):
    """
    Adding cloud variable to detect cloud cover in multiple scenes.
    Description
    ----------
    Generating cloud mask.
    Parameters
    ----------
    dataset: xr.Dataset
        dataset with mulitple time steps, including bands "water_vapour","red","green","blue","nir", and "swir1". 
        For single time steps, use ssmask() instead.
    stength: float
        A number from 0 to 1 to determine the intensity for cloud masking. A higher number allows masking of less probable cloud pixels.
    Returns
    -------
    dataset: xr.Dataset
        The input xr.Dataset with a new variable "cloud" of values 0, 1, or nan (0 = Not cloud, 1 = cloud, nan = defect pixel).
    """
    collection = []
    [collection.append(ssmask(ds.sel(time = ts), strength=strength)) for ts in ds.time.to_index()]
    maskds = xr.concat(collection, dim = "time")
    return maskds

def s2cmask_snow(ds, strength=0.4):
    """
    Adding cloud variable to detect cloud cover in multiple scenes.
    Description
    ----------
    Generating cloud mask.
    Parameters
    ----------
    dataset: xr.Dataset
        dataset with mulitple time steps, including bands "water_vapour","red","green","blue","nir", and "swir1". 
        For single time steps, use ssmask() instead.
    stength: float
        A number from 0 to 1 to determine the intensity for cloud masking. A higher number allows masking of less probable cloud pixels.
    Returns
    -------
    dataset: xr.Dataset
        The input xr.Dataset with a new variable "cloud" of values 0, 1, or nan (0 = Not cloud, 1 = cloud, nan = defect pixel).
    """
    collection = []
    [collection.append(ssmask_snow(ds.sel(time = ts), strength=strength)) for ts in ds.time.to_index()]
    maskds = xr.concat(collection, dim = "time")
    return maskds