## water_tsa.py
"""
This file includes multiple python functions for analysing surface water bodies using Sentinel-2 data for Bukina Faso in the Open Data Cube environment.

Available functions:

    get_bbox
    load_shp
    viz2d
    to_map
    sceneDisplay
    dataMask
    getQual
    vizQual
    pred_index
    water_viz
    cloud_calc
    water_ts
    ts_viz
    water_gif
    water_freq
    export_freq

Last modified: June 2021
Author: KaHeiChow

"""

# Import required packages

#%matplotlib inline
#%%output holomap='gif'

import datacube
import xarray as xr
import hvplot.xarray
import pandas as pd
import geopandas as gpd
from datetime import datetime
import warnings; warnings.simplefilter('ignore')
import imp
from time import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.dates as mdates
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta
from odc.ui import with_ui_cbk
import seaborn as sns
from dask.distributed import Client, LocalCluster
import sys
import cartopy.crs as ccrs
import warnings
import holoviews as hv
#hv.extension('bokeh')
hv.extension('matplotlib')
import cartopy.crs as ccrs
import os
import folium
import cv2
import math
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
import bokeh
from bokeh.plotting import show
from bokeh.io import output_notebook
import panel as pn
import hvplot.pandas
pn.extension()
import rioxarray
output_notebook
    
dc = datacube.Datacube(app = '1_MNDWI', config = '/home/datacube/.datacube.conf')
#client = Client(n_workers=2, threads_per_worker=3, memory_limit= '0.3GB')

# Define custom functions

def get_bbox(shp):
    """
    Getting bounding box coordinates from shapefile.
    Description
    ----------
    Use custom shapefile to acquire coordinates.
    Parameters
    ----------
    shp: shapefile (.shp)
        Path to shapefile including file format in a string.
    Returns
    -------
    dataset: xr.Dataset
        A list of the bounding box calculated from the input shapefile.
    """
    if os.path.splitext(shp)[1] != ".shp":
        raise ValueError('Input data should be a shapefile.')
    try:
        aoi = gpd.read_file(shp)
    except Exception:
        print('Cannot open shapefile.')
        
    aoi = gpd.read_file(shp)
    aoi_wgs = aoi.to_crs("EPSG:4326")
    min_lon = aoi_wgs.bounds.minx[0]
    max_lon = aoi_wgs.bounds.maxx[0]
    min_lat = aoi_wgs.bounds.miny[0]
    max_lat = aoi_wgs.bounds.maxy[0]
    string = [min_lon,max_lon,min_lat,max_lat]
    
    return string



def load_shp(shp, start, end, band=None, by_day=True):
    """
    Loading Sentinel-2 data in Bukina Faso using shapefile instead of coordinates.
    Description
    ----------
    Use custom shapefile to acquire open data cube data.
    Parameters
    ----------
    shp: shapefile (.shp)
        Path to shapefile including file format in a string.
    start: date_string
        A date_string given in the format YYYY-MM-DD. This will be use as the beginnning date for data acquisation.
    end: date_string
        A date_string given in the format YYYY-MM-DD. This will be use as the ending date for data acquisation.
    band: list
        A list of strings indicating the spectral bands to be included in the returned xr.Dataset. If omitted, 
        all available bands will be included in the returned xr.Dataset.
    by_day: boolean
        If True, returned dataset will be grouped by solar day. If False, returned dataset will has original 
        time steps of the available scene.
    Returns
    -------
    dataset: xr.Dataset
        A xr.Dataset with only pixels within the bounding box of the input shapefile.
    """
    if os.path.splitext(shp)[1] != ".shp":
        raise ValueError('Input data should be a shapefile.')
        
    try:
        datetime.strptime(start, '%Y-%m-%d')  
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD.")
        
    try: 
        aoi = gpd.read_file(shp)
    except Exception:
        print("Cannot open shapefile.")
        
    aoi = gpd.read_file(shp)
    aoi_wgs = aoi.to_crs("EPSG:4326")
    min_lon = aoi_wgs.bounds.minx[0]
    max_lon = aoi_wgs.bounds.maxx[0]
    min_lat = aoi_wgs.bounds.miny[0]
    max_lat = aoi_wgs.bounds.maxy[0]
    
    product = "s2_l2a_burkinafaso"
    
    if band == None:
        if by_day == True:
            data = dc.load(product= product,
                x= (min_lon, max_lon),
                y= (min_lat, max_lat),
                time= (start, end),
                group_by = "solar_day",
                progress_cbk=with_ui_cbk())
        else:
            data = dc.load(product= product,
                x= (min_lon, max_lon),
                y= (min_lat, max_lat),
                time= (start, end),
                progress_cbk=with_ui_cbk())
    else:
        if by_day == True:
            data = dc.load(product= product,
                measurements = band,
                x= (min_lon, max_lon),
                y= (min_lat, max_lat),
                time= (start, end),
                group_by = "solar_day",
                progress_cbk=with_ui_cbk())
        else:
            data = dc.load(product= product,
                measurements = band,
                x= (min_lon, max_lon),
                y= (min_lat, max_lat),
                time= (start, end),
                progress_cbk=with_ui_cbk())
    return data



def viz2d(ds,
          r_band='red',
          g_band='green',
          b_band='blue'):
    """
    Visualize the first time step of the dataset.
    Description
    ----------
    Visualize the first time step using specific spectral bands available in the dataset.
    Parameters
    ----------
    dataset: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.
    r: string
        Name of the data variable in string. This will be input in the red channel.
    g: string
        Name of the data variable in string. This will be input in the green channel.
    b: string
        Name of the data variable in string. This will be input in the blue channel.
    Returns
    -------
    mesh: matplotlib.collections.QuadMesh
        A two dimensional plot in RGB color, either true or false color composites.
    """
    try: 
        da_rgb = ds.isel(time=0).to_array().rename({"variable": "band"}).sel(band=[r_band,g_band,b_band])
    except NameError as error:
        print('Dataset is not defined.')
    except AttributeError as error:
        print('Input data need to be a xarray dataset.')
    except KeyError:
        print('The band(s) cannot be found.')
        
    #set projection to pre-defined CRS; CRS can be checked using `aoi.crs`
    ax = plt.subplot(projection=ccrs.UTM('33S'))
    
    plot = da_rgb.plot.imshow(
        ax=ax, 
        rgb='band', 
        transform=ccrs.UTM('33S'), 
        robust=True
    )
    return plot


## function for to_map()
def processing(df,stack,r_channel,g_channel,b_channel):
    """
    Transform xarray.Dataset to rearranged normalized numpy array for mapping.
    Description
    ----------
    Use custom shapefile to acquire coordinates.
    Parameters
    ----------
    df: xarray.Dataset
        dataset processed in the function to_map().
    stack: 3D numpy array
        array processed in the function to_map().
    r: int
        the number of layer to be mapped to red.
    g: int
        the number of layer to be mapped to green.
    b: int
        the number of layer to be mapped to blue.
    Returns
    -------
    dataset: 3D numpy array
        Processed and normalized 3D numpy array fits for further processing in to_map().
    """
    #stack layers
    stack_new = np.dstack([stack[:,:,r_channel],stack[:,:,g_channel],stack[:,:,b_channel]])

    #apply threshold
    yen_thres = threshold_yen(stack_new)
    img_bright = rescale_intensity(stack_new,(0,yen_thres),(0,255)).astype(int)

    #normalize array
    norm = np.zeros((len(df.latitude), len(df.longitude)))
    img = cv2.normalize(img_bright, norm, 0, 255, cv2.NORM_MINMAX)
    return img

def to_map(df,output='all',downscale=5,basemap='hybrid',zoom=9):
    """
    Display RGB on an interactive map.
    Description
    ----------
    Display xarray.Dataset as RGB in a folium map with multiple false color composites and basemap options.
    Parameters
    ----------
    df: xarray.Dataset
        dataset with multiple time steps, including bands "red","green","blue","nir", and "swir1".
    output: string ("all","veg","agri","rgb","water")
        options for displaying false color composite. Values can be either "all" for all combinations, "veg" for 
        vegetation (NIR, Red, Green), "argi" for agriculture (SWIR1, NIR, Blue), "rgb" for true color composite (Red, 
        Green, Blue), and "water" for water (NIR, SWIR1, Red). Defult = "all".
    downscale: float
        a floating number > 1 for downscale image resolution. Default = 5.
    basemap: string ("all","google","terrain","hybrid","esri")
        the type of basemap to be included in the folium map. Values can be either "google" for Google map, "terrain" for 
        Google Terrain, "hybrid" for Google Satellite Hybrid, "esri" for ESRI Satellite basemap, and "all" for all 
        basemap.
    Returns
    -------
    map: Folium Map
        Folium map with scene displayed as RGB images with layer control.
    """

    #error catching
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    try:
        df.red
        df.green
        df.blue
        df.nir
        df.swir1
    except Exception:
        print("RGB/NIR/SWIR1 bands not found.")
    
    r_band = df.red.isel(time=0).values
    g_band = df.green.isel(time=0).values
    b_band = df.blue.isel(time=0).values
    nir = df.nir.isel(time=0).values
    swir1 = df.swir1.isel(time=0).values
    stack = np.dstack((r_band,g_band,b_band,nir,swir1))
    
    #create RGB 3D array
    rgb = processing(df,stack,0,1,2)
    veg = processing(df,stack,3,0,1)
    agri = processing(df,stack,4,3,2)
    water = processing(df,stack,3,4,0)
      
    #boundary of the image on the map
    min_lon = df.longitude.min().values.astype(np.float) + 0.0
    max_lon = df.longitude.max().values.astype(np.float) + 0.0
    min_lat = df.latitude.min().values.astype(np.float) + 0.0
    max_lat = df.latitude.max().values.astype(np.float) + 0.0
    
    #create basemap for folium
    basemaps = {
        'Google Maps': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Basemap: Google Maps',
            overlay = True,
            control = True
        ),
        'Google Satellite': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Basemap: Google Satellite',
            overlay = True,
            control = True
        ),
        'Google Terrain': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Basemap: Google Terrain',
            overlay = True,
            control = True
        ),
        'Google Satellite Hybrid': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Basemap: Google Satellite Hybrid',
            overlay = True,
            control = True
        ),
        'Esri Satellite': folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Basemap: Esri Satellite',
            overlay = True,
            control = True
        )
    }
    
    #display layers on map
    map_ = folium.Map(location=[(min_lat+max_lat)/2, (min_lon+max_lon)/2], zoom_start = zoom)
    if basemap == 'all':
        basemaps['Esri Satellite'].add_to(map_)
        basemaps['Google Maps'].add_to(map_)
        basemaps['Google Terrain'].add_to(map_)
        basemaps['Google Satellite Hybrid'].add_to(map_)
    elif basemap == 'hybrid':
        basemaps['Google Satellite Hybrid'].add_to(map_)
    elif basemap == 'terrain':
        basemaps['Google Terrain'].add_to(map_)
    elif basemap == 'google':
        basemaps['Google Maps'].add_to(map_)
    elif basemap == 'esri':
        basemaps['Esri Satellite'].add_to(map_)
    else:
        print("Invalid value for basemap argument: Please input 'esri','google','terrain', or 'hybrid'.")
    
    if downscale == "auto":
        factor = len(df.latitude)*len(df.longitude)/2000000
        rgb = cv2.resize(rgb.astype('float32'), 
                         dsize=(math.ceil(len(df.latitude)/factor), 
                                math.ceil(len(df.longitude)/factor)), 
                         interpolation=cv2.INTER_CUBIC)
        veg = cv2.resize(veg.astype('float32'), 
                         dsize=(math.ceil(len(df.latitude)/factor), 
                                math.ceil(len(df.longitude)/factor)), 
                         interpolation=cv2.INTER_CUBIC)
        agri = cv2.resize(agri.astype('float32'), 
                          dsize=(math.ceil(len(df.latitude)/factor), 
                                 math.ceil(len(df.longitude)/factor)), 
                          interpolation=cv2.INTER_CUBIC)
        water = cv2.resize(water.astype('float32'), 
                           dsize=(math.ceil(len(df.latitude)/factor), 
                                  math.ceil(len(df.longitude)/factor)), 
                           interpolation=cv2.INTER_CUBIC)
        
    elif downscale != 0:
        assert downscale > 0
        rgb = cv2.resize(
            rgb.astype('float32'), dsize=(math.ceil(len(df.latitude)/downscale),        
                                          math.ceil(len(df.longitude)/downscale)),
            interpolation=cv2.INTER_CUBIC)
        veg = cv2.resize(veg.astype('float32'), 
                         dsize=(math.ceil(len(df.latitude)/downscale), 
                                math.ceil(len(df.longitude)/downscale)), 
                         interpolation=cv2.INTER_CUBIC)
        agri = cv2.resize(agri.astype('float32'), 
                          dsize=(math.ceil(len(df.latitude)/downscale),
                                 math.ceil(len(df.longitude)/downscale)), 
                          interpolation=cv2.INTER_CUBIC)
        water = cv2.resize(water.astype('float32'), 
                           dsize=(math.ceil(len(df.latitude)/downscale), 
                                  math.ceil(len(df.longitude)/downscale)), 
                           interpolation=cv2.INTER_CUBIC)
        
    else:
        print("Please input downscale argument larger than 0.")
        
    if output == "all":
        try:
            folium.raster_layers.ImageOverlay(
                rgb,[[min_lat, min_lon], [max_lat, max_lon]], name='RGB'
            ).add_to(map_)
            folium.raster_layers.ImageOverlay(
                veg,[[min_lat, min_lon], [max_lat, max_lon]], name='Vegetation'
            ).add_to(map_)
            folium.raster_layers.ImageOverlay(
                agri,[[min_lat, min_lon], [max_lat, max_lon]], name='Agriculture'
            ).add_to(map_)
            folium.raster_layers.ImageOverlay(
                water,[[min_lat, min_lon], [max_lat, max_lon]], name='Water'
            ).add_to(map_)
            
        except Exception:
            print("Unexpected Error for image overlay.")
            
    elif output == "rgb":
        folium.raster_layers.ImageOverlay(
            rgb,[[min_lat, min_lon], [max_lat, max_lon]], name='RGB'
        ).add_to(map_)
    elif output == "veg":
        folium.raster_layers.ImageOverlay(
            veg,[[min_lat, min_lon], [max_lat, max_lon]], name='Vegetation'
        ).add_to(map_)
    elif output == "agri":
        folium.raster_layers.ImageOverlay(
            agri,[[min_lat, min_lon], [max_lat, max_lon]], name='Agriculture'
        ).add_to(map_)
    elif output == "water":
        folium.raster_layers.ImageOverlay(
            water,[[min_lat, min_lon], [max_lat, max_lon]], name='Water'
        ).add_to(map_)
    else: 
        print("The input output argument is invalid ({}). \
        Please use 'all', 'rgb', 'veg', 'agri', or 'water'.".format(output))
  
    folium.LayerControl().add_to(map_)
    return map_



def sceneDisplay(df):
    """
    Display RGB time series.
    Description
    ----------
    Display interactive RGB image with a slidebar to control the time step to be displayed.
    Parameters
    ----------
    df: xarray.Dataset
        dataset with multiple time steps, including bands "red","green", and "blue".
    Returns
    -------
    map: hvplot
        hvplot displayed in RGB with a slidebar to control time steps.
    """
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    try:
        df[['red','green','blue']]
    except Excpetions:
        print("RGB bands cannot be found.")
    
    try:
        df[['longitude','latitude']]
    except Excpetions:
        print("'longitude' or/and 'latitude' cannot be found.")
        
    df_rgb = df.to_array().rename({"variable": "band"}).sel(band=['red','green','blue'])
    normalized = df_rgb/(df_rgb.quantile(.99,skipna=True)/255)
    mask = normalized.where(normalized < 255)
    int_arr = mask.astype(int)
    plot = int_arr.astype('uint8').hvplot.rgb(
        x='longitude', y='latitude', bands='band', data_aspect=1, 
        flip_yaxis=True, xaxis=False, yaxis=None, groupby='time', 
        widgets={'time': pn.widgets.DiscreteSlider}
    )
    
    return plot



def dataMask(df, cloudMask = False):
    """
    Masking pixels in the xarray dataset with poor quality.
    Description
    ----------
    Masking pixels not in the range of 100 and 10000. Cloud mask option masks pixels possibly covered by cloud. 
    Parameters
    ----------
    dataset: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.
    cloudMask: boolean
        If True, pixels in the xr.dataset possibly covered by cloud is given the value NaN.
    Returns
    -------
    masked_dataset: xr.Dataset
        A xr.Dataset like the input dataset with only pixels of good quality.
        Every other pixel is given the value NaN.
    """
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    if (df.red == np.nan).any():
        warnings.warn("red band is missing.")
    elif (df.blue == np.nan).any():
        warnings.warn("blue band is missing.")
    elif (df.green == np.nan).any():
        warnings.warn("green band is missing.")
    elif (df.swir1 == np.nan).any():
        warnings.warn("swir1 band is missing.")
    elif (df.nir == np.nan).any():
        warnings.warn("nir band is missing.")
    else:
        pass
    
    # Filter pixels with poor quality
    # (Sentinel-2 pixel values represent Top of Atmosphere (TOA) reflectance units x 10,000)
    if (df.red != np.nan).any().values == True:
        df_new = df.where(df.red > 100)
        df_new = df_new.where(df.red < 10000)
    if (df.blue != np.nan).any().values == True:
        df_new = df_new.where(df.blue > 100)
        df_new = df_new.where(df.blue < 10000)
    if (df.green != np.nan).any().values == True:
        df_new = df_new.where(df.green > 100)
        df_new = df_new.where(df.green < 10000)
    if (df.swir1 != np.nan).any().values == True:
        df_new = df_new.where(df.swir1 > 100)
        df_new = df_new.where(df.swir1 < 10000)
    if (df.nir != np.nan).any().values == True:
        df_new = df_new.where(df.nir > 100)
        df_new = df_new.where(df.nir < 10000)
    
    # Cloud Masking
    if cloudMask == True:
        if (df.scl == np.nan).any():
            raise ValueError('Input dataset should include the scl band for cloud masking.')
        df_new = df_new.where(df.scl != 9)
    
    return df_new



def getQual(df):
    """
    Calculate the percentage of good pixels in every time stamp of the input dataset.
    Description
    ----------
    Calculate the ratio of pixels of good quality in the scene and return a list.
    Parameters
    ----------
    dataset: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables. Input dataset can also be 
         a masked dataset.
    Returns
    -------
    dataQuality: list
        A list of values between 0 and 100 (in %) indicating percentage of good pixels in each time stamp.
    """
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    if (df.scl == np.nan).any():
        raise ValueError('dataset should include the scl band for cloud information.')
        
    ls = []
    for ts in np.arange(len(df.coords["time"])):
        # Calculate Total Pixels
        try:
            ttl_pixel = df['longitude'].count().values.tolist()*df['latitude'].count().values.tolist()
        except KeyError:
            print('Cannot find "longitude" and "latitude".')

        # Extract Good Pixels
        try:
            df_masked = df.isel(time=ts).where(
                np.logical_and(df.isel(time=ts) != np.nan, df.isel(time=ts).scl != 9)
            )
        except Exception:
            print('Unexpected Error: Cannot mask pixels for the time stamps.')

        try: 
            Qp = df_masked['scl'].count().values.tolist()
        except Exception:
            print('Unexpected Error.')

        
        # Calculate Ratio
        r = round((Qp/ttl_pixel)*100,2)
        ls.append(r)

    return ls



def vizQual(df, thres = 60, type="bar"):
    """
    Visually check the quality of loaded dataset.
    Description
    ----------
    Plotting a bar chart for checking data quality of all time stamps in the dataset.
    Parameters
    ----------
    dataset: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.
    thres: integer
        A threshold of percentage below which data will be marked as poor for each time stamp.
    type: "bar" or "line"
        If "bar", a bar chart will be returned with time stamps below threshold shown in red. If "line", a line chart 
        will be returned with the threshold marked as a horizontal line.
    Returns
    -------
    chart: matplotlib.pyplot
        A bar chart or line chart showing the percentage of good data in every time stamps.
    """
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    ls = getQual(df)
    
    try:
        arr = np.array(ls)
    except Exception:
        print('Cannot convert data quality to numpy array.')
        
    try:
        poor_mask = arr < thres
        good_mask = arr >= thres
    except Exception:
        print('Unrecognized "marked level".')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    if type == "bar":
        try:
            ax.bar(np.arange(len(ls))[poor_mask],arr[poor_mask], color = 'red', figure = fig)
            ax.bar(np.arange(len(ls))[good_mask],arr[good_mask], color = 'blue', figure = fig)
            ax.grid(figure = fig)
            plt.legend(["Under threshold", "Data Quality"])
        except Expection:
            print('Unexpected Error: the figure cannot be plotted.')
        
    elif type == "line":
        try: 
            plt.plot(np.arange(len(ls))[good_mask],arr[good_mask], color = 'blue', figure = fig)
            plt.axhline(y=thres, color='r', linestyle='-')
            ax.grid(figure = fig)
            plt.legend(["Threshold", "Data Quality"])
        except Expection:
            print('Unexpected Error: the figure cannot be plotted.')
    
    else:
        raise ValueError('Type can only be "bar" for bar chart and "line" for linechart.')
        

    plt.title("Qualitifed Pixels in the Scene", fontsize = 14.5)
    plt.ylabel("Good Pixel (%)", fontsize = 12.5)
    plt.xlabel("Timestamps", fontsize = 12.5)
    plt.xticks(np.arange(min(len(ls), max(len(ls)+1, 1.0))))
    plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1.0)
    if len(df.time) > 30:
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
    plt.close()
    
    return fig



def pred_index(df, resample = '1M', method = "mean", drop_bands = False):
    """
    Preparation of indices in the xarray dataset for water analysis.
    Description
    ----------
    Adding NDVI and MNDWI in the input dataset after resampling monthly average.
    Parameters
    ----------
    df: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.
    resample: rule; DateOffset, Timedelta or str
        The offset string or object representing target conversion. If not given, data will be resampled to 
        monthly average. If resample set to "None", data will not be resampled.
    drop_bands: Boolean
        If True, spectral bands used to calculate the indices (i.e. "red","green","blue","nir","swir1") will 
        be dropped in the returned dataset.
    Returns
    -------
    masked_dataset: xr.Dataset
        A xr.Dataset like the input dataset with resampled data with NDVI and MNDWI as new data variables.
    """
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    if (df.nir == np.nan).any():
        warnings.warn("nir band is missing")
        
    elif (df.swir1 == np.nan).any():
        warnings.warn("swir1 band is missing")
        
    elif (df.red == np.nan).any():
        warnings.warn("red band is missing")
        
    elif (df.blue == np.nan).any():
        warnings.warn("blue band is missing")
        
    elif (df.green == np.nan).any():
        warnings.warn("green band is missing")
        
    else:
        pass
    
    # Resampling
    try:
        if resample != "None" and method == "mean":
            data_resampled = df.resample(time=resample, skipna=True).mean()
        elif resample != "None" and method == "median":
            data_resampled = df.resample(time=resample, skipna=True).median()
        elif resample != "None" and method == "min":
            data_resampled = df.resample(time=resample, skipna=True).min()
        elif resample != "None" and method == "max":
            data_resampled = df.resample(time=resample, skipna=True).max()
        elif resample != "None" and method == "mode":
            data_resampled = df.resample(time=resample, skipna=True).mode()
        elif resample != "None" and method == "std":
            data_resampled = df.resample(time=resample, skipna=True).std()
        elif resample != "None" and method == "var":
            data_resampled = df.resample(time=resample, skipna=True).var()
        elif resample != "None" and method == "sum":
            data_resampled = df.resample(time=resample, skipna=True).sum()
        else:
            data_resampled = df
    except Exception:
        print('Unexpected Error: Dataset cannot be resampled.')
    
    # MNDWI
    try:
        data_resampled = data_resampled.assign(
            MNDWI = (data_resampled["green"] - data_resampled["swir1"])/
            (data_resampled["green"] + data_resampled["swir1"])
        )
    except Exception:
        print('Error occurred for MNDWI calculation.')
        
    # NDVI
    try:
        data_resampled = data_resampled.assign(
            NDVI = (data_resampled["nir"] - data_resampled["red"])/
            (data_resampled["nir"] + data_resampled["red"])
        )
    except Exception:
        print('Error occurred for NDVI calculation.')
        
    # Water Detection
    try: 
        data_resampled = data_resampled.assign(
            water = xr.where((data_resampled["NDVI"] < 0) & (data_resampled["MNDWI"] > 0), 1.0, 0.0)
        )

        data_resampled = data_resampled.assign(
            water_null = xr.where((data_resampled["NDVI"] < 0) & (data_resampled["MNDWI"] > 0), 1.0, None)
        )
    except Exception:
        print('Unexpected Error: Cannot apply threshold to the calculated indices.')
    
    if drop_bands == True:
        data_resampled = data_resampled.drop_vars(["red","green","blue","nir","swir1"])
        
        if (df.scl != -999).any():
            data_resampled = data_resampled.drop_vars("scl")
        
    return data_resampled



def water_viz(df, col = 4):
    """
    Visualize identified water area extracted from the dataset.
    Description
    ----------
    Water area is identified using thresholding of the indices (NDVI and MNDWI) and visualized for every time 
    stamp in the input dataset.
    Parameters
    ----------
    dataset: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.
    col: int
        An integer indicates the number of column in the plotting layout (default 4).
    Returns
    -------
    Figure: matplotlib.pyplot
        2D subplots indicating detected water area.
    """
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    try:
        df.water
    except Exception:
        raise ValueError("No water band.")
    
    try:
        df.latitude
    except Exception:
        raise ValueError("No latitude found.")
        
    try:
        df.longitude
    except Exception:
        raise ValueError("No longitude found.")
        
    if len(df.time) > 12:
        warnings.warn("too many timestamps (>12)")
        
    if len(df.time) == 1:
        fig = df.water.plot(x="longitude", y="latitude", cmap=plt.cm.Blues)
        
    else:
        try:
            fig = df.water.plot(x="longitude", y="latitude", col="time", col_wrap=col, cmap=plt.cm.Blues)
        except Exception:
            print('Unexpected Error: Cannot plot water band.')
        
    return fig



def cloud_calc(df): 
    """
    Quantify cloud area.
    Description
    ----------
    Calculate area covered by cloud in the scene for every time steps.
    Parameters
    ----------
    dataset: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.
    Returns
    -------
    cloud_area: list
        A list with a length of time steps in the input dataset. 
        The number indicates the area covered by cloud in square kilometers.
    """    
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    ls = []
    try: 
        ntime = len(df.coords["time"])
    except Exception:
        print('Unexpected Error: Cannot count time stamps.')
    for ts in np.arange(ntime):
        df_new = df.isel(time=[ts])
        try:
            df_masked = df_new.where(np.logical_or(df_new.scl != 8, np.logical_and(df_new.scl == 8, df_new.water == 1)))
        except Exception:
            print("Cannot mask 'scl' and 'water' band.")
        
        nx = df['longitude'].count().values.tolist()
        
        # Uncertainty calculated in km square
        uncertainty = sum((nx - df_masked['scl'].count(axis=1)).values[0])* 100/1000000
        
        ls.append(uncertainty)     
    return ls



def water_ts(df):
    """
    Calculated water area.
    Description
    ----------
    Create a new pandas dataframe for detected water area and cloud covered area.
    Parameters
    ----------
    df: xarray.Dataset
        dataset with multiple time steps.
    Returns
    -------
    map: pandas.DataFrame
        pandas.DataFrame with two columns: 'water_area_km2' and 'cloud_uncertainty_km2' indexed by time steps. 
        The areas are in square kilometers.
    """
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    #calculate cloud uncertainty
    uncertainty = cloud_calc(df)
    
    #save the time stamps to pandas series
    try:
        ts = df.time.to_series()
    except Expection:
        print('Time stamps not found.')
     
    try:
        water_area = df.water_null.groupby("time").count({"latitude","longitude"}).compute().values * 100 / 1000000

        month = pd.Series(ts).values #get the series values
        water = pd.Series(water_area).values

        frame = { 'date': month, 'water_area_km2': water, 'cloud_uncertainty_km2': uncertainty } #set up a data frame
        df_new = pd.DataFrame(frame) 
        df_new.index = pd.to_datetime(df_new["date"],format='%Y%m%d') #set up the date time index
        df_new = df_new.drop(columns=["date"]) #drop extra column
    except Exception:
        print('Unexpected Error: Cannot create new dataframe.')
    
    return df_new


def ts_viz(df, thres = 0.8, 
           uncertainty = False, interpolate = False, interact = False,
           title = "Surface Waterbodies Timeseries"): 
    """
    Water time series visualization.
    Description
    ----------
    Area plot of detected water area, including cloud uncertainty, in the time series.
    Parameters
    ----------
    dataset: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.
    thres: float
        A float between 0 (0%) and 1 (100%). It indicates the percentage of water area compared to the maximum level in 
        the time sreries. It will be used as the threshold used for 
        highlighting time steps with scarce water reosource. If thres = None, no marking will be shown.
    Returns
    -------
    figure: pandas.DataFrame.plot.area
        An area plot showing detected water area across all time stamps in square kilometers.
    """
    
    if thres != None:
        assert thres >= 0 and thres <= 1,"thres has to be between 0 and 1."

    df_new = water_ts(df)
    
    if interact == True:
        if interpolate == True:
            df_new = df_new.drop(columns=["cloud_uncertainty_km2"])          
            df_interpolate = df_new.resample('D').asfreq().interpolate(method='polynomial', order=2)
            df_interpolate.loc[df_interpolate['water_area_km2'] < 0, 'water_area_km2'] = 0
            fig = df_interpolate.water_area_km2.hvplot().opts(title=title,xlabel="Date",ylabel="Area (km²)")
        elif interpolate == False:
            fig = df_new.water_area_km2.hvplot().opts(title=title,xlabel="Date",ylabel="Area (km²)")
        else:
            raise ValueError('interpolate should be a boolean.')
        return fig
    
    if thres != None:
        assert thres >= 0 and thres <= 1,"thres has to be between 0 and 1."

    df_new = water_ts(df)
    
    #define critical point for water area
    if thres != None:
        low = df_new[df_new['water_area_km2'] < df_new['water_area_km2'].max()*thres].index
        label = r'Highlight: Area $<{}$% of Max'.format(int(thres*100))
    
    if interpolate == False:
        try:
            fig, ax = plt.subplots() #define name of the plot and the axis

            if uncertainty == False:
                df_new = df_new.drop(columns=["cloud_uncertainty_km2"])
            else: 
                pass
            df_new.plot.area(
                figsize=(16, 8), 
                ylim=(df_new.water_area_km2.min()*0.8, 
                      df_new.water_area_km2.max()*1.2), 
                title = title, x_compat=True, ax=ax)
            
            ax.plot([], [], ' ') #empty plot for the text

            if thres != None:
                for i in low:
                    ax.axvspan(i-DateOffset(months=1)+DateOffset(days=1), i-DateOffset(days=1), 
                               color='red', alpha=0.3) #highlight the critical months

            fig.autofmt_xdate()
        except Exception as err:
            print("Cannot plot figure.",err)
    elif interpolate == True:
        try:
            fig, ax = plt.subplots() #define name of the plot and the axis
            df_new = df_new.drop(columns=["cloud_uncertainty_km2"])
            
            df_interpolate = df_new.resample('D').asfreq().interpolate(method='polynomial', order=2)
            df_interpolate.loc[df_interpolate['water_area_km2'] < 0, 'water_area_km2'] = 0

            df_interpolate.plot.area(
                figsize=(16, 8), 
                ylim=(df_new.water_area_km2.min()*0.8, 
                      df_new.water_area_km2.max()*1.2), 
                title = title, x_compat=True, ax=ax)
            
            ax.plot([], [], ' ') #empty plot for the text

            if thres != None:
                for i in low:
                    ax.axvspan(
                        i-DateOffset(months=1)+DateOffset(days=1), i-DateOffset(days=1), 
                        color='red', alpha=0.3) #highlight the critical months

            fig.autofmt_xdate()
        except Exception as err:
            print("Cannot plot figure.",err)
    else:
        raise ValueError('interpolate should be a boolean.')
        
    
    ax.set_xlabel("Date", fontsize = 16) #give x labal
    ax.set_ylabel("Detected Water Area (km²)", fontsize = 16) # give y label
    if uncertainty == True:
        ax.legend(["Area","Cloud Uncertainty",label], fontsize = 14) #set legend
    elif thres != None:
        ax.legend(["Area",label], fontsize = 14)
    else:    
        ax.legend(["Area"], fontsize = 14)
    ax.title.set_size(18)
    plt.close()
    
    return fig



def water_gif(ds, 
              fps = 2, 
              animate = True): 
    """
    Water area animation.
    Description
    ----------
    Visualize the temporal dynamics of detected water area in animation.
    Parameters
    ----------
    ds: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.
    fps: int
        Frame per second (default = 2).
    animate: boolean
        If True, output will be displayed in gif format. If False, a control slider will be display for user to manually 
        visualize changes in different time steps. 
    Returns
    -------
    masked_dataset: xr.Dataset
        A xr.Dataset like the input dataset with only pixels which are within the polygons of the geopandas.Geodataframe.
        Every other pixel is given the value NaN.
    """
    try: 
        ds[['longitude', 'latitude']]
    except Exception:
        print("'latitude' and 'longitude' cannot be found.")
        
    try:
        ds_new = hv.Dataset(ds.water) #set up dataset for animation
        images = ds_new.to(
            hv.Image, ['longitude', 'latitude']).options(fig_inches=(6.5, 5), 
                                                         colorbar=True,
                                                         cmap=plt.cm.Blues)
    except Exception:
        print("Unexpected Error.")
        
    if animate == True:
        hv.output(images, holomap='gif', fps=fps) #animation output inline
    else:
        return images



def water_freq(df):
    """
    Map frequency of water occurence.
    Description
    ----------
    Map how frequent is water detected from the time series in the same pixel.
    Parameters
    ----------
    df: xr.Dataset
         A multi-dimensional array with x,y and time dimensions includeing data variable "water" calculated from 
         pred_index().
    Returns
    -------
    masked_dataset: hvplot
        An interactive hvplot with values between 0 and 1 representing the ratio of time steps with water detected.
    """
    
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    try:
        df.water
    except Exception:
        print("'water' band cannot be found. Please use pred_index() to acquire the required band.")
    try:
        df.time
    except Exception:
        print("'time' cannot be found. Please check the time dimension of the dataset.")
    try:
        df.longitude
        df.latitude
    except Exception:
        print("'longitude' or/and 'latitude' cannot be found. Please check the dimension of the dataset.")
        
    frequency = df.water.sum(dim='time',skipna=True)/len(df.time)
    show(hv.render(frequency.hvplot.image(x="longitude",y="latitude",aspect=1,cmap='bmy_r')))



def export_freq(df,path):
    """
    Export water occurence frequency.
    Description
    ----------
    Export the results for water occurence frequency in geotiff format.
    Parameters
    ----------
    df: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.
    path: path
        A path for downloading geotiff file, including file name and data format (.tif). 
    Returns
    -------
    geotiff: geotiff
        A raster file (.tif) displaying results from water_freq(). Values range from 0 to 1, representing the time 
        proportion with water present.
    """
    
    assert isinstance(df, xr.Dataset),"Input has to be a xarray.Dataset."
    
    try:
        df.water
    except Exception:
        print("'water' band cannot be found. Please use pred_index() to acquire the required band.")
    try:
        df.time
    except Exception:
        print("'time' cannot be found. Please check the time dimension of the dataset.")
        
    frequency = df.water.sum(dim='time',skipna=True)/len(df.time)
    
    assert path[-4:] == ".tif", "'path' argument should end with .tif"
        
    frequency.rio.to_raster(path)