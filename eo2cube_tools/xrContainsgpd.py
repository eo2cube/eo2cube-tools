#import xarray
import numpy as np
from shapely.geometry import box
import geopandas as gpd
from spatialextent import spatialextent


def xrContainsgpd(dataset, aoi, latlonFormat=True):

    """

    Description
    ----------
    Indicates if a defined GeoDataFrame is located within the area of a certain xarray dataset

    Parameters
    ----------
    dataset : xarray.Dataset
        A two-dimensional or multi-dimensional array
        
    aoi : geopandas.geodataframe.GeoDataFrame
        A GeoDataFrame consisting of one (or multiple?) elements
        
    latlonFormat: Boolean, (default = True)
        A boolean value indicating the coordinate format of the dataset
        False = x & y
        True = latitude & longitude

    Returns
    -------
    contain_bool : Boolean 
        True = aoi lies completely within dataset area
        False = aoi is not entirely located within dataset area

    """
    
    # create gpd polygon from xarray extent
    # extracting boundingbox
    xr_coords = spatialextent(dataset, latlonFormat)
    
    # creating shapely geometry using the boudingbox of the polygon
    b = box(float(xr_coords["lat_min"]),float(xr_coords["lon_min"]),float(xr_coords["lat_max"]),float(xr_coords["lon_max"]))
    
    # turning b into a list
    geoms = [b]
        
    # turning shapely geometry into geopandas geodataframe
    # https://stackoverflow.com/questions/56523003/how-to-create-polygon-using-4-points
    
    # definition of empty GeoDataFrame
    ds_gdf = gpd.GeoDataFrame()

    # assigining new geometry to GeoDataFrame
    ds_gdf["geometry"] = geoms
    
    # adding crs of dataset to GeoDataFrame
    # if crs does not exist
    if None == ds_gdf.crs:
        ds_gdf = ds_gdf.set_crs(dataset.crs)
    else:
        ds_gdf = ds_gdf.to_crs(dataset.crs)
      
    # reproject aoi to CRS of xarray
    aoi = aoi.to_crs(ds_gdf.crs)

    # testing if ds_gdf contains aoi_latlon
    contain_bool = ds_gdf.contains(aoi)

    return contain_bool