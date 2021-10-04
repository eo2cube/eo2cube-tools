#import xarray
import numpy as np


def spatialextent(dataset, latlonFormat=True, lengthOutput=False):

    """Extraction of spatial extent

    Description
    ----------
    Takes an xarray dataset and returns its min and max coordinates

    Parameters
    ----------
    dataset : xarray.Dataset
        A two-dimensional or multi-dimensional array
        
    latlonFormat: Boolean, (default = True)
        A boolean value indicating the format of the coordinates.
        False = x & y
        True = latitude & longitude
        
    lengthOutput: Boolean, (default = False)
        A boolean value indicating whether edge coordinates (lengthOutput = False) 
        or side lengths (lengthOutput = True) should be calculated.

    Returns
    -------
    numpy array : array containing min and max coordinates of xarray (lat_min, lat_max, lon_min, lon_max)

    """

    # adjusting coordinate format
    if latlonFormat:
        xcoord = "longitude"
        ycoord = "latitude"
    else:
        xcoord = "x"
        ycoord = "y"
    
    # extracting lat and lon extremes from dataset
    lat_min = dataset.coords[xcoord].min().values
    lat_max = dataset.coords[xcoord].max().values
    lon_min = dataset.coords[ycoord].min().values
    lon_max = dataset.coords[ycoord].max().values

    # returning coordinates or length/width information
    if lengthOutput:
        return {
            "lat_length": lat_max-lat_min,
            "lon_length": lon_max-lon_min
        }
        return np.array([lat_max-lat_min, lon_max-lon_min])
    else:
        return {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max
        }