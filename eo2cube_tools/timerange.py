import xarray
import numpy as np


def timerange(dataset, inputFormat="ns"):

    """Time range extraction

    Description
    ----------
    Takes an xarray dataset and returns time information for first and last layer

    Parameters
    ----------
    dataset : xarray.Dataset
        A two-dimensional or multi-dimensional array
        
    inputFormat: String, (default = "ns")
        A String indicating the unit of the datetime64 objects within the xarray dataset

    Returns
    -------
    start_end : numpy array containing two datetime64 objects (start and end date)

    """

    start_date = np.datetime64(dataset.time.item(0), inputFormat)
    end_date = np.datetime64(dataset.time.item(-1), inputFormat)
    
    start_end = np.array([start_date, end_date])

    return start_end