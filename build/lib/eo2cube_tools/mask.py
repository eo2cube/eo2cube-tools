import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation

scl_cat = {'1': 'Saturated or defective pixel',
    '2': 'Dark features / Shadows',
    '3': 'Cloud shadows',
    '4': 'Vegetation',
    '5': 'Not vegetated',
    '6': 'Water',
    '7': 'Unclassified',
    '8': 'Cloud medium probability',
    '9': 'cloud high probability',
    '10': 'Thin cirrus',
    '11': 'Snow or ice'}

def dilate(array, dilation=10, invert=True):
    y, x = np.ogrid[
        -dilation: (dilation + 1),
        -dilation: (dilation + 1),
    ]

    # disk-like radial dilation
    kernel = (x * x) + (y * y) <= (dilation + 0.5) ** 2

    # If invert=True, invert True values to False etc
    if invert:
        array = ~array

    return ~binary_dilation(array.astype(np.bool),
                            structure=kernel.reshape((1,) + kernel.shape))

def scl_mask(dataset, categories = ['Dark features / Shadows','Vegetation', 'Not vegetated', 'Water', 'Unclassified', 'Snow or ice'], dilation = None):
    """
    Takes an xarray dataset and creates a mask based on categories defined in the SCL band

    Parameters
    ----------
    ds : xarray Dataset
       A two-dimensional or multi-dimensional array including the SCL band
    categories : list
       A list of Sentinel-2 Scene Classification Layer (SCL) names. The default is
       ['Dark features / Shadows','Vegetation', 'Not vegetated', 'Water',
       'Unclassified', 'Snow or ice'] which will return non-cloudy or
       non-shadowed land, snow, water, veg, and non-veg pixels.
    dilation : int, optional
        An optional integer specifying the number of pixels to dilate
        by. Defaults to 10, which will dilate `array` by 10 pixels.

    Returns
    -------
    An xarray dataset containing a mask for each time step
    """

    assert "scl" in list(dataset.data_vars.keys()), "scl band is missing"
    if dilation != None:
        return xr.apply_ufunc(dilate, dataset["scl"].isin([int(k) for k,v in scl_cat.items() if v in categories]), dilation, keep_attrs=True)
    else:
        return dataset["scl"].isin([int(k) for k,v in scl_cat.items() if v in categories])
