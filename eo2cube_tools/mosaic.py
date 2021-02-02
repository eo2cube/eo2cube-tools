"""
 Copyright 2016 United States Government as represented by the Administrator
 of the National Aeronautics and Space Administration. All Rights Reserved.

 Portion of this code is Copyright Geoscience Australia, Licensed under the
 Apache License, Version 2.0 (the "License"); you may not use this file
 except in compliance with the License. You may obtain a copy of the License
 at

    http://www.apache.org/licenses/LICENSE-2.0

 The CEOS 2 platform is licensed under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0.

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 License for the specific language governing permissions and limitations
 under the License.

###

Tested in the ODC environment of the Department of Remote Sensing, 
JMU Wuerzburg. Due to usage of a metric reference system the usage
of x/y coordinates instead of lat/lon degrees is used here.

contact: sebastian.foertsch@uni-wuerzburg.de
###

"""

import gdal, osr
import collections
import gc
import numpy as np
import xarray as xr
from datetime import datetime
import collections
from collections import OrderedDict
import hdmedians as hd

import datacube
from . import dc_utilities as utilities
from .dc_utilities import create_default_clean_mask
import hdmedians as hd

"""
Utility Functions
"""

def convert_to_dtype(data, dtype):
    """
    A utility function converting xarray, pandas, or NumPy data to a given dtype.

    Parameters
    ----------
    data: xarray.Dataset, xarray.DataArray, pandas.Series, pandas.DataFrame,
             or numpy.ndarray
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.
    """
    if dtype is None: # Don't convert the data type.
        return data
    return data.astype(dtype)


"""
Compositing Functions
"""

def create_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, intermediate_product=None, **kwargs):
    """
    Creates a most-recent-to-oldest mosaic of the input dataset.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, x, y
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: x, y
        variables: same as dataset_in
    """
    dataset_in = dataset_in.copy(deep=True)

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    # Mask data with clean_mask. All values where clean_mask==False are set to no_data.
    for key in list(dataset_in.data_vars):
        dataset_in[key].values[np.invert(clean_mask)] = no_data

    # Save dtypes because masking with Dataset.where() converts to float64.
    band_list = list(dataset_in.data_vars)
    dataset_in_dtypes = {}
    for band in band_list:
        dataset_in_dtypes[band] = dataset_in[band].dtype

    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None

    time_slices = reversed(
        range(len(dataset_in.time))) if 'reverse_time' in kwargs else range(len(dataset_in.time))
    for index in time_slices:
        dataset_slice = dataset_in.isel(time=index).drop('time')
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_in.data_vars):
                dataset_out[key].values[dataset_out[key].values == -9999] = dataset_slice[key].values[dataset_out[key]
                                                                                                      .values == -9999]
                dataset_out[key].attrs = OrderedDict()

    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out

def create_mean_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, **kwargs):
    """
    Method for calculating the mean pixel value for a given dataset.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, x, y
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: x, y
        variables: same as dataset_in
    """
    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    # Save dtypes because masking with Dataset.where() converts to float64.
    band_list = list(dataset_in.data_vars)
    dataset_in_dtypes = {}
    for band in band_list:
        dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and scan lines.
    dataset_in = dataset_in.where((dataset_in != no_data) & (clean_mask))
    dataset_out = dataset_in.mean(dim='time', skipna=True, keep_attrs=False)

    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out


def create_median_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, **kwargs):
    """
    Method for calculating the median pixel value for a given dataset.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, x, y
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: x, y
        variables: same as dataset_in
    """
    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    # Save dtypes because masking with Dataset.where() converts to float64.
    band_list = list(dataset_in.data_vars)
    dataset_in_dtypes = {}
    for band in band_list:
        dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and Landsat 7 scan lines.
    dataset_in = dataset_in.where((dataset_in != no_data) & (clean_mask))
    dataset_out = dataset_in.median(dim='time', skipna=True, keep_attrs=False)

    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out


def create_max_ndvi_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, intermediate_product=None, **kwargs):
    """
    Method for calculating the pixel value for the max ndvi value.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, x, y
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: x, y
        variables: same as dataset_in
    """
    dataset_in = dataset_in.copy(deep=True)

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    # Save dtypes because masking with Dataset.where() converts to float64.
    band_list = list(dataset_in.data_vars)
    dataset_in_dtypes = {}
    for band in band_list:
        dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and scan lines.
    dataset_in = dataset_in.where((dataset_in != -9999) & clean_mask)

    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None

    time_slices = range(len(dataset_in.time))
    for timeslice in time_slices:
        dataset_slice = dataset_in.isel(time=timeslice).drop('time')
        ndvi = (dataset_slice.nir - dataset_slice.red) / (dataset_slice.nir + dataset_slice.red)
        ndvi.values[np.invert(clean_mask)[timeslice, ::]] = -1000000000
        dataset_slice['ndvi'] = ndvi
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_slice.data_vars):
                dataset_out[key].values[dataset_slice.ndvi.values > dataset_out.ndvi.values] = \
                    dataset_slice[key].values[dataset_slice.ndvi.values > dataset_out.ndvi.values]
    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out


def create_min_ndvi_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, intermediate_product=None, **kwargs):
    """
    Method for calculating the pixel value for the min ndvi value.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, x, y
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: x, y
        variables: same as dataset_in
    """
    dataset_in = dataset_in.copy(deep=True)

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    # Save dtypes because masking with Dataset.where() converts to float64.
    band_list = list(dataset_in.data_vars)
    dataset_in_dtypes = {}
    for band in band_list:
        dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and scan lines.
    dataset_in = dataset_in.where((dataset_in != -9999) & clean_mask)

    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None

    time_slices = range(len(dataset_in.time))
    for timeslice in time_slices:
        dataset_slice = dataset_in.isel(time=timeslice).drop('time')
        ndvi = (dataset_slice.nir - dataset_slice.red) / (dataset_slice.nir + dataset_slice.red)
        ndvi.values[np.invert(clean_mask)[timeslice, ::]] = 1000000000
        dataset_slice['ndvi'] = ndvi
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_slice.data_vars):
                dataset_out[key].values[dataset_slice.ndvi.values <
                                        dataset_out.ndvi.values] = dataset_slice[key].values[dataset_slice.ndvi.values <
                                                                                             dataset_out.ndvi.values]
    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out


def create_hdmedians_multiple_band_mosaic(dataset_in,
                                          clean_mask=None,
                                          no_data=-9999,
                                          dtype=None,
                                          intermediate_product=None,
                                          operation="median",
                                          **kwargs):
    """
    Calculates the geomedian or geomedoid using a multi-band processing method.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, latitude, longitude (in that order)
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.
    operation: str in ['median', 'medoid']

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: x, y
        variables: same as dataset_in
    """
    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)
    assert operation in ['median', 'medoid'], "Only median and medoid operations are supported."

    # Save dtypes because masking with Dataset.where() converts to float64.
    band_list = list(dataset_in.data_vars)
    dataset_in_dtypes = {}
    for band in band_list:
        dataset_in_dtypes[band] = dataset_in[band].dtype
    # Mask out clouds and scan lines.
    dataset_in = dataset_in.where((dataset_in != no_data) & clean_mask)

    arrays = [dataset_in[band] for band in band_list]
    stacked_data = np.stack(arrays)
    bands_shape, time_slices_shape, lat_shape, lon_shape = stacked_data.shape[0], \
                                                           stacked_data.shape[1], stacked_data.shape[2], \
                                                           stacked_data.shape[3]
    # Reshape to remove lat/lon
    reshaped_stack = stacked_data.reshape(bands_shape, time_slices_shape,
                                          lat_shape * lon_shape)
    # Build zeroes array across time slices.
    hdmedians_result = np.zeros((bands_shape, lat_shape * lon_shape))

    # For each pixel (lat/lon combination), find the geomedian or geomedoid across time.
    for x in range(reshaped_stack.shape[2]):
        try:
            hdmedians_result[:, x] = hd.nangeomedian(
                reshaped_stack[:, :, x], axis=1) if operation == "median" else hd.nanmedoid(
                reshaped_stack[:, :, x], axis=1)
        except ValueError as e:
            # If all bands have nan values across time, the geomedians are nans.
            hdmedians_result[:, x] = np.full((bands_shape), np.nan)
    output_dict = {
        value: (('y', 'x'), hdmedians_result[index, :].reshape(lat_shape, lon_shape))
        for index, value in enumerate(band_list)
    }
    dataset_out = xr.Dataset(output_dict,
                             coords={'x': dataset_in['x'],
                                     'y': dataset_in['y']})
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out

