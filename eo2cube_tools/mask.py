import numpy as np
import xarray as xr
import dask

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


def np_keep(array, mask, nodata, out_array=None):
    if out_array is None:
        out_array = np.full_like(array, nodata)
    else:
        assert out_array.shape == array.shape
        assert out_array.dtype == array.dtype
        assert out_array is not array
        out_array[:] = nodata
    np.copyto(out_array, array, where=mask)
    return out_array

def mask_values(ds, mask, inplace=False, nodata=None):

    if isinstance(ds, xr.Dataset):
        return ds.map(
            lambda ds: mask_values(ds, mask, inplace=inplace), keep_attrs=True)

    assert ds.shape == mask.shape
    if nodata is None:
        nodata = getattr(ds, "nodata", 0)

    if inplace:
        if dask.is_dask_collection(ds):
            raise ValueError("Can not perform inplace operation on a dask array")

        np.copyto(ds.data, nodata, where=~mask.data)
        return ds

    if dask.is_dask_collection(ds):
        data = da.map_blocks(
            keep_good_np,
            ds.data,
            where.data,
            nodata,
            name=randomize("good_pixels"),
            dtype=ds.dtype,
        )
    else:
        data = np_keep(ds.data, mask.data, nodata)

    return xr.DataArray(data, dims=ds.dims, coords=ds.coords, attrs=ds.attrs, name=ds.name)

def slc_mask(dataset, categories = ['Dark features / Shadows','Vegetation', 'Not vegetated', 'Water', 'Unclassified', 'Snow or ice']):
    """
    Takes an xarray dataset, creates a mask based on categories defined in the SCL band and removes
    all Pixels not included in the specified categories.

    Parameters
    ----------
    ds : xarray Dataset
       A two-dimensional or multi-dimensional array including the SCL band

    categories : list
       A list of Sentinel-2 Scene Classification Layer (SCL) names. The default is ['Dark features / Shadows',
       'Vegetation', 'Not vegetated', 'Water', 'Unclassified', 'Snow or ice'] which will return non-cloudy or
       non-shadowed land, snow, water, veg, and non-veg pixels.

    Returns
    -------
    An xarray dataset only including pixels of the specified classes
    """

    assert "scl" in list(dataset.data_vars.keys()), "scl band is missing"

    mask = dataset["scl"].isin([int(k) for k,v in scl_cat.items() if v in categories])

    return mask_values(dataset, mask=mask)
