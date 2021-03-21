import odc.algo

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

    return odc.algo.keep_good_only(dataset, where=mask)
