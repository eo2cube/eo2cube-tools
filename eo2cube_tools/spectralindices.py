def spectralindices(dataset, indices=["NDVI"], norm=True, drop=False):

    """Calculate a suite of spectral indices

    Description
    ----------
    Takes an xarray dataset containing spectral bands, calculates one or more
    spectral indices and adds the resulting array as a new variable in the original dataset.

     Indices:
        - EVI: Enhanced Vegetation Index (Huete 2002)
        - LAI: Leaf Area Index (Boegh 2002)
        - MNDWI: Modified Normalised Difference Water Index (Xu 1996)
        - MSAVI: Modified Soil Adjusted Vegetation Index (Qi et al. 1994)
        - NBR: Normalised Burn Ratio (Lopez Garcia 1991)
        - NDVI: Normalised Difference Vegetation Index (Rouse 1973)
        - NDWI: Normalised Difference Water Index (McFeeters 1996)
        - SAVI: Soil Adjusted Vegetation Index (Huete 1988)


    Parameters
    ----------
    dataset : xarray.Dataset
        A two-dimensional or multi-dimensional array containing the spectral bands required to calculate the index.
    indices : str or list of strs
       A string or a list of strings giving the names of the indices to calculate
    norm : bool, optional
        If norm = True values are scaled to 0.0-1.0 by dividing by 10000.0
    drop : bool, optional
        If drop = True only the index will be returned and the other bands will be dropped

    Returns
    -------
    dataset : xarray Dataset
        The original xarray Dataset containing the calculated spectral indices as a DataArray.
    """

    index_dict = {
        "NDVI": lambda dataset: (dataset.nir - dataset.red)
        / (dataset.nir + dataset.red),
        "EVI": lambda dataset: (
            (2.5 * (dataset.nir - dataset.red))
            / (dataset.nir + 6 * dataset.red - 7.5 * dataset.blue + 1)
        ),
        "LAI": lambda dataset: (
            3.618
            * (
                (2.5 * (dataset.nir - dataset.red))
                / (dataset.nir + 6 * dataset.red - 7.5 * dataset.blue + 1)
            )
            - 0.118
        ),
        "SAVI": lambda dataset: (
            (1.5 * (dataset.nir - dataset.red)) / (dataset.nir + dataset.red + 0.5)
        ),
        "MSAVI": lambda dataset: (
            (
                2 * dataset.nir
                + 1
                - ((2 * dataset.nir + 1) ** 2 - 8 * (dataset.nir - dataset.red)) ** 0.5
            )
            / 2
        ),
        "NBR": lambda dataset: (dataset.nir - dataset.swir2)
        / (dataset.nir + dataset.swir2),
        "NDWI": lambda dataset: (dataset.green - dataset.nir)
        / (dataset.green + dataset.nir),
        "MNDWI": lambda dataset: (dataset.green - dataset.swir1)
        / (dataset.green + dataset.swir1),
    }

    if drop:
        drop_bands = list(dataset.data_vars)

    for index in indices:
        if index not in index_dict:
            raise ValueError(f"{index} is not a valid index")
        func = index_dict.get(str(index))
        try:
            if norm:
                index_new = func(dataset / 10000.0)
            else:
                index_new = func(dataset / 1.0)
        except AttributeError:
            raise ValueError(
                "Missing bands: The data set does not seem to contain all bands needed to calculate this index"
            )

        dataset[index] = index_new

    if drop:
        dataset = dataset.drop(drop_bands)

    return dataset
