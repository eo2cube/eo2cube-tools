import rasterio.features
import xarray as xr


def clip_xr2gpd(dataset, gpd):
    """
    Clips a xArray Dataset to a GeoPandasDataframe.

    Description
    ----------
    Takes an xArray dataset and a Geodataframe and clips the xArray dataset to the shape of the Geodataframe.

    Parameters
    ----------
    dataset: xr.Dataset
         A multi-dimensional array with x,y and time dimensions and one or more data variables.

    gpd: geopandas.Geodataframe
        A geodataframe object with one or more observations or variables and a geometry column. A filterd geodataframe
        can also be used as input.

    Returns
    -------
    masked_dataset: xr.Dataset
        A xr.Dataset like the input dataset with only pixels which are within the polygons of the geopandas.Geodataframe.
        Every other pixel is given the value NaN.
    """

    # selects geometry of the desired gpd and formsa boolean mask from it
    ShapeMask = rasterio.features.geometry_mask(
        gpd.loc[:, "geom"],
        out_shape=(len(dataset.latitude), len(dataset.longitude)),
        transform=dataset.geobox.transform,
        invert=True,
    )
    ShapeMask = xr.DataArray(
        ShapeMask, dims=("latitude", "longitude")
    )  # converts boolean mask into an xArray format
    masked_dataset = dataset.where(
        ShapeMask == True
    )  # combines mask and dataset so only the pixels within the gpd
    #                                                   polygons are still valid

    del ShapeMask
    return masked_dataset
