import rasterio.features
import xarray as xr


def mask(dataset, gpd, dims):
    """
    Mask a xArray Dataset to a GeoPandasDataframe.

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
      
    dims: list, str
        A list containing the names of the x and y dimension.
          
    Returns
    -------
    masked_dataset: xr.Dataset
        A xr.Dataset like the input dataset with only pixels which are within the polygons of the geopandas.Geodataframe.
        Every other pixel is given the value NaN.
    """
    x = dims[0]
    y = dims[1]
    # selects geometry of the desired gpd and forms a boolean mask from it
    ShapeMask = rasterio.features.geometry_mask(
        gpd.loc[:, "geometry"],
        out_shape=(len(dataset['y']), len(dataset['x'])),
        transform=dataset.geobox.transform,
        invert=True,
    )
    ShapeMask = xr.DataArray(
        ShapeMask, dims=(y, x)
    )  # converts boolean mask into an xArray format
    masked_dataset = dataset.where(
        ShapeMask == True
    )  # combines mask and dataset so only the pixels within the gpd
    #                                                   polygons are still valid

    del ShapeMask
    return masked_dataset
