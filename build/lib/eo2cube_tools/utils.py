import xarray
import pandas as pd


def remove_empty_scenes(dataset, nodata=-9999, bands=["red"], dim=["x", "y"]):

    """Remove time steps containing only no data values

    Description
    ----------
    Takes an xarray dataset an drops all time steps which only contain no data values

    Parameters
    ----------
    dataset : xarray.Dataset
        A two-dimensional or multi-dimensional array
    nodata : int
       nodata value used for filtering empty time steps
    band : list
        A list of bands which should be used to check for no data scenes
    dim : list
        A list containing the names of the x and y dimension

    Returns
    -------
    dataset : xarray Dataset

    """

    mean = dataset.mean(dim=dim)
    ts = len(dataset.time)

    for band in bands:
        dataset.sel(
            time=pd.to_datetime(
                mean.time.where(mean[band] != nodata, drop=True).values.tolist()
            )
        )

    tsnew = len(dataset.time)

    print("Found {0} time steps with no data".format(ts - tsnew))

    return dataset
