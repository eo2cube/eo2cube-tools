import rasterio.features
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import warnings


def extract(dataset, gpd, bands, func="raw", na_rm=True):
    """
    Extracts values of xArray Dataset at the locations of spatial data and gives out all cell values or
    calculates zonal statistics for the spatial data.

    Description
    ----------
    Extract values from xArray Dataset at the locations of other spatial data. You can use points, lines,
    or polygons in the form of a GeopandasDataframe. The values are extracted for every band defined in "bands".
    It is possible to get all the "raw" datavalues (all values which are within the shape) for a spatial data shape
    or to use a predefined function (e.g. mean, min, max, std) to calculate zonal statistics of all the values of the
    spatial data shape.

    Parameters
    ----------
    dataset: xr.Dataset
        A multi-dimensional array with x,y and time dimensions and one or more data variables.

    gpd: GeoPandasDataFrame
        A geodataframe object with one or more observations or variables and a geometry column. A filtered geodataframe
        can also be used as input.

    bands: list,string
        List with band names of dataset, e.g (["blue", "green", "red"]), for which values should be extracted.

    func: string, (default = "raw")
        Select name of statistics function like "mean", "max", "min", "std". If function is selected the values are
        calculated with the defined function. Gives out one value for each band. The default is set to "raw". In this
        case no function is applied and the values are displayed in "raw" format (all values) in the output dictionary.

    na_rm: bool, (default = True)
        If True all nan values are eliminated from the results. If False the results are displayed with nan values.
        If a statistic function is defined in "func" the nan values are automatically removed.

    Returns
    -------
    results_scenes: dictionary
        Dict which contains dataframes for every scene of dataset with values of all bands or values calculated by a
        function which are located within the spatial data. The dataframes contain the values for each bands and the
        same index like gpd, to assign the values to a specific spatial data shape. To display the extracted values
        of a single timestep (scene) just select the index of the desired timestep in the output dictionary, like e.g.

        example = extract(dataset, gpd, bands = ["blue", "green", "red"], func = "mean")
        example["scene_index_0"]

    """

    warnings.filterwarnings("ignore")  # ignore warnings for nan values

    results_scenes = {}  # empty array for storring all bandvalues for a single scene
    index = gpd.index  # creates array of indexes of the polygones

    for t in range(len(dataset.time)):  # selects the single scene of a dataset
        scene = dataset.isel(time=t)

        results_df = {}  # empty array for storing band values
        for i in index:
            vec = gpd.loc[i]
            ShapeMask = rasterio.features.geometry_mask(
                vec["geom"],
                # selects geometry of the desired gpd and forms a boolean mask from it
                out_shape=(len(scene.latitude), len(scene.longitude)),
                transform=scene.geobox.transform,
                invert=True,
            )
            ShapeMask = xr.DataArray(
                ShapeMask, dims=("latitude", "longitude")
            )  # converts boolean mask into an xArray format

            masked_dataset = scene.where(
                ShapeMask == True
            )  # combines mask and dataset so only the pixels within the gpd polygons are still valid

            results = {}
            for j in bands:
                values = masked_dataset[j].values
                if (
                    na_rm == True
                ):  # if na remove argument is true remove all nan values before storing the array
                    values = values[np.logical_not(np.isnan(values))]
                    if func == "raw":  # stores "raw" values
                        results[j] = pd.DataFrame({i: [values]})
                    elif func == "mean":  # calculates mean and stores mean value
                        mean = np.mean(values)
                        results[j] = pd.DataFrame({i: [mean]})
                    elif func == "min":  # calculates the min value
                        minn = np.min(values)
                        results[j] = pd.DataFrame({i: [minn]})
                    elif func == "max":  # calculates the max value
                        maxx = np.max(values)
                        results[j] = pd.DataFrame({i: [maxx]})
                    elif func == "std":  # calculates the standard deviation
                        std = np.std(values)
                        results[j] = pd.DataFrame({i: [std]})
                else:  # keeps all nan values
                    if func == "raw":  # stores "raw" values
                        results[j] = pd.DataFrame(
                            {i: [values]}
                        )  # stores the array with nan values
                    elif func == "mean":  # calculates mean and stores mean value
                        mean = np.nanmean(values)  # eliminates nan values
                        results[j] = pd.DataFrame({i: [mean]})
                    elif func == "min":  # calculates the min value
                        minn = np.nanmin(values)
                        results[j] = pd.DataFrame({i: [minn]})
                    elif func == "max":  # calculates the max value
                        maxx = np.nanmax(values)
                        results[j] = pd.DataFrame({i: [maxx]})
                    elif func == "std":  # calculates the standard deviation
                        std = np.nanstd(values)
                        results[j] = pd.DataFrame({i: [std]})

            vec_df = pd.concat(
                results, axis=1
            )  # concatenate all band values of a shape i to a data frame
            vec_df.columns = bands  # rename column names to band names
            vec_df["id"] = i  # add index of polygon

            results_df[str(i)] = vec_df  # store dataframe for in a dictionary

        df = pd.concat(
            results_df, ignore_index=True
        )  # concatenate all dataframes for each shape to one dataframe
        df = df.set_index("id")  # sets index to shape index
        results_scenes[
            "scene_index_" + str(t)
        ] = df  # store dataframe with raw pixel values into dict

    return results_scenes
