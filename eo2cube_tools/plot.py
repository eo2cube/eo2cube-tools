import numpy as np
import xarray as xr
import panel as pn
import holoviews as hv
from holoviews import opts
import datashader as ds
import pandas as pd
from holoviews.operation.datashader import regrid, shade
from datashader.utils import ngjit


@ngjit
def normalize_data(agg):
    out = np.zeros_like(agg)
    min_val = 0
    max_val = 2 ** 16 - 1
    range_val = max_val - min_val
    col, rows = agg.shape
    c = 70
    th = 0.02
    for x in range(col):
        for y in range(rows):
            val = agg[x, y]
            norm = (val - min_val) / range_val
            norm = 1 / (1 + np.exp(c * (th - norm)))  # bonus
            out[x, y] = norm * 255.0
    return out


def plot_band(
    dataset,
    dims=["x", "y"],
    height=500,
    width=500,
    clims=None,
    norm="eq_hist",
    cmap=["black", "white"],
    nodata=1,
):

    """Interactive visualization of xarray time series

    Description
    ----------
    Takes an xarray time dataset and creates an interactive panel which allows to inspect the different data variables at different time steps

    Parameters
    ----------
    dataset : xarray.Dataset
        A multi-dimensional array with x,y and time dimensions and one or more data variables.
    dim : list, str
        A list containing the names of the x and y dimension.
    height: int
        Height of the created plot specified in pixels.
    width: int
        Width of the created plot specified in pixels.
    clims: int,float
        Min and max data values to use for colormap interpolation.
    norm :
        The normalization operation applied before colormapping. Valid options include 'linear', 'log', 'eq_hist', 'cbrt'.
    cmap : list, str
        Used for the colormap of single-layer datashader output.
    nodata: int
        Value defining the nodata value for the dataset

    """

    pn.extension()

    list_vars = []
    time_list = []

    for ts in dataset.time.values:
        ts = pd.Timestamp(ts).to_pydatetime("%Y-%M-%D")
        time_list.append(ts)

    for var in dataset.data_vars:
        list_vars.append(var)

    bands_select = pn.widgets.Select(name="Band", value=list_vars[0], options=list_vars)
    time_select = pn.widgets.Select(name="Time", value=time_list[0], options=time_list)

    def one_band(band, time):
        xs, ys = (
            dataset[band].sel(time=time)[dims[0]],
            dataset[band].sel(time=time)[dims[1]],
        )
        b = ds.utils.orient_array(dataset[band].sel(time=time))
        a = (np.where(np.logical_or(np.isnan(b), b <= nodata), 0, 255)).astype(np.uint8)
        return (
            shade(
                regrid(hv.RGB((xs, ys[::-1], b, b, b, a), vdims=list("RGBA"))),
                cmap=cmap,
                clims=clims,
                normalization=norm,
            )
            .redim(x=dims[0], y=dims[1])
            .opts(width=width, height=height)
        )

    def on_var_select(event):
        var = event.obj.value
        col[-1] = one_band(band=bands_select.value, time=time_select.value)
        print(time=time_select.value)

    def on_time_select(event):
        time = event.obj.value
        col[-1] = one_band(band=bands_select.value, time=time_select.value)
        print(time_select.value)

    bands_select.param.watch(on_var_select, parameter_names=["value"])
    time_select.param.watch(on_time_select, parameter_names=["value"])
    col = pn.Row(
        pn.Column(pn.WidgetBox(bands_select, time_select)),
        one_band(band=bands_select.value, time=time_select.value),
    )

    return col


def plot_rgb(
    dataset,
    bands=["blue", "green", "red"],
    dims=["x", "y"],
    height=700,
    width=700,
    clims=None,
    norm="eq_hist",
    cmap=["black", "white"],
    nodata=1,
):

    """Interactive RGB visualization of a xarray time series

    Description
    ----------
    Takes an xarray time dataset and creates an interactive panel which allows to inspect the different data variables at different time steps

    Parameters
    ----------
    dataset : xarray.Dataset
        A multi-dimensional array with x,y and time dimensions and one or more data variables.
    bands: int, str
        A list of names defining the data variables used for the red, green and blue band
    dim : list, str
        A list containing the names of the x and y dimension.
    height: int
        Height of the created plot specified in pixels.
    width: int
        Width of the created plot specified in pixels.
    clims: int,float
        Min and max data values to use for colormap interpolation.
    norm :
        The normalization operation applied before colormapping. Valid options include 'linear', 'log', 'eq_hist', 'cbrt'.
    cmap : list, str
        Used for the colormap of single-layer datashader output.
    nodata: int
        Value defining the nodata value for the dataset

    """

    pn.extension()

    list_vars = []
    time_list = []

    for ts in dataset.time.values:
        ts = pd.Timestamp(ts).to_pydatetime("%Y-%M-%D")
        time_list.append(ts)

    for var in dataset.data_vars:
        list_vars.append(var)

    r_select = pn.widgets.Select(name="R", value="red", options=list_vars)
    g_select = pn.widgets.Select(name="G", value="green", options=list_vars)
    b_select = pn.widgets.Select(name="B", value="blue", options=list_vars)
    time_select = pn.widgets.Select(name="Time", value=time_list[0], options=time_list)

    def combine_bands(r, g, b, time):
        xs, ys = dataset[r].sel(time=time)[dims[0]], dataset[r].sel(time=time)[dims[1]]
        r, g, b = [
            ds.utils.orient_array(img)
            for img in (
                dataset[r].sel(time=time),
                dataset[g].sel(time=time),
                dataset[b].sel(time=time),
            )
        ]
        a = (np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)).astype(np.uint8)
        r = (normalize_data(r)).astype(np.uint8)
        g = (normalize_data(g)).astype(np.uint8)
        b = (normalize_data(b)).astype(np.uint8)
        return (
            regrid(hv.RGB((xs, ys[::-1], r, g, b, a), vdims=list("RGBA")))
            .redim(x=dims[0], y=dims[1])
            .opts(width=width, height=height)
        )

    def on_r_select(event):
        var = event.obj.value
        col[-1] = combine_bands(
            r=r_select.value, b=b_select.value, g=g_select.value, time=time_select.value
        )

    def on_b_select(event):
        var = event.obj.value
        col[-1] = combine_bands(
            r=r_select.value, b=b_select.value, g=g_select.value, time=time_select.value
        )

    def on_g_select(event):
        var = event.obj.value
        col[-1] = combine_bands(
            r=r_select.value, b=b_select.value, g=g_select.value, time=time_select.value
        )

    def on_time_select(event):
        time = event.obj.value
        col[-1] = combine_bands(
            r=r_select.value, b=b_select.value, g=g_select.value, time=time_select.value
        )

    r_select.param.watch(on_r_select, parameter_names=["value"])
    g_select.param.watch(on_r_select, parameter_names=["value"])
    b_select.param.watch(on_r_select, parameter_names=["value"])
    time_select.param.watch(on_time_select, parameter_names=["value"])

    col = pn.Row(
        pn.Column(pn.WidgetBox(r_select, g_select, b_select, time_select)),
        combine_bands(
            r=r_select.value, b=b_select.value, g=g_select.value, time=time_select.value
        ),
    )
    return col


def spectral_analyze(
    dataset,
    timeindex=0,
    bands=["red", "green", "blue"],
    dims=["x", "y"],
    height=500,
    width=500,
    clims=None,
    norm="eq_hist",
    cmap=["black", "white"],
    nodata=1,
):

    """Interactivly visualize the spectral profile of single pixels in a multi-band xarray dataset

    Description
    ----------
    Takes an xarray dataset and creates an interactive panel which allows to inspect the spectral profile for each pixel in a multi-band xarray dataset

    Parameters
    ----------
    dataset : xarray.Dataset
        A multi-dimensional array with x,y and time dimensions and one or more data variables.
    timeindex : int
        Integer value used to select one time step from the input dataset for plotting.
    bands: int, str
        A list of names defining the data variables used for the red, green and blue band
    dim : list, str
        A list containing the names of the x and y dimension.
    height: int
        Height of the created plot specified in pixels.
    width: int
        Width of the created plot specified in pixels.
    clims: int,float
        Min and max data values to use for colormap interpolation.
    norm :
        The normalization operation applied before colormapping. Valid options include 'linear', 'log', 'eq_hist', 'cbrt'.
    cmap : list, str
        Used for the colormap of single-layer datashader output.
    nodata: int
        Value defining the nodata value for the dataset

    """

    timestep = str(dataset.time.values[timeindex])

    list_vars = []
    for var in dataset.data_vars:
        list_vars.append(var)

    x = np.mean(dataset.x.values)
    y = np.mean(dataset.y.values)

    def combine_bands():
        xs, ys = (
            dataset[bands[0]].sel(time=timestep)[dims[0]],
            dataset[bands[0]].sel(time=timestep)[dims[1]],
        )
        r, g, b = [
            ds.utils.orient_array(img)
            for img in (
                dataset[bands[0]].sel(time=timestep),
                dataset[bands[1]].sel(time=timestep),
                dataset[bands[2]].sel(time=timestep),
            )
        ]
        a = (np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)).astype(np.uint8)
        r = (normalize_data(r)).astype(np.uint8)
        g = (normalize_data(g)).astype(np.uint8)
        b = (normalize_data(b)).astype(np.uint8)
        return regrid(hv.RGB((xs, ys[::-1], r, g, b, a), vdims=list("RGBA"))).redim(
            x=dims[0], y=dims[1]
        )

    def spectrum(x, y):
        try:
            values = []
            for b in list_vars:
                values.append(
                    dataset[b].sel(x=x, y=y, time=timestep, method="nearest").values
                )
        except:
            values = np.zeros(11)
        return hv.Curve(values)

    tap = hv.streams.PointerXY(x=x, y=y)
    spectrum_curve = hv.DynamicMap(spectrum, streams=[tap])

    return combine_bands() * spectrum_curve
