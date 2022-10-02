import numpy as np
import xarray as xr
import panel as pn
import holoviews as hv
from holoviews import opts
import datashader as ds
import pandas as pd
from holoviews.operation.datashader import regrid, shade
from datashader.utils import ngjit
import folium
import itertools
import math
import param as pm
import geoviews as gv
import geoviews.tile_sources as gts
from collections import OrderedDict as odict
import rioxarray


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
    wm_dataset = dataset.rio.reproject("EPSG:3857", nodata=np.nan)
    maps   = ['EsriImagery','EsriNatGeo', 'EsriTerrain', 'OSM']
    bases  = odict([(name, gts.tile_sources[name].relabel(name)) for name in maps])
    gopts  = hv.opts.WMTS(responsive=True, xaxis=None, yaxis=None, bgcolor='black', show_grid=False)
    times   = [np.datetime64(ts) for ts in dataset.time.values]
    bands   = [var for var in dataset.data_vars]
    
    class band():
        def __init__(self,dataset, band, time, dims=dims, nodata=nodata):
            self.dataset =dataset
            self.band = band
            self.time = time
            self.dims = dims
            self.nodata = nodata

        def calc_band(self):       
            data = self.dataset[self.band].sel(time=self.time)
            xs, ys = (data[self.dims[0]],data[self.dims[1]],)
            b = ds.utils.orient_array(data)
            a = (np.where(np.logical_or(np.isnan(b), b <= self.nodata), 0, 255)).astype(np.uint8)
            self.view = hv.RGB((xs, ys[::-1], b, b, b, a), vdims=list("RGBA"))
            return hv.RGB((xs, ys[::-1], b, b, b, a), vdims=list("RGBA"))
                                                                                                                              
    class bandExplorer(pm.Parameterized):
        band       = pm.Selector(bands, default= bands[0])
        time       = pm.Selector(times, default=times[0])
        basemap = pm.Selector(bases)
        data_opacity = pm.Magnitude(1.00)
        map_opacity = pm.Magnitude(1.00)
        
        @pm.depends('map_opacity', 'basemap')
        def tiles(self):
            return self.basemap.opts(gopts).opts(alpha=self.map_opacity)
        
        @pm.depends('time', 'band','data_opacity', on_init=True)
        def update_image(self):
            b = band(wm_dataset, band = self.band, time = self.time, dims=['x','y'], nodata= 0)
            b.calc_band()
            return (shade(regrid(b.view,dynamic=False),dynamic=False, cmap=cmap, clims=clims).opts(alpha=self.data_opacity)) 
        
        def view(self):
            return hv.DynamicMap(self.tiles) * hv.DynamicMap(self.update_image)
    
    explorer = bandExplorer(name = 'Image Explorer')    
    col = pn.Row(pn.panel(explorer.param), explorer.view())
    return col


def plot_rgb(
    dataset,
    bands=["blue", "green", "red"],
    dims=["x", "y"]
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
    """
    
    pn.extension()
    wm_dataset = dataset.rio.reproject("EPSG:3857", nodata=np.nan)
    maps   = ['EsriImagery','EsriNatGeo', 'EsriTerrain', 'OSM']
    bases  = odict([(name, gts.tile_sources[name].relabel(name)) for name in maps])
    gopts  = hv.opts.WMTS(responsive=True, xaxis=None, yaxis=None, bgcolor='black', show_grid=False)
    times   = [np.datetime64(ts) for ts in dataset.time.values]
    bands   = [var for var in dataset.data_vars]
    
    class rgb():
        def __init__(self, dataset, red, green, blue, time, dims=dims, nodata=nodata):
            self.r = red
            self.g = green
            self.b = blue
            self.dataset = dataset
            self.time = time
            self.dims = dims
            self.nodata = nodata

        def calc_rgb(self):
            xs, ys = self.dataset[self.r].sel(time=self.time)[self.dims[0]], self.dataset[self.r].sel(time=self.time)[self.dims[1]]
            r, g, b = [
                ds.utils.orient_array(img)
                for img in (
                    self.dataset[self.r].sel(time=self.time),
                    self.dataset[self.g].sel(time=self.time),
                    self.dataset[self.b].sel(time=self.time),
                )
            ]
            a = (np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)).astype(np.uint8)
            r = (normalize_data(r)).astype(np.uint8)
            g = (normalize_data(g)).astype(np.uint8)
            b = (normalize_data(b)).astype(np.uint8)
            self.view = hv.RGB((xs, ys[::-1], r, g, b, a), vdims=list("RGBA"))
            return (
                hv.RGB((xs, ys[::-1], r, g, b, a), vdims=list("RGBA"))
        )
                                                                                                                              
    class rgbExplorer(pm.Parameterized):
        red = pm.Selector(bands, default= bands[0])
        green = pm.Selector(bands, default= bands[1])
        blue = pm.Selector(bands, default= bands[2])
        time = pm.Selector(times, default=times[0])
        basemap = pm.Selector(bases)
        data_opacity = pm.Magnitude(1.00)
        map_opacity = pm.Magnitude(1.00)
        
        @pm.depends('map_opacity', 'basemap')
        def tiles(self):
            return self.basemap.opts(gopts).opts(alpha=self.map_opacity)
        
        @pm.depends('time', 'red','green','blue','data_opacity', on_init=True)
        def update_image(self):
            b = rgb(wm_dataset, red = self.red, blue = self.blue, green = self.green, time = self.time, dims=['x','y'], nodata= 0)
            b.calc_rgb()
            return (regrid(b.view,dynamic=False)).opts(alpha=self.data_opacity)
        
        def view(self):
            return hv.DynamicMap(self.tiles) * hv.DynamicMap(self.update_image)
    
    explorer = rgbExplorer(name = 'Image Explorer')    
    col = pn.Row(pn.panel(explorer.param), explorer.view())
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
    


def _degree_to_zoom_level(l1, l2, margin=0.0):

    degree = abs(l1 - l2) * (1 + margin)
    zoom_level_int = 0
    if degree != 0:
        zoom_level_float = math.log(360 / degree) / math.log(2)
        zoom_level_int = int(zoom_level_float)
    else:
        zoom_level_int = 18
    return zoom_level_int


def map_polygon(
    gdf=None, tooltip_attributes=None
):
    """
    Generates a folium map based on a lat-lon bounded rectangle.
    Description
    ----------
    Takes a geodataframe and plots the shapes of the geodataframe in a folium map. The extent is defined by a lat-lon
    bounded rectangle. Also information or column values from the geodataframe can be defined for single shapes.
    Parameters
    ----------
    gdf: geopandas.Geodataframe
        A geodataframe object with one or more observations or variables and a geometry column. A filterd geodataframe
        can also be used as input.
    tooltop_attributes: (string,string)
        A tuple of column names of the geodataframe. The value of the defined columns are displayed for each observation
        in the folium map by clicking on the shape.
    Returns
    ----------
    folium.Map
        A map centered on the lat lon bounds displaying all shapes of the geodataframe with the defined column
        informations.
    .. _Folium
        https://github.com/python-visualization/folium
    """
    
    longitude = (gdf.total_bounds[0],gdf.total_bounds[2])
    latitude = (gdf.total_bounds[1], gdf.total_bounds[3])

    ###### ###### ######   CALC ZOOM LEVEL     ###### ###### ######
    margin = -0.5
    zoom_bias = 0

    lat_zoom_level = _degree_to_zoom_level(margin=margin, *latitude) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(margin=margin, *longitude) + zoom_bias
    zoom_level = min(lat_zoom_level, lon_zoom_level)

    ###### ###### ######   CENTER POINT        ###### ###### ######

    center = [np.mean(latitude), np.mean(longitude)]

    ###### ###### ######   CREATE MAP         ###### ###### ######
    map_hybrid = folium.Map(location=center, zoom_start=zoom_level, control_scale=True)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ).add_to(map_hybrid)
    ###### ###### ######     POLYGONS    ###### ###### ######
    
    if tooltip_attributes != None:
    	tooltip_attributes = folium.features.GeoJsonTooltip(fields=tooltip_attributes)
    
    if gdf is not None:
        gjson = gdf.to_json()
        folium.features.GeoJson(gjson)
        folium.GeoJson(
            gjson,
            name="Polygons",
            style_function=lambda feature: {
                "fillColor": "white",
                "color": "red",
                "weight": 3,
                "fillOpacity": 0.1,
            },
            highlight_function=lambda x: {"weight": 5, "fillOpacity": 0.2},
            tooltip=tooltip_attributes,
        ).add_to(map_hybrid)
    folium.LayerControl().add_to(map_hybrid)
    return map_hybrid
