'''
Description: This file contains a python function for animating 
satellite data time-series, ts_animation() developed by Digital 
Earth Australia data using following conditions:

License: The code in this notebook is licensed under the Apache License, 
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Digital Earth 
Australia data is licensed under the Creative Commons by Attribution 4.0 
license (https://creativecommons.org/licenses/by/4.0/).

Contact: If you need assistance, please post a question on the Open Data
Cube Slack channel (http://slack.opendatacube.org/) or on the GIS Stack 
Exchange (https://gis.stackexchange.com/questions/ask?tags=open-data-cube) 
using the `open-data-cube` tag (you can view previously asked questions 
here: https://gis.stackexchange.com/questions/tagged/open-data-cube). 

If you would like to report an issue with this script, file one on 
Github: https://github.com/GeoscienceAustralia/dea-notebooks/issues/new


Last modified: October 2020


This function was tested in the eo2cube environments using a single bansd,
credits goes to DEA!


'''

# Import required packages
import math
import branca
import folium
import calendar
import ipywidgets
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from pyproj import Proj, transform
from IPython.display import display
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipyleaflet import Map, Marker, Popup, GeoJSON, basemaps, Choropleth
from skimage import exposure
from odc.ui import image_aspect

from matplotlib.animation import FuncAnimation
import pandas as pd
from pathlib import Path
from shapely.geometry import box
from skimage.exposure import rescale_intensity
from tqdm.auto import tqdm
import warnings



def ts_animation(ds,
                 bands=None,
                 output_path='timeseries.mp4',
                 width_pixels=750,
                 interval=100,                 
                 percentile_stretch=(0.01, 0.99),
                 image_proc_funcs=None,
                 show_gdf=None,
                 show_date='%d %b %Y',
                 show_text=None,
                 show_colorbar=True,
                 gdf_kwargs={},
                 annotation_kwargs={},
                 imshow_kwargs={},
                 colorbar_kwargs={},
                 limit=None):
    
    """
    Takes an `xarray` timeseries and animates the data as either a 
    three-band (e.g. true or false colour) or single-band animation, 
    allowing changes in the landscape to be compared across time.
    
    Animations can be customised to include text and date annotations 
    or use specific combinations of input bands. Vector data can be 
    overlaid and animated on top of imagery, and custom image 
    processing functions can be applied to each frame.
    
    Supports .mp4 (ideal for Twitter/social media) and .gif (ideal 
    for all purposes, but can have large file sizes) format files. 
    
    Last modified: October 2020
    
    Parameters
    ----------  
    ds : xarray.Dataset
        An xarray dataset with multiple time steps (i.e. multiple 
        observations along the `time` dimension).        
    bands : list of strings
        An list of either one or three band names to be plotted, 
        all of which must exist in `ds`. 
    output_path : str, optional
        A string giving the output location and filename of the 
        resulting animation. File extensions of '.mp4' and '.gif' are 
        accepted. Defaults to 'animation.mp4'.
    width_pixels : int, optional
        An integer defining the output width in pixels for the 
        resulting animation. The height of the animation is set 
        automatically based on the dimensions/ratio of the input 
        xarray dataset. Defaults to 500 pixels wide.        
    interval : int, optional
        An integer defining the milliseconds between each animation 
        frame used to control the speed of the output animation. Higher
        values result in a slower animation. Defaults to 100 
        milliseconds between each frame.         
    percentile_stretch : tuple of floats, optional
        An optional tuple of two floats that can be used to clip one or
        three-band arrays by percentiles to produce a more vibrant, 
        visually attractive image that is not affected by outliers/
        extreme values. The default is `(0.02, 0.98)` which is 
        equivalent to xarray's `robust=True`. This parameter is ignored
        completely if `vmin` and `vmax` are provided as kwargs to
        `imshow_kwargs`.
    image_proc_funcs : list of funcs, optional
        An optional list containing functions that will be applied to 
        each animation frame (timestep) prior to animating. This can 
        include image processing functions such as increasing contrast, 
        unsharp masking, saturation etc. The function should take AND 
        return a `numpy.ndarray` with shape [y, x, bands]. If your 
        function has parameters, you can pass in custom values using 
        a lambda function:
        `image_proc_funcs=[lambda x: custom_func(x, param1=10)]`.
    show_gdf: geopandas.GeoDataFrame, optional
        Vector data (e.g. ESRI shapefiles or GeoJSON) can be optionally
        plotted over the top of imagery by supplying a 
        `geopandas.GeoDataFrame` object. To customise colours used to
        plot the vector features, create a new column in the
        GeoDataFrame called 'colors' specifying the colour used to plot 
        each feature: e.g. `gdf['colors'] = 'red'`.
        To plot vector features at specific moments in time during the
        animation, create new 'start_time' and/or 'end_time' columns in
        the GeoDataFrame that define the time range used to plot each 
        feature. Dates can be provided in any string format that can be 
        converted using the `pandas.to_datetime()`. e.g.
         `gdf['end_time'] = ['2001', '2005-01', '2009-01-01']`    
    show_date : string or bool, optional
        An optional string or bool that defines how (or if) to plot 
        date annotations for each animation frame. Defaults to 
        '%d %b %Y'; can be customised to any format understood by 
        strftime (https://strftime.org/). Set to False to remove date 
        annotations completely.       
    show_text : str or list of strings, optional
        An optional string or list of strings with a length equal to 
        the number of timesteps in `ds`. This can be used to display a 
        static text annotation (using a string), or a dynamic title 
        (using a list) that displays different text for each timestep. 
        By default, no text annotation will be plotted.        
    show_colorbar : bool, optional
        An optional boolean indicating whether to include a colourbar 
        for single-band animations. Defaults to True.
    gdf_kwargs : dict, optional
        An optional dictionary of keyword arguments to customise the 
        appearance of a `geopandas.GeoDataFrame` supplied to 
        `show_gdf`. Keyword arguments are passed to `GeoSeries.plot` 
        (see http://geopandas.org/reference.html#geopandas.GeoSeries.plot). 
        For example: `gdf_kwargs = {'linewidth': 2}`. 
    annotation_kwargs : dict, optional
        An optional dict of keyword arguments for controlling the 
        appearance of  text annotations. Keyword arguments are passed 
        to `matplotlib`'s `plt.annotate` 
        (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html 
        for options). For example, `annotation_kwargs={'fontsize':20, 
        'color':'red', 'family':'serif'}.  
    imshow_kwargs : dict, optional
        An optional dict of keyword arguments for controlling the 
        appearance of arrays passed to `matplotlib`'s `plt.imshow` 
        (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html 
        for options). For example, a green colour scheme and custom
        stretch could be specified using: 
        `onebandplot_kwargs={'cmap':'Greens`, 'vmin':0.2, 'vmax':0.9}`.
        (some parameters like 'cmap' will only have an effect for 
        single-band animations, not three-band RGB animations).
    colorbar_kwargs : dict, optional
        An optional dict of keyword arguments used to control the 
        appearance of the colourbar. Keyword arguments are passed to
        `matplotlib.pyplot.tick_params` 
        (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.tick_params.html
        for options). This can be used to customise the colourbar 
        ticks, e.g. changing tick label colour depending on the 
        background of the animation: 
        `colorbar_kwargs={'colors': 'black'}`.
    limit: int, optional
        An optional integer specifying how many animation frames to 
        render (e.g. `limit=50` will render the first 50 frames). This
        can be useful for quickly testing animations without rendering 
        the entire time-series.    
            
    """

    def _start_end_times(gdf, ds):
        """
        Converts 'start_time' and 'end_time' columns in a 
        `geopandas.GeoDataFrame` to datetime objects to allow vector
        features to be plotted at specific moments in time during an
        animation, and sets default values based on the first
        and last time in `ds` if this information is missing from the
        dataset.
        """

        # Make copy of gdf so we do not modify original data
        gdf = gdf.copy()

        # Get min and max times from input dataset
        minmax_times = pd.to_datetime(ds.time.isel(time=[0, -1]).values)

        # Update both `start_time` and `end_time` columns
        for time_col, time_val in zip(['start_time', 'end_time'], minmax_times):

            # Add time_col if it does not exist
            if time_col not in gdf:
                gdf[time_col] = np.nan

            # Convert values to datetimes and fill gaps with relevant time value
            gdf[time_col] = pd.to_datetime(gdf[time_col], errors='ignore')
            gdf[time_col] = gdf[time_col].fillna(time_val)

        return gdf


    def _add_colorbar(fig, ax, vmin, vmax, imshow_defaults, colorbar_defaults):
        """
        Adds a new colorbar axis to the animation with custom minimum 
        and maximum values and styling.
        """

        # Create new axis object for colorbar
        cax = fig.add_axes([0.02, 0.02, 0.96, 0.03])

        # Initialise color bar using plot min and max values
        img = ax.imshow(np.array([[vmin, vmax]]), **imshow_defaults)
        fig.colorbar(img,
                     cax=cax,
                     orientation='horizontal',
                     ticks=np.linspace(vmin, vmax, 2))

        # Fine-tune appearance of colorbar
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(axis='x', **colorbar_defaults)
        cax.get_xticklabels()[0].set_horizontalalignment('left')
        cax.get_xticklabels()[-1].set_horizontalalignment('right')


    def _frame_annotation(times, show_date, show_text):
        """
        Creates a custom annotation for the top-right of the animation
        by converting a `xarray.DataArray` of times into strings, and
        combining this with a custom text annotation. Handles cases 
        where `show_date=False/None`, `show_text=False/None`, or where
        `show_text` is a list of strings.
        """

        # Test if show_text is supplied as a list
        is_sequence = isinstance(show_text, (list, tuple, np.ndarray))

        # Raise exception if it is shorter than number of dates
        if is_sequence and (len(show_text) == 1):
            show_text, is_sequence = show_text[0], False
        elif is_sequence and (len(show_text) < len(times)):
            raise ValueError(f'Annotations supplied via `show_text` must have '
                             f'either a length of 1, or a length >= the number '
                             f'of timesteps in `ds` (n={len(times)})')

        times_list = (times.dt.strftime(show_date).values if show_date else [None] *
                      len(times))
        text_list = show_text if is_sequence else [show_text] * len(times)
        annotation_list = ['\n'.join([str(i) for i in (a, b) if i])
                           for a, b in zip(times_list, text_list)]

        return annotation_list


    def _update_frames(i, ax, extent, annotation_text, gdf, gdf_defaults,
                       annotation_defaults, imshow_defaults):
        """
        Animation called by `matplotlib.animation.FuncAnimation` to 
        animate each frame in the animation. Plots array and any text
        annotations, as well as a temporal subset of `gdf` data based
        on the times specified in 'start_time' and 'end_time' columns.
        """        

        # Clear previous frame to optimise render speed and plot imagery
        ax.clear()
        ax.imshow(array[i, ...].clip(0.0, 1.0), 
                  extent=extent, 
                  vmin=0.0, vmax=1.0, 
                  **imshow_defaults)

        # Add annotation text
        ax.annotate(annotation_text[i], **annotation_defaults)

        # Add geodataframe annotation
        if show_gdf is not None:

            # Obtain start and end times to filter geodataframe features
            time_i = ds.time.isel(time=i).values

            # Subset geodataframe using start and end dates
            gdf_subset = show_gdf.loc[(show_gdf.start_time <= time_i) &
                                      (show_gdf.end_time >= time_i)]           

            if len(gdf_subset.index) > 0:

                # Set color to geodataframe field if supplied
                if ('color' in gdf_subset) and ('color' not in gdf_kwargs):
                    gdf_defaults.update({'color': gdf_subset['color'].tolist()})

                gdf_subset.plot(ax=ax, **gdf_defaults)

        # Remove axes to show imagery only
        ax.axis('off')
        
        # Update progress bar
        progress_bar.update(1)
        
    
    # Test if bands have been supplied, or convert to list to allow
    # iteration if a single band is provided as a string
    if bands is None:
        raise ValueError(f'Please use the `bands` parameter to supply '
                         f'a list of one or three bands that exist as '
                         f'variables in `ds`, e.g. {list(ds.data_vars)}')
    elif isinstance(bands, str):
        bands = [bands]
    
    # Test if bands exist in dataset
    missing_bands = [b for b in bands if b not in ds.data_vars]
    if missing_bands:
        raise ValueError(f'Band(s) {missing_bands} do not exist as '
                         f'variables in `ds` {list(ds.data_vars)}')
    
    # Test if time dimension exists in dataset
    if 'time' not in ds.dims:
        raise ValueError(f"`ds` does not contain a 'time' dimension "
                         f"required for generating an animation")
                
    # Set default parameters
    outline = [PathEffects.withStroke(linewidth=2.5, foreground='black')]
    annotation_defaults = {
        'xy': (1, 1),
        'xycoords': 'axes fraction',
        'xytext': (-5, -5),
        'textcoords': 'offset points',
        'horizontalalignment': 'right',
        'verticalalignment': 'top',
        'fontsize': 20,
        'color': 'white',
        'path_effects': outline
    }
    imshow_defaults = {'cmap': 'magma', 'interpolation': 'nearest'}
    colorbar_defaults = {'colors': 'white', 'labelsize': 12, 'length': 0}
    gdf_defaults = {'linewidth': 1.5}

    # Update defaults with kwargs
    annotation_defaults.update(annotation_kwargs)
    imshow_defaults.update(imshow_kwargs)
    colorbar_defaults.update(colorbar_kwargs)
    gdf_defaults.update(gdf_kwargs)

    # Get info on dataset dimensions
    height, width = ds.geobox.shape
    scale = width_pixels / width
    left, bottom, right, top = ds.geobox.extent.boundingbox

    # Prepare annotations
    annotation_list = _frame_annotation(ds.time, show_date, show_text)

    # Prepare geodataframe
    if show_gdf is not None:
        show_gdf = show_gdf.to_crs(ds.geobox.crs)
        show_gdf = gpd.clip(show_gdf, mask=box(left, bottom, right, top))
        show_gdf = _start_end_times(show_gdf, ds)

    # Convert data to 4D numpy array of shape [time, y, x, bands]
    ds = ds[bands].to_array().transpose(..., 'variable')[0:limit, ...]
    array = ds.astype(np.float32).values

    # Optionally apply image processing along axis 0 (e.g. to each timestep)
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ({remaining_s:.1f} ' \
                   'seconds remaining at {rate_fmt}{postfix})'
    if image_proc_funcs:
        print('Applying custom image processing functions')
        for i, array_i in tqdm(enumerate(array),
                               total=len(ds.time),
                               leave=False,
                               bar_format=bar_format,
                               unit=' frames'):
            for func in image_proc_funcs:
                array_i = func(array_i)
            array[i, ...] = array_i

    # Clip to percentiles and rescale between 0.0 and 1.0 for plotting
    vmin, vmax = np.quantile(array[np.isfinite(array)], q=percentile_stretch)
        
    # Replace with vmin and vmax if present in `imshow_defaults`
    if 'vmin' in imshow_defaults:
        vmin = imshow_defaults.pop('vmin')
    if 'vmax' in imshow_defaults:
        vmax = imshow_defaults.pop('vmax')
    
    # Rescale between 0 and 1
    array = rescale_intensity(array, 
                              in_range=(vmin, vmax), 
                              out_range=(0.0, 1.0))
    array = np.squeeze(array)  # remove final axis if only one band

    # Set up figure
    fig, ax = plt.subplots()
    fig.set_size_inches(width * scale / 72, height * scale / 72, forward=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    # Optionally add colorbar
    if show_colorbar & (len(bands) == 1):
        _add_colorbar(fig, ax, vmin, vmax, imshow_defaults, colorbar_defaults)

    # Animate
    print(f'Exporting animation to {output_path}') 
    anim = FuncAnimation(
        fig=fig,
        func=_update_frames,
        fargs=(
            ax,  # axis to plot into
            [left, right, bottom, top],  # imshow extent
            annotation_list,  # list of text annotations
            show_gdf,  # geodataframe to plot over imagery
            gdf_defaults,  # any kwargs used to plot gdf
            annotation_defaults,  # kwargs for annotations
            imshow_defaults),  # kwargs for imshow
        frames=len(ds.time),
        interval=interval,
        repeat=False)
    
    # Set up progress bar
    progress_bar = tqdm(total=len(ds.time), 
                        unit=' frames', 
                        bar_format=bar_format) 

    # Export animation to file
    if Path(output_path).suffix == '.gif':
        anim.save(output_path, writer='pillow')
    else:
        anim.save(output_path, dpi=72)

    # Update progress bar to fix progress bar moving past end
    if progress_bar.n != len(ds.time):
        progress_bar.n = len(ds.time)
        progress_bar.last_print_n = len(ds.time)
