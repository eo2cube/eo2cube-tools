import folium
import itertools
import math
import numpy as np


def _degree_to_zoom_level(l1, l2, margin=0.0):

    degree = abs(l1 - l2) * (1 + margin)
    zoom_level_int = 0
    if degree != 0:
        zoom_level_float = math.log(360 / degree) / math.log(2)
        zoom_level_int = int(zoom_level_float)
    else:
        zoom_level_int = 18
    return zoom_level_int


def display_map_polygons(
    gdf=None, tooltip_attributes=None, longitude=None, latitude=None
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

    latitude: (float,float)
        A tuple of latitude bounds in (min,max) format.

    longitude: (float, float)
        A tuple of longitude bounds in (min,max) format.


    Returns
    ----------
    folium.Map
        A map centered on the lat lon bounds displaying all shapes of the geodataframe with the defined column
        informations.

    .. _Folium
        https://github.com/python-visualization/folium

    """
    # assert locations is not None
    assert latitude is not None
    assert longitude is not None

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
    if gdf is not None:
        gjson = gdf.to_json()
        folium.features.GeoJson(gjson)
        folium.GeoJson(
            gjson,
            name="Felder",
            style_function=lambda feature: {
                "fillColor": "white",
                "color": "red",
                "weight": 3,
                "fillOpacity": 0.1,
            },
            highlight_function=lambda x: {"weight": 5, "fillOpacity": 0.5},
            tooltip=folium.features.GeoJsonTooltip(fields=tooltip_attributes),
        ).add_to(map_hybrid)
    folium.LayerControl().add_to(map_hybrid)
    return map_hybrid
