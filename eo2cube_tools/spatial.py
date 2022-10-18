import numpy as np
import geopandas as gpd
import xarray as xr
import pandas as pd

def coords_to_indices(x, y, transform):
    col_index, row_index = ~transform * (x, y)
    return np.int64(col_index), np.int64(row_index)


def extract_by_points(ds, gdf, bands):
    if isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataArray):
        pass
    else:
        raise TypeError (f'ds must be of type {xr.Dataset} or {xr.DataArray}')
    if isinstance:
        pass
    else:
        raise TypeError (f'gdf must be of type {gpd.GeoDataFrame}')
    if isinstance(bands, list):
        pass
    else:
        raise TypeError (f'bands must be of type {str(list)}')
        
    data = ds[bands]

    x, y = coords_to_indices(
                    gdf.geometry.x.values, gdf.geometry.y.values, ds.odc.geobox.affine)

            #for band in bands_names
    yidx = xr.DataArray(y, dims='z')
    xidx = xr.DataArray(x, dims='z')
    res = data.isel(y=yidx, x=xidx)

    ime_names = res.time.values
    time_format='%Y%m%d'
    time_names = [f't{t}' for t in  np.datetime_as_string(ds.time.values).tolist()]
    
    for band in bands:
        band_names_concat = []
        for t in time_names:
            band_names_concat.append(f'{t}_{band}')
        df = pd.concat(
            (
            gdf,pd.DataFrame(
                data=res[band].values.T.squeeze(),
                columns=band_names_concat,
            ),
        ),
        axis=1,
        )
        
    return df
