import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import odc.geo.xr
from odc.geo.geobox import GeoBox
from odc.geo import resyx_, wh_
from odc.geo.crs import CRS
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from rasterio.enums import MergeAlg
import rasterio.features as features


class extracter():
    def __init__(self,ds, gdf, bands, attribute = 'id'):
        self.ds = ds[bands]
        if not hasattr(self.ds, 'geobox'):
            self.ds = self.add_geobox(self.ds)
        self.gdf = gdf
        self.attribute = attribute
        self.xr_crs = self.ds.odc.geobox.crs.to_wkt()
        self.gdf_crs = self.gdf.crs.to_wkt()
        if self.xr_crs != self.gdf_crs:
            self.gdf = self.gdf.to_crs(self.ds.odc.geobox.crs.crs_str) 
        if self.gdf.within(self.ds.odc.geobox.extent.geom).any() == False:
            self.gdf = self.fit_to_xr(self.gdf, self.ds)
        
        self.bands = bands
        
    def add_geobox(self, xds):
        height, width = xds.rio.shape
        try:
            transform = xds.rio.transform()
        except AttributeError:
            transform = xds[xds.rio.vars[0]].rio.transform()
        xds.attrs['geobox'] = GeoBox(
            shape=wh_(width, height),
            affine=transform,
            crs=CRS(xds.rio.crs.to_wkt()),
        )
        return xds
    
    def fit_to_xr(self, gdf, ds):
        if isinstance(gdf.iloc[0].geometry, (Polygon, MultiPolygon)):
            gdf = gpd.overlay(
                gdf,
                gpd.GeoDataFrame(data=[0], geometry=[self.ds.odc.geobox.extent.geom], crs=gdf.crs),
                how='intersection',).drop(columns=[0])
        else:
            gdf = gdf[gdf.geometry.intersects(self.ds.odc.geobox.extent.geom)]
                  
        return gdf
    
    def rasterize(self,
        gdf,
        attribute,
        merge_alg = 'replace',
        all_touched=False,
    ):

        crs = self.xr_crs
        transform = self.ds.odc.geobox.transform
        dims = self.ds.odc.geobox.dims
        xy_coords = [self.ds[dims[0]], self.ds[dims[1]]]
        y, x = self.ds.odc.geobox.shape
        
        shapes = zip(self.gdf.geometry, self.gdf[attribute])

        if merge_alg == 'replace':
            algo = MergeAlg.replace
        elif merge_alg == 'add':
            algo = MergeAlg.add

        arr = features.rasterize(
            shapes=shapes, out_shape=(y, x), fill=np.nan,  transform=transform, all_touched=all_touched, merge_alg = algo,  default_value=1, dtype = None,
        )
        xarr = xr.DataArray(
            arr,
            coords=xy_coords,
            dims=dims,
            attrs=self.ds.attrs,
        )
        return xarr
    
    def sample_feature_by_shape(self,
        ds,
        gdf_row,
        attribute,
        subset,
        merge_alg,
        all_touched,
                                
        ):
        
        df_columns = gdf.columns
        fid = gdf_row[attribute]
        other_cols = [col for col in df_columns if col not in ['extractor_id', 'geometry']]
        raster = self.rasterize(
                    gdf_row,
                    attribute = attribute,
                    merge_alg = merge_alg,
                    all_touched=all_touched,
        )
        samples = np.where(raster != np.nan)
        x_coords, y_coords = self.ds.odc.geobox.affine * (samples[1], samples[0])
        x_coords += abs(self.ds.odc.geobox.resolution.xy[0]) * 0.5
        y_coords -= abs(self.ds.odc.geobox.resolution.xy[1]) * 0.5

        if subset < 1:
            rand_idx = np.random.choice(
                    np.arange(0, y_coords.shape[0]),
                    size=int(y_coords.shape[0] * subset),
                    replace=False,
                )
            y_coords = y_coords[rand_idx]
            x_coords = x_coords[rand_idx]

        n_samples = y_coords.shape[0]
        
        try:
            fid_ = int(fid)
            fid_ = np.zeros(n_samples, dtype='int64') + fid_
        except ValueError:
            fid_ = str(fid)
            fid_ = np.zeros([fid_] * n_samples, dtype=object)

        crs = ds.odc.geobox.crs.crs_str
        df = gpd.GeoDataFrame(
            data=np.c_[fid_, np.arange(0, n_samples)],
            geometry=gpd.points_from_xy(x_coords, y_coords),
            crs=crs,
            columns=[attribute],
        )

        if not df.empty:
            for col in other_cols:
                fea_df = df.assign(**{col: gdf_row[col]})

        return df        

    
    def coords_to_indices(self, x, y, transform):
        col_index, row_index = ~transform * (x, y)
        return np.int64(col_index), np.int64(row_index)
 
    
    def extract_by_points(self):         
        if isinstance(self.gdf.iloc[0].geometry, (Point, MultiPoint)):
            pass
        else:
            raise TypeError (f'gdf must be of type {Point} or {MultiPoint}')
    
        x, y = self.coords_to_indices(self.gdf.geometry.x.values, self.gdf.geometry.y.values, self.ds.odc.geobox.transform)
        yidx = xr.DataArray(y, dims='z')
        xidx = xr.DataArray(x, dims='z')
        res = self.ds.isel(y=yidx, x=xidx)
        time_names = res.time.values
        time_format='%Y%m%d'
        time_names = [f't{t}' for t in  np.datetime_as_string(ds.time.values).tolist()]
        for band in self.bands:
            band_names_concat = []
            for t in time_names:
                band_names_concat.append(f'{t}_{band}')
            df = pd.concat((
             self.gdf,pd.DataFrame(
                data=res[band].values.T.squeeze(),
                columns=band_names_concat).reset_index()),axis=1,)
        return df
    
    def extract_by_polygon(self,
                    merge_alg = 'replace',
                    all_touched=False,
                    subset = 1,
    ):

        if isinstance(self.bands, list):
            pass
        else:
            raise TypeError (f'bands must be of type {str(list)}')
        
        self.gdf['polygon_id'] = self.gdf.index.values    
        dataframes = []
        gdf_columns = self.gdf.columns.tolist()

        for i in range(0, len(self.gdf.index)):
            point_df = self.sample_feature_by_shape(self.ds,
                        gdf_row = self.gdf.iloc[[i]],
                        attribute = 'polygon_id',                            
                        merge_alg = merge_alg,
                        all_touched=all_touched,
                        subset = subset,
                    )
            if not point_df.empty:
                dataframes.append(point_df)

        dataframes = pd.concat(dataframes, axis=0)
        self.gdf = dataframes.assign(point=np.arange(0, dataframes.shape[0]))
        if not self.gdf.empty:
            self.gdf.index = list(range(0, self.gdf.shape[0]))
        df = self.extract_by_points() 
        return df

def point_extract(ds, gdf, bands):
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
    if isinstance(gdf.iloc[0].geometry, (Point, MultiPoint)):
        pass
    else:
        raise TypeError (f'geometry must be of type {Point} or {MultiPoint}')
        
    return extracter(ds = ds, gdf = gdf, bands = bands).extract_by_points()
    
def polygon_extract(ds, gdf, bands, agg=None, subset = 1, merge_alg = 'replace', all_touched=False):
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
    if isinstance(gdf.iloc[0].geometry, (Polygon, MultiPolygon)):
        pass
    else:
        raise TypeError (f'geometry must be of type {Polygon} or {MultiPolygon}')
    
    if agg is not None:
        result = extracter(ds = ds, gdf = gdf, bands = bands).extract_by_polygon(merge_alg = merge_alg ,all_touched=all_touched, subset = subset )
        dic = dict()
        for column in result.columns:
            if column == 'geometry':
                dic['geometry'] = 'first'
            else:
                dic[column] = agg
        return result.groupby('polygon_id').aggregate(dic)
    else:   
        return extracter(ds = ds, gdf = gdf, bands = bands).extract_by_polygon(merge_alg = merge_alg ,all_touched=all_touched, subset = subset )
    
