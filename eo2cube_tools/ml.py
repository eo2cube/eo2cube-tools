import geopandas as gpd
import xarray as xr
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.base import is_classifier, is_regressor
import rasterio
import geopandas as gpd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

def rasterize(gdf,da,attribute,):    
    crs = da.geobox.crs
    transform = da.geobox.transform
    dims = da.geobox.dims
    xy_coords = [da[dims[0]], da[dims[1]]]
    y, x = da.geobox.shape
    gdf_reproj = gdf.to_crs(crs=crs)
    shapes = zip(gdf_reproj.geometry, gdf_reproj[attribute])
    arr = rasterio.features.rasterize(shapes=shapes,out_shape=(y, x),transform=transform)
    xarr = xr.DataArray(arr,coords=xy_coords,dims=dims,attrs=da.attrs,)               
    return xarr

def extract_samples(ds, gdf, attribute):
    mask = rasterize(gdf, ds, attribute=attribute)
    return mask

def get_features(ds, feature_dims=[]):
    data_dims = get_dimensions(ds, feature_dims=feature_dims)
    features = tuple(feature_dims) + ('variable',)
    if isinstance(ds, xr.Dataset):
        variables = get_vars_for_dims(ds, data_dims)
        data = ds[variables].to_array()
    else:
        data = ds.expand_dims('variable')

    data = data.stack(feature=features).transpose(
        *data_dims, 'feature', transpose_coords=True).values
    return data.reshape((-1, data.shape[-1]))

def get_dimensions(ds, feature_dims=[]):
    data_dims = tuple([d for d in ds.coords if d in ds.dims
                       and d not in feature_dims])
    return data_dims

def get_shape(ds, feature_dims=[]):
    data_dims = get_dimensions(ds, feature_dims=feature_dims)
    shape = tuple([ds.sizes[d] for d in data_dims])
    return shape

def broadcast_array(arr, shape):
    matching = list(shape)
    new_shape = [1] * len(shape)
    for dim in arr.shape:
        i = matching.index(dim)
        new_shape[i] = dim
        matching[i] = None
    return np.broadcast_to(arr.reshape(new_shape), shape)

def broadcast_labels(labels, ds, feature_dims=[]):
    shape = get_shape(ds, feature_dims=feature_dims)
    if isinstance(labels, np.ndarray):
        return broadcast_array(labels, shape)

    elif isinstance(labels, xr.DataArray):
        data_dims = get_dimensions(ds, feature_dims=feature_dims)
        bc_dims = set(data_dims) - set(labels.dims) - \
            set(feature_dims)
        for dim in bc_dims:
            labels = xr.concat([labels] * ds.sizes[dim], dim=dim)
            labels.coords[dim] = ds.coords[dim]
        labels = labels.transpose(*data_dims, transpose_coords=True)
        return labels
    
def get_vars_for_dims(ds, dims, invert=False):
    return [v for v in ds.data_vars
            if set(ds[v].dims).issuperset(set(dims)) != invert]


class ML:
    def __init__(self, model, feature_dims=[], scale=False, train_size=0.8, stratify=None):
        self.clf = model
        self.scale = scale
        self._scaler = None
        self.train_size = train_size
        self.stratify = stratify
        self.feature_dims = feature_dims

    def extract_Xy(self, ds, labels=None, attribute=None):
        if labels is not None:
            if 'GeoDataFrame' in str(type(labels)):
                if attribute is not None:
                    labels = extract_samples(ds, labels, attribute)
                else:
                    print('Please provide the name of attribute column for rasterize')
        if isinstance(labels, xr.Dataset):
            raise ValueError("`labels` should be an xarray.DataArray or "
                             "numpy array")
        elif isinstance(labels, (xr.DataArray, np.ndarray)):
            labels = labels.squeeze()

        labels = broadcast_labels(
            labels, ds, feature_dims=self.feature_dims)
        if labels is not None:
            ymask = ~np.isnan(np.array(labels))
            np.greater(labels, 0, out=ymask, where=ymask)
            ymask = ymask.reshape(-1)
        else:
            ymask = slice(None)
        
        X = get_features(ds)[ymask]
        Xmask = ~np.isnan(X).any(axis=1)
        X = X[Xmask]
        
        if labels is not None:
            y = np.array(labels).reshape(-1)[ymask][Xmask]
        else:
            y = None
        if self.scale:
            self._scaler = preprocessing.StandardScaler()
            self._scaler.fit(X)
            X = self._scaler.transform(X) 
        return X,y
    
    def split(self, X, y):
        X, X_test, y, y_test = train_test_split(X, y, train_size=self.train_size, stratify=self.stratify)
        return X, X_test, y, y_test
    
    def train_supervised(self, ds, labels=None, attribute=None):
        X,y = self.extract_Xy(ds, labels=labels, attribute=attribute)
        if self.train_size != None:
            self.X, self.X_test, self.y, self.y_test = self.split(X, y)
        self.model = self.clf.fit(self.X, self.y)
        
    def train_unsupervised(self, ds, labels=None, attribute=None):
        self.model = self.clf.fit(self.X)

    def predict(self, ds):
        X = get_features(ds, feature_dims=self.feature_dims)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        result = self.model.predict(X)
        data_dims = get_dimensions(ds, feature_dims=self.feature_dims)
        data_shape = get_shape(ds, feature_dims=self.feature_dims)
        data_coords = OrderedDict(
            (dim, c) for dim, c in ds.coords.items() if dim in data_dims
        )

        labels_flat = np.empty(mask.shape + result.shape[1:]) * np.nan
        labels_flat[mask] = result
        labels_data = labels_flat.reshape(data_shape + result.shape[1:])
        labels = xr.DataArray(labels_data,
                              dims=data_dims, coords=data_coords)
        return labels
        
class Regression(ML):
    def __init__(self, model, feature_dims=[], scale=False, train_size=0.8, stratify=None):
        super().__init__(model, feature_dims, scale, train_size, stratify)
        
    def extract_Xy(self, ds, labels=None, attribute=None):
        return super().extract_Xy(ds, labels, attribute)
        
    def train(self, ds, labels=None, attribute=None):
         return super().train_supervised(ds, labels, attribute)
        
    def predict(self, ds):
        return super().predict(ds)
    
    def model_performance(self):
        mp = self.model.predict(self.X_test)
        return mp
        
    def get_metrics(self):
        mp = self.model_performance()
        self.r2 = metrics.r2_score(self.y_test, mp)
        self.explained_variance = metrics.explained_variance_score(self.y_test, mp)
        self.max_error = metrics.max_error(self.y_test, mp) 
        self.neg_mean_absolute_error = metrics.mean_absolute_error(self.y_test, mp)
        self.neg_mean_squared_error = metrics.mean_squared_error(self.y_test, mp)
        self.neg_root_mean_squared_error = metrics.mean_squared_error(self.y_test, mp)
        self.neg_mean_squared_log_error = metrics.mean_squared_log_error(self.y_test, mp)
        self.neg_median_absolute_error = metrics.median_absolute_error(self.y_test, mp)
        
    def model_metrics(self):
        self.get_metrics()
        print(f'R2 : {self.r2}')
        print(f'explained_variance : {self.explained_variance}')
        print(f'max_error : {self.max_error}')
        print(f'mean_absolute_error : {self.neg_mean_absolute_error}')
        print(f'mean_squared_error : {self.neg_mean_squared_error}')
        print(f'root_mean_squared_error : {self.neg_root_mean_squared_error}')
        print(f'mean_squared_log_error : {self.neg_mean_squared_log_error}')
        print(f'median_absolute_error : {self.neg_median_absolute_error}')
