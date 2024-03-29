import geopandas as gpd
import xarray as xr
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import rasterio
import rasterio.features as features
import geopandas as gpd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import itertools


def extract_Xy(ds, feature_dims=[], labels=None, attribute=None):
    if labels is not None:
        if "GeoDataFrame" in str(type(labels)):
            if attribute is not None:
                labels = extract_samples(ds, labels, attribute)
            else:
                print("Please provide the name of attribute column for rasterize")
    if isinstance(labels, xr.Dataset):
        raise ValueError("`labels` should be an xarray.DataArray or " "numpy array")
    elif isinstance(labels, (xr.DataArray, np.ndarray)):
        labels = labels.squeeze()

    labels = broadcast_labels(labels, ds, feature_dims)
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

    return X, y


def rasterize(
    gdf,
    da,
    attribute,
):
    if hasattr(da, "geobox"):
        crs = da.geobox.crs
        transform = da.geobox.transform
        dims = da.geobox.dims
        xy_coords = [da[dims[0]], da[dims[1]]]
        y, x = da.geobox.shape
    else:
        crs = da.odc.geobox.crs
        transform = da.odc.geobox.transform
        dims = da.odc.geobox.dims
        xy_coords = [da[dims[0]], da[dims[1]]]
        y, x = da.odc.geobox.shape
    gdf_reproj = gdf.to_crs(crs=crs)
    shapes = zip(gdf_reproj.geometry, gdf_reproj[attribute])
    arr = features.rasterize(
        shapes=shapes, out_shape=(y, x), transform=transform
    )
    xarr = xr.DataArray(
        arr,
        coords=xy_coords,
        dims=dims,
        attrs=da.attrs,
    )
    return xarr


def extract_samples(ds, gdf, attribute):
    mask = rasterize(gdf, ds, attribute=attribute)
    return mask


def get_features(ds, feature_dims=[]):
    data_dims = get_dimensions(ds, feature_dims=feature_dims)
    features = tuple(feature_dims) + ("variable",)
    if isinstance(ds, xr.Dataset):
        variables = get_vars_for_dims(ds, data_dims)
        data = ds[variables].to_array()
    else:
        data = ds.expand_dims("variable")

    data = (
        data.stack(feature=features)
        .transpose(*data_dims, "feature", transpose_coords=True)
        .values
    )
    return data.reshape((-1, data.shape[-1]))


def get_dimensions(ds, feature_dims=[]):
    data_dims = tuple([d for d in ds.coords if d in ds.dims and d not in feature_dims])
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
        bc_dims = set(data_dims) - set(labels.dims) - set(feature_dims)
        for dim in bc_dims:
            labels = xr.concat([labels] * ds.sizes[dim], dim=dim)
            labels.coords[dim] = ds.coords[dim]
        labels = labels.transpose(*data_dims, transpose_coords=True)
        return labels


def get_vars_for_dims(ds, dims, invert=False):
    return [v for v in ds.data_vars if set(ds[v].dims).issuperset(set(dims)) != invert]


class ML:
    def __init__(
        self,
        pipeline,
        feature_dims=[],
        train_size=0.8,
        stratify=None,
        drop_bands=False,
        name="result",
        to_xarray=False,
    ):
        self.clf = pipeline
        self.train_size = train_size
        self.stratify = stratify
        self.feature_dims = feature_dims
        self.drop_bands = drop_bands
        self.name = name
        self.to_xarray = to_xarray

    def split(self, X, y):
        X, X_test, y, y_test = train_test_split(
            X, y, train_size=self.train_size, stratify=self.stratify
        )
        return X, X_test, y, y_test

    def model_performance(self):
        mp = self.model.predict(self.X_test)
        return mp

    def train_supervised(self, ds, labels=None, attribute=None):
        X, y = extract_Xy(ds, feature_dims=[], labels=labels, attribute=attribute)
        self.pipe = pipeline = Pipeline(steps=self.clf)
        if self.train_size != None:
            self.X_train, self.X_test, self.y_train, self.y_test = self.split(X, y)
        else:
            self.X_train = X
            self.y_train = y
        self.model = self.pipe.fit(self.X_train, self.y_train)
        self.mp = self.model_performance()

    def train_unsupervised(self, ds):
        X, y = extract_Xy(ds)
        self.pipe = Pipeline(steps=self.clf)
        self.model = self.pipe.fit(X)

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
        labels = xr.DataArray(labels_data, dims=data_dims, coords=data_coords)
        if self.to_xarray:
            return self.to_xr(ds, labels, self.drop_bands, self.name)
        else:
            return labels

    def metrics(self, metrics):
        for metric in metrics:
            return metric(self.y_test, self.mp)

    def print_metrics(self, metrics):
        for metric in metrics:
            print(f"{metric.__name__} : {metric(self.y_test, self.mp)}")

    def to_xr(self, ds, labels, drop_bands, name):
        ds[name] = labels
        if drop_bands:
            ds = ds[["result"]]
        return ds


class Regression(ML):
    def __init__(
        self,
        pipeline,
        feature_dims=[],
        train_size=0.8,
        stratify=None,
        drop_bands=False,
        name="result",
        to_xarray=False,
    ):
        super().__init__(
            pipeline, feature_dims, train_size, stratify, drop_bands, name, to_xarray
        )

    def train(self, ds, labels=None, attribute=None):
        return super().train_supervised(ds, labels, attribute)


class Classification(ML):
    def __init__(
        self,
        pipeline,
        feature_dims=[],
        train_size=0.8,
        stratify=None,
        drop_bands=False,
        name="result",
        to_xarray=False,
    ):
        super().__init__(
            pipeline, feature_dims, train_size, stratify, drop_bands, name, to_xarray
        )

    def train_supervised(self, ds, labels=None, attribute=None):
        return super().train_supervised(ds, labels, attribute)
    
    def train_unsupervised(self, ds):
        return super().train_unsupervised(ds)

    def get_test_class(self):
        return np.unique(self.y_test).tolist()

    def confusion_matrix(
        self, classes=None, title="Confusion matrix", cmap=plt.cm.Blues
    ):
        cm = self.metrics(metrics=[metrics.confusion_matrix])
        if classes == None:
            classes = self.get_test_class()
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title, fontsize=30)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
        plt.yticks(tick_marks, classes, fontsize=22)

        fmt = ".2f"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True label", fontsize=25)
        plt.xlabel("Predicted label", fontsize=25)
