"""
# Adapted from:

Copyright 2016 United States Government as represented by the Administrator
of the National Aeronautics and Space Administration. All Rights Reserved.

Portion of this code is Copyright Geoscience Australia, Licensed under the
Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License
at
   http://www.apache.org/licenses/LICENSE-2.0

The CEOS 2 platform is licensed under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

# Use the sen2cor scene classifiaction band (scl) for creating masks using
different classification types
"""


def unpack_s2bits(s2_scl_endcoding, data_array, cover_type):
    """
    Description:
        Unpack bits for mask boolean mask
    -----
    Input:
        land_cover_encoding(dict hash table) land cover endcoding provided by sen2cor scl
        data_array( xarray DataArray)
        cover_type(String) type of cover
    Output:
        unpacked DataArray

    """
    boolean_mask = np.isin(data_array.values, s2_scl_endcoding[cover_type])
    return xr.DataArray(
        boolean_mask.astype(bool),
        coords=data_array.coords,
        dims=data_array.dims,
        name=cover_type + "_mask",
        attrs=data_array.attrs,
    )


def s2_unpack_scl(data_array, cover_type):
    s2_scl_endcoding = dict(
        no_data=[0],
        saturated_or_defective=[1],
        dark_area_pixels=[2],
        cloud_shadows=[3],
        vegetation=[4],
        not_vegetated=[5],
        water=[6],
        unclassified=[7],
        cloud_medium_probability=[8],
        cloud_high_probability=[9],
        thin_cirrus=[10],
        snow=[11],
        cloud=[3, 8, 9, 10],
        cloud_free=[4, 5, 6],
    )
    return unpack_s2bits(s2_scl_endcoding, data_array, cover_type)
