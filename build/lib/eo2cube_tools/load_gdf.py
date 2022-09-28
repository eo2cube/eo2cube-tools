import geopandas as gpd
import psycopg2


def load_gdf(filename, cols="all"):
    """
    Loads a vectordataset from the PostGIS database and stores it as a gpd.Geodataframe.

    Description
    ----------
    Connects to the PostGIS database and loads a vectordataset from there. The vectordataset must be imported to the PostGIS
    database in advance (this can be done by a user of the Departement of Remote Sensing of the University Würzburg).
    The function calls the file by its filename in the PostGIS database. It´s possible to define which columns should be
    included. The default alternative is to load all existing columns.

    Parameters
    ----------
    filename: string
        The name of the vectordataset in the PostGIS database. If unsure, ask the staff member of the Department of Remote
        Sensing which imported your dataset to the PostGIS database.

    cols: list,string (default = "all")
        List with column names of the vectordataset, e.g (["column1", "column2", "column3"]), which should be loaded. If
        used, only the "real" existing columns need to be defined. The "geom" column and the "gid" column (indexing of
        the PostGIS database) are always loaded. The default is set to "all" which loads all columns of the vectordataset.

    Returns
    -------
    gdf: gpd.GeoDataframe
        Geodataframe from PostGIS database with the defined or all columns.

    """

    # PostGIS credentials
    user = "dc_user"
    password = "localuser1234"
    host = "127.0.0.1"
    port = "5432"
    database = "datacube"

    connection = psycopg2.connect(
        database=database, user=user, password=password, host=host
    )  # connection to PostGIS
    if type(cols) is list:
        sql = (
            "select geom, gid, " + ", ".join(cols) + " from " + filename + ";"
        )  # define which attributes of shapefile should be included
        gdf = gpd.GeoDataFrame.from_postgis(
            sql, connection
        )  # creates GeoDataFrame from the sql table
    elif cols == "all":
        sql = (
            "select * from " + filename + ";"
        )  # define which attributes of shapefile should be included
        gdf = gpd.GeoDataFrame.from_postgis(
            sql, connection
        )  # creates GeoDataFrame from the sql table

    connection.close()

    return gdf
