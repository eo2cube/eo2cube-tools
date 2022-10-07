from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["xarray>=0.18", "rasterio>=1.2", "pandas>=1.2", "geopandas>=0.9","folium>=0.12.1", "dask>=2021 ", "holoviews>=1.14","datashader>=0.12","bokeh>=2.2","panel>=0.11", "Rbeast"]

setup(
    name="eo2cube_tools",
    version="0.0.1",
    author="Steven Hill",
    author_email="steven.hill@uni-wuerzburg.de",
    description="A package with usefull tools for the eo2cube datacubes",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/eo2cube/eo2cube-tools",
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=dependency_links,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
