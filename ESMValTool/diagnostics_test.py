#!/bin/python

from esmvaltool.diag_scripts.shared import run_diagnostic
from esmvaltool.diag_scripts.shared._base import get_plot_filename

import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

import fiona
import cartopy.crs as ccrs
from cartopy.io import shapereader
from shapely.geometry import Polygon


def rect_from_bound(xmin, xmax, ymin, ymax):
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]

def get_country_polygon(country):
    resolution = '10m'
    category = 'cultural'
    name = 'admin_0_countries'
    shpfilename = shapereader.natural_earth(resolution, category, name)
    f = fiona.open(shpfilename)
    reader = shapereader.Reader(shpfilename)
    records = list(reader.records())
    for record in records:
        if record.attributes['ADMIN'] == 'Australia':
            return [record.geometry]

def plot_australia(data, name, cmap):
    poly = get_country_polygon('Australia')
    fig = plt.figure(figsize = (11.8, 8.5))
    ax = plt.axes(projection = ccrs.PlateCarree())
    ax.add_geometries(poly, crs = ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')
    pad1 =0.1
    exts = [poly[0].bounds[0] - pad1, poly[0].bounds[2] + pad1, poly[0].bounds[1]-pad1, poly[0].bounds[3]+pad1]
    exts = [110, 165, -45, -9]
    ax.set_extent(exts, crs = ccrs.PlateCarree())
    min_lon, max_lon, min_lat, max_lat = exts
    pad2 = 1
    data = data.sel(lat = slice(min_lat - pad2, max_lat + pad2), lon = slice(min_lon - pad2, max_lon + pad2))
    data = data.mean('time')
    msk = Polygon(rect_from_bound(*exts)).difference(poly[0].simplify(0.01))
    msk_stm = ccrs.PlateCarree().project_geometry (msk, ccrs.PlateCarree())
    cs = ax.contourf(data['lon'], data['lat'], data['tas'], transform = ccrs.PlateCarree(), cmap = 'coolwarm', extend = 'both')
    ax.add_geometries(msk_stm, ccrs.PlateCarree(), zorder=12, facecolor = 'white', edgecolor = 'none', alpha = 1.0)
    cbar = plt.colorbar(cs, shrink = 0.7, orientation = 'vertical', label = 'Surface air temperature (K)')
    plt.title('Australia Temperature')
    return fig


def main(cfg):
    input_data = cfg['input_data'].values()

    for dataset in input_data:
        # load data
        input_file = dataset['filename']
        name = dataset['variable_group']
        data = xr.open_dataset(input_file)
        cmap = cfg['colormap']

        fig = plot_australia(data, name, cmap)

        # save output
        output_file = Path(input_file).stem.replace('tas', name)
        output_path = get_plot_filename(output_file, cfg)
        fig.savefig(output_path)



if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)




















