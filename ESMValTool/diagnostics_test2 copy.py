
from pathlib import Path
import logging
import xarray as xr
from esmvaltool.diag_scripts.shared import (
    run_diagnostic
)
logger = logging.getLogger(Path(__file__).stem)



import os
import sys

logger.info(f'importing script from: {os.getcwd()}')
path = '/home/565/cb4968/Documents/code/phd/util-plot/get_plot'
sys.path.insert(0, path)
import map_plot as mP


def calc_diag(da):
    ''' Most extreme daily gridpoint value locally over set time period (1-year here) '''
    # nb_days = '1'
    # if int(nb_days) > 1:
    #     da = da.resample(time=f'{nb_days}D').mean(dim='time')
    # da = da.resample(time='Y').max(dim='time')

    da = da.isel(time = 0)
    return da

def plot_diag(ds, label, vmin, vmax, cmap, title):
    fig, ax = mP.plot_dsScenes(ds, label = label, title = title, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
    return fig, ax

def main(cfg):
    logger.info(cfg)
    input_data = cfg['input_data'].values()

    # get data
    logger.info(input_data)
    for dataset in input_data:
        input_file = dataset['filename']
        short_name = dataset['short_name']
        logger.info(f'running variable: \n \n {short_name}')
        data = xr.open_dataset(input_file)[short_name]
        logger.info(f'plotting variable: {short_name}')
        logger.info(f'{data}')

        # calculate metric        
        diag_data = calc_diag(data)
        logger.info(f'{diag_data}')
        ds = xr.Dataset({'data': diag_data})
        label = '' 
        vmin = 0
        vmax = 100
        cmap = 'Blues'
        title = ''
        logger.info(f'{diag_data}')
        
        # plot
        fig, ax = plot_diag(ds, label, vmin, vmax, cmap, title)

        # save
        plot_dir = cfg['plot_dir']
        filename = 'test' + '.png'
        output_path = Path(plot_dir) / filename
        fig.savefig(output_path)
        logger.info(f'Saved figure in {plot_dir}/{filename}')


if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)







































