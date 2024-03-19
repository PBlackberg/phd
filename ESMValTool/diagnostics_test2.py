
from pathlib import Path
import logging
import xarray as xr
from pprint import pformat
from esmvaltool.diag_scripts.shared import (
    group_metadata,
    run_diagnostic,
    save_data,
    save_figure,
    select_metadata,
    sorted_metadata,
)
logger = logging.getLogger(Path(__file__).stem)

import os
import sys
path = '/home/565/cb4968/Documents/code/phd/util-plot/get_plot'
sys.path.insert(0, path)
import map_plot as mP


def get_provenance_record(attributes, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""
    # Associated recipe uses contains a caption string with placeholders
    # like {long_name} that are now populated from attributes dictionary.
    # Note that for simple recipes, caption can be set here as a simple string
    caption = attributes['caption'].format(**attributes)

    record = {
        'caption': caption,
        'statistics': ['mean'],
        'domains': ['global'],
        'plot_types': ['zonal'],
        'authors': [
            'andela_bouwe',
            'righi_mattia',
        ],
        'references': [
            'acknow_project',
        ],
        'ancestors': ancestor_files,
    }
    return record

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
    input_data = cfg['input_data'].values() # logger.info(cfg), logger.info(input_data)
    groups = group_metadata(input_data, 'variable_group', sort='dataset')
    
    # get data
    for group_name in groups:
        logger.info("Processing variable %s", group_name)
        for attributes in groups[group_name]:
            logger.info("Processing dataset %s", attributes['dataset'])
            input_file = attributes['filename']
            
            data = xr.open_dataset(input_file)['pr'].sel(lat = slice(-30, 30))
            logger.info(f'{data}')

            # calculate metric        
            diag_data = calc_diag(data)
            logger.info(f'{diag_data}')
            ds = xr.Dataset({'data': diag_data})
            label = '' 
            vmin = 0
            vmax = 25
            cmap = 'Blues'
            title = ''
            logger.info(f'{diag_data}')

            # create output file
            output_basename = Path(input_file).stem
            if group_name != attributes['short_name']:
                output_basename = group_name + '_' + output_basename
            if "caption" not in attributes:
                attributes['caption'] = input_file
            provenance_record = get_provenance_record(attributes, ancestor_files=[input_file])

            # plot
            fig, ax = plot_diag(ds, label, vmin, vmax, cmap, title)

            # save
            plot_dir = cfg['plot_dir']
            filename = 'test' + '.png'
            output_path = Path(plot_dir) / filename
            fig.savefig(output_path)
            logger.info(f'Saved figure in {plot_dir}/{filename}')


    # Demonstrate use of metadata access convenience functions.


    grouped_input_data = group_metadata(input_data,
                                        'variable_group',
                                        sort='dataset')
    logger.info(
        "Example of how to group and sort input data by variable groups from "
        "the recipe:\n%s", pformat(grouped_input_data))

    # Example of how to loop over variables/datasets in alphabetical order
    groups = group_metadata(input_data, 'variable_group', sort='dataset')
    for group_name in groups:
        logger.info("Processing variable %s", group_name)
        for attributes in groups[group_name]:
            logger.info("Processing dataset %s", attributes['dataset'])
            input_file = attributes['filename']
            cube = compute_diagnostic(input_file)

            output_basename = Path(input_file).stem
            if group_name != attributes['short_name']:
                output_basename = group_name + '_' + output_basename
            if "caption" not in attributes:
                attributes['caption'] = input_file
            provenance_record = get_provenance_record(
                attributes, ancestor_files=[input_file])
            plot_diagnostic(cube, output_basename, provenance_record, cfg)



if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)







































