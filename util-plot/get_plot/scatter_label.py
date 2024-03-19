''' 
# -----------------------------
#   Scatter plot (time-mean)
# -----------------------------
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from scipy import stats


# ------------------------
#     Plot functions
# ------------------------
# ---------------------------------------------------------------------------------------- General --------------------------------------------------------------------------------------------------- #
def create_figure(width = 12, height = 4, nrows = 1, ncols = 1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width,height))
    return fig, axes

def move_col(ax, moveby):
    ax_position = ax.get_position()
    _, bottom, width, height = ax_position.bounds
    new_left = _ + moveby
    ax.set_position([new_left, bottom, width, height])

def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def scale_ax_x(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2
    ax.set_position([left, bottom, new_width, new_height])

def scale_ax_y(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def plot_xlabel(fig, ax, xlabel, pad, fontsize):
    ax_position = ax.get_position()
    lon_text_x =  ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2
    lon_text_y =  ax_position.y0 - pad
    ax.text(lon_text_x, lon_text_y, xlabel, ha = 'center', fontsize = fontsize, transform=fig.transFigure)

def plot_ylabel(fig, ax, ylabel, pad, fontsize):
    ax_position = ax.get_position()
    lat_text_x = ax_position.x0 - pad
    lat_text_y = ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2
    ax.text(lat_text_x, lat_text_y, ylabel, va = 'center', rotation='vertical', fontsize = fontsize, transform=fig.transFigure)

def plot_axtitle(fig, ax, title, xpad, ypad, fontsize):
    ax_position = ax.get_position()
    title_text_x = ax_position.x0 + xpad 
    title_text_y = ax_position.y1 + ypad
    ax.text(title_text_x, title_text_y, title, fontsize = fontsize, transform=fig.transFigure)

def add_correlation_coeff(x, y, ax, color):
    res= stats.pearsonr(x,y) #if not mF.find_ifWithObs(mV.datasets) else stats.pearsonr(x[0:-len(mV.observations)],y[0:-len(mV.observations)])
    print('r: ', res[0])
    print('p-value:', res[1])
    if not color == 'k':
        # if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', 
                    xytext=(0.86, 0.9), textcoords='axes fraction', fontsize = 12, color = color)
    else:
        # if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', 
                    xytext=(0.86, 0.95), textcoords='axes fraction', fontsize = 12, color = color)
        
        
# ---------------------------------------------------------------------------------------- plot figure ----------------------------------------------------------------------------------------------------- #
def format_ax(x_label, y_label, fig, ax, xmin, xmax, ymin, ymax, x, y, fig_title):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    move_col(ax, 0)
    move_row(ax, 0.03)
    scale_ax_x(ax, 1)
    scale_ax_y(ax, 1)
    plot_xlabel(fig, ax, x_label, pad=0.1, fontsize = 12)
    plot_ylabel(fig, ax, y_label, pad = 0.075, fontsize = 12)
    plot_axtitle(fig, ax, fig_title, xpad = 0.15, ypad = 0.0075, fontsize = 12)
    plt.axhline(y=0, color='k', linestyle='--') if np.min(y)<0 and np.max(y)>0 else None    # add dashed line if datapoints cross zero 
    plt.axvline(x=0, color='k', linestyle='--') if np.min(x)<0 and np.max(x)>0 else None    # add dashed line if datapoints cross zero 
    
def plot_letters(ax, x, y, variable_list, color, models_highlight=[''], color_highlight = ''):
    legend_handles, legend_labels = [], []
    for variable in variable_list:
        dataset_idx = variable_list.index(variable)
        label_text = variable[0:2]
        label_text = f'{label_text}-L'  if variable[-2] == 'L'                                      else label_text 
        label_text = f'{label_text}-L'  if variable[-1] == 'L'                                      else label_text 
        label_text = f'{label_text}-E'  if variable in ['CNRM-ESM2-1', 'CMCC-ESM2', 'GFDL-ESM4']    else label_text 
        label_text = f'{label_text}-H'  if variable[-2] == 'H'                                      else label_text
        text_color = color_highlight    if variable in models_highlight                             else color
        # text_color = 'g'                if variable in mV.observations                              else text_color
        ax.text(x[dataset_idx], y[dataset_idx], label_text, color=text_color, ha='center', va='center')
        legend_handles.append(Patch(facecolor=text_color, edgecolor='none'))
        legend_labels.append(f"{label_text} - {variable}")
    ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.1, 0.475), loc='center left')

def plot_dsScatter(ds_x, ds_y, variable_list = ['a', 'b'], fig_title = '', x_label = '', y_label = '', 
                    xmin = None, xmax = None, ymin = None, ymax = None,
                    fig_given = False, fig = '', ax = '', 
                    color = 'k', models_highlight = [''], color_highlight = 'b', 
                    add_correlation = True, put_point = True, observations = ['a', 'b']):
    if not fig_given:
        fig, ax = create_figure(width = 11, height = 6.5)
        scale_ax_x(ax, scaleby=0.75)
    x, y = [], []
    for variable in variable_list:
        x_point, y_point = ds_x[variable].data, ds_y[variable].data
        x.append(x_point)
        y.append(y_point)    
    add_correlation_coeff(x, y, ax, color)  if add_correlation else None
    ax.scatter(x, y, alpha=0)                                                                   # Completely transparent points 
    if put_point:
        plot_letters(ax, x, y, variable_list, color, models_highlight, color_highlight)           # latters at points
    format_ax(x_label, y_label, fig, ax, xmin, xmax, ymin, ymax, x, y, fig_title) if not fig_given else None
    return fig, ax



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
    import os
    import sys
    home = os.path.expanduser("~")
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myVars               as mV
    import myFuncs_plots        as mFp
    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import get_data.metric_data as mD


# --------------------------------------------------------------------------------------- get data --------------------------------------------------------------------------------------------------- #
    metric_type = 'conv_org'
    metric_name = f'rome_{mV.conv_percentiles[0]}thprctile'
    ds_x = xr.Dataset()
    for dataset in mV.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, mV.experiments[0]).mean(dim = 'time')
        ds_x[dataset] = da
    # print(ds_x)

    metric_type = 'pr'
    metric_name = f'pr_sMean'
    ds_y = xr.Dataset()
    for dataset in mV.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, mV.experiments[0]).mean(dim = 'time')
        ds_y[dataset] = da
    # print(ds_y)
        

# --------------------------------------------------------------------------------------- test plot --------------------------------------------------------------------------------------------------- #
    switch = {
        'delete_previous_plots':    True,
        'point_and_correlation':    False,
        'points_only':              False,
        'points_only_highlight':    False,
        'correlation_only':         False,
        }

    mFp.remove_test_plots() if switch['delete_previous_plots'] else None
    # exit()

    if switch['point_and_correlation']:
        filename = 'point_and_correlation.png'
        if set(mV.datasets) & set(mV.observations):
            ds_x_test = ds_x.drop_vars(mV.observations[0])
            ds_y_test = ds_y.drop_vars(mV.observations[0])
        fig, ax = plot_dsScatter(ds_x_test, ds_y_test, variable_list = list(ds_x_test.data_vars.keys()), fig_title = 'test', x_label = 'test', y_label = 'test', 
                    xmin = None, xmax = None, ymin = None, ymax = None,
                    fig_given = False, fig = '', ax = '', 
                    color = 'k', models_highlight = [''], color_highlight = 'b', 
                    add_correlation = True, put_point = True)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch['points_only']:
        filename = 'points_only.png'
        fig, ax = plot_dsScatter(ds_x, ds_y, variable_list = list(ds_x.data_vars.keys()), fig_title = 'test', x_label = 'test', y_label = 'test', 
                    xmin = None, xmax = None, ymin = None, ymax = None,
                    fig_given = False, fig = '', ax = '', 
                    color = 'k', models_highlight = [''], color_highlight = 'b', 
                    add_correlation = False, put_point = True)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch['points_only_highlight']:
        filename = 'points_only_highlight.png'
        models_highlight = [    
            'INM-CM5-0',         # 1
            'IITM-ESM',          # 2
            'FGOALS-g3',         # 3    
            'INM-CM4-8',         # 4       
            ]
        fig, ax = plot_dsScatter(ds_x, ds_y, variable_list = list(ds_x.data_vars.keys()), fig_title = 'test', x_label = 'test', y_label = 'test', 
                    xmin = None, xmax = None, ymin = None, ymax = None,
                    fig_given = False, fig = '', ax = '', 
                    color = 'k', models_highlight = models_highlight, color_highlight = 'b', 
                    add_correlation = False, put_point = True)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch['correlation_only']:
        filename = 'correlation_only.png'
        models_highlight = [    
            'INM-CM5-0',         # 1
            'IITM-ESM',          # 2
            'FGOALS-g3',         # 3    
            'INM-CM4-8',         # 4       
            ]
        fig, ax = plot_dsScatter(ds_x, ds_y, variable_list = list(ds_x.data_vars.keys()), fig_title = 'test', x_label = 'test', y_label = 'test', 
                    xmin = None, xmax = None, ymin = None, ymax = None,
                    fig_given = False, fig = '', ax = '', 
                    color = 'k', models_highlight = models_highlight, color_highlight = 'b', 
                    add_correlation = False, put_point = True)
        
        fig, ax = plot_dsScatter(ds_x, ds_y, variable_list = mV.models_cmip6, fig_title = 'test', x_label = 'test', y_label = 'test', 
                    xmin = None, xmax = None, ymin = None, ymax = None,
                    fig_given = True, fig = fig, ax = ax, 
                    color = 'k', models_highlight = models_highlight, color_highlight = 'b', 
                    add_correlation = True, put_point = False)

        fig, ax = plot_dsScatter(ds_x, ds_y, variable_list = models_highlight, fig_title = 'test', x_label = 'test', y_label = 'test', 
                    xmin = None, xmax = None, ymin = None, ymax = None,
                    fig_given = True, fig = fig, ax = ax, 
                    color = 'b', models_highlight = models_highlight, color_highlight = 'b', 
                    add_correlation = True, put_point = False)
        
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)





