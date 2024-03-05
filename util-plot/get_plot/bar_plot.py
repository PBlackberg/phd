''' 
# ----------------
#   Bar plot
# ----------------
This script plots a bar plot from a dataset with each variable corresponding to one number
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# ------------------------
#     Plot functions
# ------------------------
# ---------------------------------------------------------------------------------------- General --------------------------------------------------------------------------------------------------- #
def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def scale_ax_y(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])


# ---------------------------------------------------------------------------------------- specific --------------------------------------------------------------------------------------------------- #
def bar_highlight(variables_sorted, highlight_color, vars_highlight, nb_highlight):
    default_color = 'k'
    colors = [default_color] * len(variables_sorted) 
    if isinstance(vars_highlight, list):
        colors = [highlight_color if variable in vars_highlight else default_color for variable in variables_sorted]
    if isinstance(nb_highlight, int):
        colors = [highlight_color if i < nb_highlight else default_color for i in range(len(variables_sorted))]
    for variable in variables_sorted:
        if variable in mV.observations:
            index = variables_sorted.index(variable)
            colors[index] = 'green'
    return colors

def plot_legend(highlight_bars, label, label_highlight, color_highlight):
    patches = []
    patches.append(mpatches.Patch(color='k', label=label))
    if highlight_bars:
        patches.append(mpatches.Patch(color=color_highlight, label=label_highlight))
    plt.legend(handles=patches)

def plot_dsBar(ds, title = 'test', x_label = 'test', y_label = 'test', label = 'test', 
               highlight_bars = False, color_highlight = 'blue', vars_highlight = ['a', 'b'], nb_highlight = '', label_highlight = 'test'):
    variables, alist = [], []
    for variable in ds.data_vars:   # variable is a string
        variables.append(variable)
        alist.append(ds[variable])
    idx_sorted = np.argsort(alist)  # ascending order
    variables_sorted = [variables[i] for i in idx_sorted]
    alist_sorted = [alist[i] for i in idx_sorted]
    da = xr.DataArray(data = alist_sorted, dims = [x_label], coords={x_label: variables_sorted})
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot()
    colors = bar_highlight(variables_sorted, color_highlight, vars_highlight, nb_highlight)
    da.to_series().plot.bar(ax=ax, color = colors, rot=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(y_label)
    scale_ax_y(ax, scaleby=0.8)
    move_row(ax, moveby = 0.1)
    move_row(ax, moveby = 0.065)
    plot_legend(highlight_bars, label, label_highlight, color_highlight)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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
    ds = xr.Dataset()
    for dataset in mV.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, mV.experiments[0]).mean(dim = 'time')
        ds[dataset] = da
    # print(ds)


# --------------------------------------------------------------------------------------- test plot --------------------------------------------------------------------------------------------------- #
    switch = {
        'delete_previous_plots':    True,
        'basic':                    True,
        'subset highlighted':       False,
        'number_highligted':        False
        }

    mFp.remove_test_plots() if switch['delete_previous_plots'] else None
    # exit()

    if switch['basic']:
        filename = 'basic.png'
        fig, ax = plot_dsBar(ds, title = 'test', x_label = 'test', y_label = 'test', label = 'test', 
               highlight_bars = False, color_highlight = 'blue', vars_highlight = ['a', 'b'], nb_highlight = '', label_highlight = 'test')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch['subset highlighted']:
        filename = 'models_highligted.png'
        models_highlight = [    
            'INM-CM5-0',         # 1
            'IITM-ESM',          # 2
            'FGOALS-g3',         # 3    
            'INM-CM4-8',         # 4        
            ]
        fig, ax = plot_dsBar(ds, title = 'test', x_label = 'test', y_label = 'test', label = 'test', 
               highlight_bars = True, color_highlight = 'blue', vars_highlight = models_highlight, nb_highlight = '', label_highlight = 'test1')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch['number_highligted']:
        filename = 'number_highligted.png'
        nb_highlight = 10
        fig, ax = plot_dsBar(ds, title = 'test', x_label = 'test', y_label = 'test', label = 'test', 
               highlight_bars = True, color_highlight = 'blue', vars_highlight = '', nb_highlight = nb_highlight, label_highlight = 'test1')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)















