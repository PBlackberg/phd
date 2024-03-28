''' 
# ----------------
#   show_plots
# ----------------
This script shows plots in zome_plots
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import os
import matplotlib.pyplot as plt
import subprocess



# -----------------------
#  Show plots in folder
# -----------------------
def open_in_tab(figures):
    for figure in figures:
        subprocess.run(["code", "-r", figure])

def open_in_new_window(figures):
    subprocess.run(["code", "-n", figures[0]])  # open first figure in new window
    if len(figures)>1:
        figures_rest = figures[1:][::-1] 
        for figure in figures_rest:
            subprocess.run(["code", "-r", figure])  # reuse the same window for the rest of the plots

def open_with_default_app(figures):
    for figure in figures:
        subprocess.run(["open", figure])

def open_in_preview(figures):
    command = ["open", "-a", "Preview"] + [f'"{figure}"' for figure in figures]
    command_string = ' '.join(command)
    subprocess.run(command_string, shell=True)

def show_plot_type(switch, figures):
    figures.sort()
    if switch.get('tab', False):
        open_in_tab(figures)
    if switch.get('window', False):
        open_in_new_window(figures)
    if switch.get('default_app', False):
        open_with_default_app(figures)
    if switch.get('with_preview', False):
        open_in_preview(figures)

def show_folder_plots(switch, fig_dir = f'{os.getcwd()}/zome_plots'):
    if os.path.exists(fig_dir) and os.path.isdir(fig_dir):
        figures = [os.path.join(fig_dir, f) for f in os.listdir(fig_dir) if f.endswith('.png') or f.endswith('.pdf')]
        # print(figures)
        if figures:
            show_plot_type(switch, figures)



# ------------
#     Run
# ------------
if __name__ == '__main__':
    switch = {
        'tab':          False, 
        'window':       True,
        'default_app':  False,
        'with_preview': False
        }

    test_switch = {
        'show_folder':  False,
        'show_all':     True,
        }


    if test_switch['show_folder']:
        folder = '/home/565/cb4968/Documents/code/phd/zome_plots/mse_var'
        show_folder_plots(switch, fig_dir = os.path.join(fig_dir, folder))

    if test_switch['show_all']:
        fig_dir = f'{os.getcwd()}/zome_plots'
        show_folder_plots(switch, fig_dir = fig_dir)
        for folder in os.listdir(fig_dir):
            show_folder_plots(switch, fig_dir = os.path.join(fig_dir, folder))


