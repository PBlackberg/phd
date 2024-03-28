''' 
# ----------------
#   show_plots
# ----------------
This script can save test plots, show plots, or cycle plots in a loop
It can also show plots placed in the test_plot folder in a new window or with default app
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import os
import matplotlib.pyplot as plt
import subprocess



# ---------------------
#  Save / show plots
# ---------------------
def save_figure(figure, folder =f'{os.getcwd()}/zome_plots', filename = 'test.pdf'):
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    figure.savefig(path)

def show_plot(fig, show_type = 'cycle', cycle_time = 0.5, folder = f'{os.getcwd()}/zome_plots', filename = 'test'):
    ''' If using this on supercomputer, x11 forwarding is required with XQuartz installed on your computer to show / cycle '''
    if show_type == 'save_cwd':
        save_figure(figure = fig, folder = folder, filename = filename)
        plt.close(fig)
        print(f'saved {filename}')
        return True
    elif show_type == 'show':
        plt.show()
        return True
    elif show_type == 'cycle':
        plt.ion()
        plt.show()
        plt.pause(cycle_time)
        plt.close(fig)
        plt.ioff()



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
        if figures:
            show_plot_type(switch, figures)


# ---------------------
#  Remove test plots
# ---------------------
def remove_plots(fig_dir, indent = 0):
    indent_level = "\t" * indent
    figure_files = [f for f in os.listdir(fig_dir) if f.endswith('.png') or f.endswith('.pdf')]
    for figure in figure_files:
        os.remove(os.path.join(fig_dir, figure))
        print(f"{indent_level}{figure} has been removed")
        
def remove_empty_dir(sub_dir):
    ''' Only removes directory if there are no files inside '''
    contents = [content for content in os.listdir(sub_dir)]
    if not contents:
        os.rmdir(sub_dir)
    else:
        print(f'{sub_dir} not deleted as there are not only figures in this folder (check)')

def remove_sub_dir_plots(fig_dir):
    ''' removes figure subfolders in folder '''
    sub_folders = [sub_folder for sub_folder in os.listdir(fig_dir)]
    sub_dirs = [os.path.join(fig_dir, sub_folder) for sub_folder in sub_folders if os.path.isdir(os.path.join(fig_dir, sub_folder))]
    if sub_dirs:
        for sub_dir, sub_folder in zip(sub_dirs, sub_folders):
            print(f'\t from sub_folder: {sub_folder}:')
            remove_plots(sub_dir, indent = 2)
            remove_empty_dir(sub_dir) # only removes if there is no content in the folder

def remove_folder_plots(folder = f'{os.getcwd()}/zome_plots/sub_folder'):
    ''' Removes one figure folder in zome_plots  '''
    if os.path.exists(folder) and os.path.isdir(folder):
        print(f'from {folder}:')
        remove_plots(folder, indent = 1)
        remove_sub_dir_plots(folder)
        remove_empty_dir(folder)
    else:
        print(f"Directory {folder} does not exist or is not a directory")

def remove_test_plots(directory = f'{os.getcwd()}/zome_plots'):
    ''' Removes all figure folders in zome_plots '''
    if os.path.exists(directory) and os.path.isdir(directory):
        print(f'from {directory}:')
        remove_plots(directory)
        folders = [folder for folder in os.listdir(directory)]
        folder_paths = [os.path.join(directory, folder) for folder in folders if os.path.isdir(os.path.join(directory, folder))]
        for folder_path, _ in zip(folder_paths, folders):
            remove_folder_plots(folder = folder_path)
    else:
        print(f"Directory {directory} does not exist or is not a directory")



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

    switch_test = {
        'save_plot':                False,
        'show_plots':               True,
        'remove_folder_plots':      False,
        'remove_all_plots':         False,
        }

    if switch_test['save_plot']:
        fig = plt.figure()
        x = [1, 2, 3, 4, 5]  
        y = [2, 3, 5, 7, 11] 
        plt.plot(x, y)
        show_plot(fig, show_type = 'save_cwd', folder = f'{os.getcwd()}/zome_plots/sub_folder1', filename = 'test1')
        show_plot(fig, show_type = 'save_cwd', folder = f'{os.getcwd()}/zome_plots/sub_folder1', filename = 'test2')
        show_plot(fig, show_type = 'save_cwd', folder = f'{os.getcwd()}/zome_plots/sub_folder2', filename = 'test')
        show_plot(fig, show_type = 'save_cwd', folder = f'{os.getcwd()}/zome_plots/sub_folder2/sub_folder3', filename = 'test')

    if switch_test['show_plots']:
        show_folder_plots(switch, fig_dir = f'{os.getcwd()}/zome_plots/sub_folder1')
        show_folder_plots(switch, fig_dir = f'{os.getcwd()}/zome_plots/sub_folder2/sub_folder3')

    if switch_test['remove_folder_plots']:
        remove_folder_plots(folder = f'{os.getcwd()}/zome_plots/sub_folder1')
        remove_folder_plots(folder = f'{os.getcwd()}/zome_plots/sub_folder2')

    if switch_test['remove_all_plots']:
        remove_test_plots(directory = f'{os.getcwd()}/zome_plots')
