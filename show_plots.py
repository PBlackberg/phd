import subprocess
import glob
import os

def open_test_plots_tab(figures_path = f'{os.getcwd()}/zome_plots/*'):
    for figure in glob.glob(figures_path):
        subprocess.run(["code", "-r", figure])

def open_test_plots_window(figures_path=f'{os.getcwd()}/zome_plots/*'):
    figures = glob.glob(figures_path)
    if figures:
        subprocess.run(["code", "-n", figures[0]])
        for figure in figures[1:]:
            subprocess.run(["code", "-r", figure])

def open_plots_with_default_app(figures_path=f'{os.getcwd()}/zome_plots/*'):
    figures = glob.glob(figures_path)
    for figure in figures:
        subprocess.run(["open", figure])

def open_plots_with_preview(figures_path=os.path.join(os.getcwd(), 'zome_plots', '*.png')):
    figures = glob.glob(figures_path)
    figures.sort()
    if not figures:
        print("No PNG files found in the specified directory.")
        return
    command = ["open", "-a", "Preview"] + [f'"{figure}"' for figure in figures]
    command_string = ' '.join(command)
    subprocess.run(command_string, shell=True)

switch = {
    'tab':          False, 
    'window':       False,
    'default_app':  False,
    'with_preview': True
    }

if switch['tab']:
    open_test_plots_tab()
if switch['window']:
    open_test_plots_window()
if switch['default_app']:
    open_plots_with_default_app()
if switch['with_preview']:
    open_plots_with_preview()


