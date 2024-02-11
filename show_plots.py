import subprocess
import glob
import os

def open_test_plots_tab(figures_path = f'{os.getcwd()}/zome_plots/*.png'):
    for figure in glob.glob(figures_path):
        subprocess.run(["code", "-r", figure])


def open_test_plots_window(figures_path=f'{os.getcwd()}/zome_plots/*.png'):
    figures = glob.glob(figures_path)
    if figures:
        subprocess.run(["code", "-n", figures[0]])
        for figure in figures[1:]:
            subprocess.run(["code", "-r", figure])

tab = False
if tab:
    open_test_plots_tab()

window = True
if window:
    open_test_plots_window()
