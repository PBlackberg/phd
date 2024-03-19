'''
# --------------------------------
# Folder for saving metrics / data
# --------------------------------
to use
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-files')
import save_folders as sF
'''



# ------------------------------------------------------------------------ Packages --------------------------------------------------------------------------------------------------- #
import os



# ------------------------------------------------------------------------- Local --------------------------------------------------------------------------------------------------- #
folder_save = (os.path.expanduser("~") + '/Documents/data')             # Local - /Users/cbla0002
folder_scratch = (os.path.expanduser("~") + '/Documents/data/scratch')  # Local - /Users/cbla0002


# -------------------------------------------------------------------------- Gadi --------------------------------------------------------------------------------------------------- #
if os.path.expanduser("~") == '/home/565/cb4968':
    folder_save = ('/g/data/k10/cb4968/data')
    folder_scratch = ('/scratch/w40/cb4968')


# ------------------------------------------------------------------------- Levante --------------------------------------------------------------------------------------------------- #
if os.path.expanduser("~") == '/home/b/b382628':
    folder_save = ('/work/bb1153/b382628/data')
    folder_scratch = ('/scratch/b/b382628')



if __name__ == '__main__':
    print(f'folder save: {folder_save}')
    print(f'folder scratch: {folder_scratch}')
