''' 
# ------------------------
#    Rename variable
# ------------------------
This script renames a variable in an xarray dataset
'''



# ------------------------------------------------------------------------------------ Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import os



# ------------------------
#    Rename variable
# ------------------------
# ----------------------------------------------------------------------------------- Save file --------------------------------------------------------------------------------------------------------- #
def save_file(data, folder='', filename='', path = ''):
    ''' Saves file to specified folder and filename, or path '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path)


# --------------------------------------------------------------------------------- Rename variable --------------------------------------------------------------------------------------------------------- #
def rename_var(folder_path, old_var_name, new_var_name):
    for filename in os.listdir(folder_path):
        if filename.endswith('.nc'):
            file_path = os.path.join(folder_path, filename)
            ds = xr.open_dataset(file_path)
            if old_var_name in ds.data_vars:
                ds = ds.rename({old_var_name: new_var_name})    # rename variable
                save_file(ds, folder = folder_path, filename = filename)
                print(f"Renamed variable in {file_path}")
            ds.close()
    print("Variable renaming completed for all files.")



# ------------------------
#          Run
# ------------------------
# ------------------------------------------------------------------------------ Choose variable to rename --------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    folder_path = '/Users/cbla0002/Documents/data/sample_data/tas/obs'
    old_var_name, new_var_name = 'sst', 'tas'
    
    rename_var(folder_path, old_var_name, new_var_name)








