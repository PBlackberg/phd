import xarray as xr
import os

def save_file(data, folder='', filename='', path = ''):
    ''' Saves file to specified folder and filename, or path '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path)


def rename_var(folder_path, old_var_name, new_var_name):
    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        # Ensure we're only processing .nc (netCDF) files
        if filename.endswith('.nc'):
            file_path = os.path.join(folder_path, filename)
            
            # Load the dataset from the current file
            ds = xr.open_dataset(file_path)
            
            # Check if old_var_name exists in the dataset before attempting to rename
            if old_var_name in ds.data_vars:
                # Rename the variable
                ds = ds.rename({old_var_name: new_var_name})
                
                # Save the dataset back to the same file
                save_file(ds, folder = folder_path, filename = filename)
                print(f"Renamed variable in {file_path}")

            # Close the dataset
            ds.close()

    print("Variable renaming completed for all files.")



if __name__ == '__main__':
    folder_path = '/Users/cbla0002/Documents/data/pr/sample_data/obs'
    old_var_name = 'precip'
    new_var_name = 'pr'
    rename_var(folder_path, old_var_name, new_var_name)








