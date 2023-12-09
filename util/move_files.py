''' 
# ------------------------
#      Move files
# ------------------------
This script moves downloaded metrics into corresponding folders in the data folder on laptop
'''



# ------------------------------------------------------------------------------------ Packages --------------------------------------------------------------------------------------------------------- #
import os
import shutil



# ------------------------
#      Move files
# ------------------------
# ----------------------------------------------------------------------------------- Move files --------------------------------------------------------------------------------------------------------- #
def move_files(src_base_dir, dest_base_dir):
    current_metric_folder = None
    for foldername, _, filenames in os.walk(src_base_dir, topdown=False):               # topdown=False is used to traverse from bottom to top
        if foldername == src_base_dir:
            continue
        
        for filename in filenames:
            src_file_path = os.path.join(foldername, filename)
            if filename == '.DS_Store':
                continue
            relative_path = os.path.relpath(src_file_path, src_base_dir)
            parts = relative_path.split(os.path.sep)
            if len(parts) < 2:
                continue                                                                # Skip files directly under src_base_dir
            metric_folder = parts[1]                                                    # Considering the second level as the metric folder
            if current_metric_folder is not None and metric_folder != current_metric_folder:
                print(f'All files from metric folder {current_metric_folder} have been moved')
            current_metric_folder = metric_folder
            dest_file_path = os.path.join(dest_base_dir, relative_path)                 # Construct the corresponding destination file path
            dest_folder = os.path.dirname(dest_file_path)                               # Create destination directory if it does not exist
            os.makedirs(dest_folder, exist_ok=True)
            if os.path.exists(dest_file_path):                                          # Remove the destination file if it already exists, to allow overwriting
                os.remove(dest_file_path)
            shutil.move(src_file_path, dest_file_path)                                  # Move the file
        ds_store_file = os.path.join(foldername, '.DS_Store')                           # Delete .DS_Store file if exists
        if os.path.exists(ds_store_file):
            os.remove(ds_store_file)
        try:                                                                            # After moving all files from a directory, try to remove the directory
            os.rmdir(foldername) 
        except OSError as e:
            print(f'Could not remove directory {foldername}: {str(e)}')
            print('Contents:', os.listdir(foldername))
    if current_metric_folder is not None:
        print(f'All files from metric folder {current_metric_folder} have been moved')  # Print the completion message for the last metric folder processed



# ------------------------
#          Run
# ------------------------
if __name__ == '__main__':
# ------------------------------------------------------------------------------ Choose base directories --------------------------------------------------------------------------------------------------------- #
    src_base_dir = '/Users/cbla0002/Desktop/metrics'
    dest_base_dir = '/Users/cbla0002/Documents/data/metrics'
    move_files(src_base_dir, dest_base_dir)








