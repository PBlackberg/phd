import os
import shutil

# Define source and destination base directories
src_base_dir = '/Users/cbla0002/Desktop/metrics'
dest_base_dir = '/Users/cbla0002/Documents/data/metrics'

# Track the current metric folder being processed
current_metric_folder = None

# Walk through the source directory structure
for foldername, _, filenames in os.walk(src_base_dir, topdown=False):  # Note: topdown=False is used to traverse from bottom to top
    # Skip removing the base "metrics" directory
    if foldername == src_base_dir:
        continue
    
    for filename in filenames:
        # Construct full file path
        src_file_path = os.path.join(foldername, filename)
        
        # Skip .DS_Store files or other unwanted files
        if filename == '.DS_Store':
            continue
        
        # Find the relative path of the file within the source directory
        relative_path = os.path.relpath(src_file_path, src_base_dir)
        
        # Find the metric folder (considering the second folder level as the metric folder)
        parts = relative_path.split(os.path.sep)
        if len(parts) < 2:
            continue  # Skip files directly under src_base_dir
        
        metric_folder = parts[1]  # Considering the second level as the metric folder
        
        # If the metric folder changes, print a message indicating completion of the previous metric folder
        if current_metric_folder is not None and metric_folder != current_metric_folder:
            print(f'All files from metric folder {current_metric_folder} have been moved')
        
        # Update the current metric folder being processed
        current_metric_folder = metric_folder
        
        # Construct the corresponding destination file path
        dest_file_path = os.path.join(dest_base_dir, relative_path)
        
        # Create destination directory if it does not exist
        dest_folder = os.path.dirname(dest_file_path)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Remove the destination file if it already exists, to allow overwriting
        if os.path.exists(dest_file_path):
            os.remove(dest_file_path)
        
        # Move the file
        shutil.move(src_file_path, dest_file_path)
    
    # Delete .DS_Store file if exists
    ds_store_file = os.path.join(foldername, '.DS_Store')
    if os.path.exists(ds_store_file):
        os.remove(ds_store_file)
    
    # After moving all files from a directory, try to remove the directory
    try:
        os.rmdir(foldername)
    except OSError as e:
        print(f'Could not remove directory {foldername}: {str(e)}')
        print('Contents:', os.listdir(foldername))

# Print the completion message for the last metric folder processed
if current_metric_folder is not None:
    print(f'All files from metric folder {current_metric_folder} have been moved')
