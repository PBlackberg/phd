''' 
# ------------------------
#      Rename files
# ------------------------
This script renames files.
Each file has words divided by _. Words between _ can be removed or replaced
'''



# ------------------------------------------------------------------------------------ Packages --------------------------------------------------------------------------------------------------------- #
import os



# ------------------------
#      Rename files
# ------------------------
# ------------------------------------------------------------------------------------- add word --------------------------------------------------------------------------------------------------------- #
def add_word(folder_path, word_next_to, word_to_add, put_before=True):
    try:
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return
        items = os.listdir(folder_path)
        for item in items:
            if item.startswith('.') or item == '.DS_Store':
                continue
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):               # For file
                file_name, file_ext = os.path.splitext(item)
                parts = file_name.split("_")            # Split the filename into parts based on underscores
                try:                                    # Find the position of the 'word_before'
                    before_index = parts.index(word_next_to)
                except ValueError:
                    print(f"Word '{word_next_to}' not found in the filename: {item}")
                    continue

                if put_before:
                    parts.insert(before_index, word_to_add)
                else:
                    parts.insert(before_index + 1, word_to_add)
                new_filename = "_".join(parts) + file_ext
                old_item_path = os.path.join(folder_path, item)
                new_item_path = os.path.join(folder_path, new_filename)


                os.rename(old_item_path, new_item_path) # Rename the file
            elif os.path.isdir(item_path):              # for folder
                if put_before:
                    new_folder_name = item.replace(word_next_to, word_to_add + "_" + word_next_to)
                else:
                    new_folder_name = item.replace(word_next_to, word_next_to + "_" + word_to_add)
                old_item_path = os.path.join(folder_path, item)
                new_item_path = os.path.join(folder_path, new_folder_name)
                os.rename(old_item_path, new_item_path) # Rename the folder
        print("Word successfully added!")
    except Exception as e:
        print(f"An error occurred: {e}")


# ------------------------------------------------------------------------------------ remove word --------------------------------------------------------------------------------------------------------- #
def remove_word(folder_path, word_to_remove):
    try:
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return
        items = os.listdir(folder_path)
        for item in items:
            if item.startswith('.') or item == '.DS_Store':
                continue
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):               # For file
                file_name, file_ext = os.path.splitext(item)
                parts = file_name.split("_")            # Split the filename into parts based on underscores
                if word_to_remove in parts:
                    parts.remove(word_to_remove)
                new_filename = "_".join(parts) + file_ext
                old_item_path = os.path.join(folder_path, item)
                new_item_path = os.path.join(folder_path, new_filename)
                os.rename(old_item_path, new_item_path) # Rename the file
            elif os.path.isdir(item_path):              # For folder
                new_folder_name = item.replace(word_to_remove, "")
                old_item_path = os.path.join(folder_path, item)
                new_item_path = os.path.join(folder_path, new_folder_name)
                os.rename(old_item_path, new_item_path) # Rename the folder
        print("Word successfully removed!")
    except Exception as e:
        print(f"An error occurred: {e}")

def rename_files(folder_path, word_before='', word_to_add='', word_to_remove='', put_before=True):
    add_word(folder_path, word_before, word_to_add, put_before) if word_before and word_to_add else None
    remove_word(folder_path, word_to_remove) if word_to_remove else None



# ------------------------
#      Rename files
# ------------------------
# ---------------------------------------------------------------------------------- Choose what to change --------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    folder_path = '/scratch/w40/cb4968/metrics/pr/pr_o_95thprctile/obs'
    word_next_to, word_to_add = 'regridded', '144x72'   # if this is '', the function won't add anything
    put_before = False
    word_to_remove = ''                 # if this is '', the function won't remove anything
    rename_files(folder_path, word_next_to, word_to_add, word_to_remove, put_before)





