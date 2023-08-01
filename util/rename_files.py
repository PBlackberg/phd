import os

def add_word(folder_path, word_next_to, word_to_add, put_before=True):
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return

        # Get a list of all files and subfolders in the folder
        items = os.listdir(folder_path)

        # Iterate through the items and rename files and folders
        for item in items:
            # Skip hidden files/folders (starting with a dot) and .DS_Store files
            if item.startswith('.') or item == '.DS_Store':
                continue

            item_path = os.path.join(folder_path, item)

            if os.path.isfile(item_path):  # Check if it's a file
                # Get the file extension (if any)
                file_name, file_ext = os.path.splitext(item)

                # Split the filename into parts based on underscores
                parts = file_name.split("_")

                # Find the position of the 'word_before'
                try:
                    before_index = parts.index(word_next_to)
                except ValueError:
                    print(f"Word '{word_next_to}' not found in the filename: {item}")
                    continue

                if put_before:
                    # Insert the 'word_to_add' before 'word_before'
                    parts.insert(before_index, word_to_add)
                else:
                    # Insert the 'word_to_add' after 'word_before'
                    parts.insert(before_index + 1, word_to_add)

                # Join the parts back to create the new filename
                new_filename = "_".join(parts) + file_ext

                # Get the full paths of the old and new filenames
                old_item_path = os.path.join(folder_path, item)
                new_item_path = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(old_item_path, new_item_path)
            elif os.path.isdir(item_path):  # Check if it's a directory
                # Add the 'word_to_add' before or after 'word_before' in the folder name
                if put_before:
                    new_folder_name = item.replace(word_next_to, word_to_add + "_" + word_next_to)
                else:
                    new_folder_name = item.replace(word_next_to, word_next_to + "_" + word_to_add)

                # Get the full paths of the old and new folders
                old_item_path = os.path.join(folder_path, item)
                new_item_path = os.path.join(folder_path, new_folder_name)

                # Rename the folder
                os.rename(old_item_path, new_item_path)

        print("Word successfully added!")
    except Exception as e:
        print(f"An error occurred: {e}")


def remove_word(folder_path, word_to_remove):
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return

        # Get a list of all files and subfolders in the folder
        items = os.listdir(folder_path)

        # Iterate through the items and rename files and folders
        for item in items:
            # Skip hidden files/folders (starting with a dot) and .DS_Store files
            if item.startswith('.') or item == '.DS_Store':
                continue

            item_path = os.path.join(folder_path, item)

            if os.path.isfile(item_path):  # Check if it's a file
                # Get the file extension (if any)
                file_name, file_ext = os.path.splitext(item)

                # Split the filename into parts based on underscores
                parts = file_name.split("_")

                # Remove the 'word_to_remove' if it exists in the parts list
                if word_to_remove in parts:
                    parts.remove(word_to_remove)

                # Join the parts back to create the new filename
                new_filename = "_".join(parts) + file_ext

                # Get the full paths of the old and new filenames
                old_item_path = os.path.join(folder_path, item)
                new_item_path = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(old_item_path, new_item_path)
            elif os.path.isdir(item_path):  # Check if it's a directory
                # Remove the 'word_to_remove' from the folder name
                new_folder_name = item.replace(word_to_remove, "")

                # Get the full paths of the old and new folders
                old_item_path = os.path.join(folder_path, item)
                new_item_path = os.path.join(folder_path, new_folder_name)

                # Rename the folder
                os.rename(old_item_path, new_item_path)

        print("Word successfully removed!")
    except Exception as e:
        print(f"An error occurred: {e}")


def rename_files(folder_path, word_before='', word_to_add='', word_to_remove='', put_before=True):
    add_word(folder_path, word_before, word_to_add, put_before) if word_before and word_to_add else None
    remove_word(folder_path, word_to_remove) if word_to_remove else None



if __name__ == '__main__':
    folder_path = '/Users/cbla0002/Documents/data/org/metrics/obj_snapshot/cmip6'
    word_next_to = 'daily'
    word_to_add = 'obj_snapshot'
    word_to_remove = ''

    rename_files(folder_path, word_next_to, word_to_add, word_to_remove, put_before = True)





