import os
import inspect
import matplotlib.pyplot as plt
from pathlib import Path

def get_path(name):
    '''
    Get the path of the specified file or folder within the root directory.
    Searches recursively starting from the directory of the current script.
    
    Parameters:
    - name: str, the name of the file or folder to search for.

    Returns:
    - str or None: the full path to the file or folder if found, otherwise None.
    '''
    # Set the root directory to the directory where this script is located
    root_directory = Path(__file__).resolve().parent
    
    # Use rglob to recursively search for files and directories with the given name
    for path in root_directory.rglob(name):
        return str(path)  # Return the first match as a string
    
    # If not found, return None
    return None

def save_fig(name):
    base_folder = 'fig'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, name)

    plt.savefig(file_path)

    return None