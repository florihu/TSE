import os
import inspect
import matplotlib.pyplot as plt

def get_path(file_name):
    '''
    Get the path of the file
    '''
    root_directory = os.path.abspath(os.path.dirname(__file__))
    return next(
    (os.path.join(root, file_name) for root, _, files in os.walk(root_directory) if file_name in files),
    None
)

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