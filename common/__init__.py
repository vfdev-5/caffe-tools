
# Python
from os.path import abspath, exists, join

# Project
from config import cfg


def find_file(relative_path, folders):
    """
    :param relative_path: input relative path to a file
    :param folders: list of folders where to search the file
    :return: absolute file path if found otherwise None
    """

    for folder in folders:
        if exists(join(folder, relative_path)):
            return abspath(join(folder, relative_path))
    return None


def get_abspath(path):
    """
    :param path: relative or absolute path
    :return: absolute path if found otherwise None
    """
    folders = [cfg['DATASET_PATH']]
    folders.extend(cfg['MODELS_PATH_LIST'])
    folders.extend(cfg['RESOURCES_PATH_LIST'])
    return path if exists(path) else find_file(path, folders)

