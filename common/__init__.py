
# Python
from os.path import abspath, exists, join


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