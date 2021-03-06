import os
import shutil
import _pickle as pickle
import json
import time
import gc
from functools import wraps
from inspect import isfunction
import calc_utils as cx


def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        diff = round(end-start, 0)
        fn_name = f.__name__
        # args = f.__code__.co_varnames #parameters row_names
        args = locals().get('args')

        print("-" * 80)
        print('function: ' + fn_name + str(args))

        if diff < 60:
            print('elapsed time: {} seconds'.format(diff))
        elif diff > 60:
            print('elapsed time: {} minutes'.format(round(diff/60)))
        elif diff > 3600:
            print('elapsed time: {} hours'.format(round(diff/3600)))

        print("-" * 80)

        return result
    return wrapper


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    #  Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            filename = filename
            #  Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            # replace '\\' with '/'
            filepath = filepath.replace('\\', '/')
            # Add it to the list.
            file_paths.append(filepath)

    return file_paths


def get_folders(directory):
    return os.listdir(directory)


# delete all files in folder
def clean_folder(folder):
    files = os.listdir(os.path.abspath(os.curdir) + '/' + folder)
    files = [os.path.abspath(os.curdir) + '/' + folder + '/' + f for f in files]
    delete_files(files)


# create folder
def create_folder(folder):
    os.makedirs(os.path.dirname(folder + '/'), exist_ok=True)


def delete_folder(folder):
    shutil.rmtree(folder, ignore_errors=True)


def get_fnames(directory):
    """
    Return filenames of specific folder without the extension
    """
    return [f.split('.')[0] for f in os.listdir(directory)]


def load_pickle(fname):
    output = open(fname, 'rb')

    # disable garbage collector (performance hack)
    gc.disable()

    file_ = pickle.load(output)

    # enable garbage collector again (performance hack)
    gc.enable()
    output.close()

    return file_


def save_pickle(fname, item):

    output = open(fname, 'wb')

    # disable garbage collector (performance hack)
    gc.disable()

    pickle.dump(item, output)

    output.close()
    # enable garbage collector again (performance hack)
    gc.enable()


# save items in list (errors in urls p.e.)
def pickle_list(item, folder, fname, append=True):

    fname = folder + '/' + fname + '.pkl'

    if os.path.isfile(fname) is False:
        save_pickle(fname, [item])
    else:
        lst = load_pickle(fname)
        if append:
            lst.append(item)
        else:
            lst.extend(item)
        save_pickle(fname, lst)


# save function and files results
@timer
def pickle_fn(item, folder, filename):

    fname = folder + '/' + filename + '.pkl'

    if os.path.isfile(fname) is False:
        create_folder(folder)
        if isfunction(item):
            item = item()
        save_pickle(fname, item)
    else:
        item = load_pickle(fname)

    return item


# delete files in folder
def delete_files(files):

    for file_ in files:
        try:
            if os.path.isfile(file_):
                os.unlink(file_)
            else:
                print('File not found.')
        except Exception as e:
            print(e)

    if len(files) > 0:
        print('%s files deleted. \n' % len(files))
    else:
        print('Empty folder.')


# function to flatten arrays from json files
def get_values(lVals):
    res = []
    for val in lVals:
        if type(val) not in [list, set, tuple]:
            res.append(val)
        else:
            res.extend(get_values(val))
    return res


def read_json(fname, nested=False):
    with open(fname) as f:
        data = json.load(f)

    if nested:
        data = cx.get_values(data)

    return data


# save data in a json file
def save_json(fname, data):

    fname = fname + '.json'

    if os.path.isfile(fname) is False:
        with open(fname, 'w') as f:
            json.dump(data, f)
            print('%s first records were saved in file %s.' % (len(data), fname))
    else:
        with open(fname) as f:
            lst = json.load(f)

        lst.append(data)

        with open(fname, 'w') as f:
            json.dump(lst, f)
            print('%s records were saved in file %s.' % (len(data), fname))


# save data in a json file
def save_txt(fname, data):

    if os.path.isfile(fname) is False:
        with open(fname, 'w') as f:
            f.write(data)
    else:
        with open(fname) as f:
            txt = f.read(f) + '\n'

        with open(fname, 'w') as f:
            f.write(txt + data)


def logger(directory, fname, message):
    logf = open(directory + '/' + fname + '.txt', 'a')
    logf.write(message + '\n')


    # try:
    #     fname = fname + '.json'
    #
    #     # check if json files exists to prevent "[Errno 2] No such file or directory" error
    #     # and write an empty list otherwise
    #     if os.path.isfile(fname):
    #         with open(fname) as f:
    #             data = json.load(f)
    #     else:
    #         with open(fname, 'w') as f:
    #             json.dump(data, f)
    #
    #     data.append(array_)
    #
    #     with open(fname, 'w') as f:
    #         json.dump(data, f)
    #         print('%s records were saved in file %s.' % (len(data), fname))
    #
    # except Exception as e:
    #     print("File saving failed with error: %s" % e)
