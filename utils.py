import os
import shutil
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
from pymongo import MongoClient
from dynaconf import settings


def talk(args, path, stdout=True, stdin=False, dry_run=False):
    """
    Execute a process with a command.
    Args:
        args: Command to run
        path: Path to run the command in
        stdout: Capture the STDOUT and return
        stdin: Send input to the command
        dry_run: Don't execute the command

    Returns:
        Returns a tuple of (return code, the output of the command in STDOUT and the output of STDERR,)
    """
    # print("Running command: {}".format(" ".join(args)))
    if dry_run:
        return 0, None

    p = Popen(args,
              cwd=path,
              stdout=None if stdout == False else PIPE,
              stdin=None if stdin == False else PIPE,
              stderr=PIPE)
    if stdin:
        comm = p.communicate(stdin)
    elif stdout:
        comm = p.communicate()
    else:
        return (p.returncode, None, None)

    out, err = None if comm[0] == None else comm[0].decode(
        "utf-8"), None if comm[1] == None else comm[1].decode("utf-8")
    return (
        p.returncode,
        out,
        err,
    )

def filter_images(label):
    """filters images based on label
    
    Arguments:
        data_directory {str} -- directory of hands data with trailing /
        label {str} -- valid label to filter
    
    Returns:
        list -- list of image paths with given label
    """
    #map on which column to check based on filter values
    filter_to_column = {
        'left': 'aspectOfHand',
        'right': 'aspectOfHand',
        'dorsal': 'aspectOfHand',
        'palmar': 'aspectOfHand',
        'male': 'gender',
        'female': 'gender',
        'with_acs': 'accessories',
        'without_acs': 'accessories'
    }
    if label not in filter_to_column.keys():
        raise Exception('invalid filter. valid filters are ' + ', '.join(list(filter_to_column.keys())))

    #get image paths where the filter holds true
    client = MongoClient(host=settings.HOST,
                         port=settings.PORT,
                         username=settings.USERNAME,
                         password=settings.PASSWORD)
    coll = client.db[settings.IMAGES.METADATA_COLLECTION]
    column = filter_to_column[label]
    
    filter_image_paths = []
    for row in coll.find({column: {'$regex': label}}, {'path':1}):
        filter_image_paths.append(row['path'])
    return filter_image_paths

