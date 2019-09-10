import os
import shutil
from subprocess import Popen, PIPE, STDOUT

def talk(args, path, stdout=True, stdin=False, dry_run=False):
    # print("Running command: {}".format(" ".join(args)))
    if dry_run:
        return 0, None

    p = Popen(args, cwd=path, stdout=None if stdout == False else PIPE, stdin=None if stdin == False else PIPE, stderr=PIPE)
    if stdin:
        comm = p.communicate(stdin)
    elif stdout:
        comm = p.communicate()
    else:
        return (p.returncode, None, None)

    out, err = None if comm[0] == None else comm[0].decode("utf-8"), None if comm[1] == None else comm[1].decode("utf-8")
    return (p.returncode, out, err,)

