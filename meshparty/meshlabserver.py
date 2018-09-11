import os
import subprocess
import glob
import numpy as np

from multiwrapper import multiprocessing_utils as mu

HOME = os.path.expanduser("~")

# path_to_scripts = os.path.dirname(__file__) + "/meshlabserver_scripts/"
path_to_scripts = HOME + "/PyChunkedGraph/pychunkedgraph/meshing/meshlabserver_scripts/"


def run_meshlab_script(script_name, arg_dict):
    """ Runs meshlabserver script --headless

    No X-Server required

    :param script_name: str
    :param arg_dict: dict [str: str]
    """
    arg_string = "".join(["-{0} {1} ".format(k, arg_dict[k])
                          for k in arg_dict.keys()])
    command = "xvfb-run --auto-servernum --server-num=1 meshlabserver -s {0}/{1} {2}".\
        format(path_to_scripts, script_name, arg_string)
    p = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    p.wait()


def _run_meshlab_script_on_dir_thread(args):
    script_name, path_block, out_dir, suffix, arg_dict = args

    for path in path_block:
        out_path = "{}/{}{}.obj".format(out_dir,
            "".join(os.path.basename(path).split(".")[:-1]), suffix)

        this_arg_dict = {"i": path, "o": out_path}
        this_arg_dict.update(arg_dict)

        run_meshlab_script(script_name, this_arg_dict)


def run_meshlab_script_on_dir(script_name, in_dir, out_dir, suffix, arg_dict={},
                      n_threads=1):
    paths = glob.glob(in_dir + "/*.obj")

    print(len(paths))

    if len(suffix) > 0:
        suffix = "_{}".format(suffix)

    n_jobs = n_threads * 3
    if len(paths) < n_jobs:
        n_jobs = len(paths)

    path_blocks = np.array_split(paths, n_jobs)

    multi_args = []
    for path_block in path_blocks:
        multi_args.append([script_name, path_block, out_dir, suffix, arg_dict])

    if n_threads == 1:
        mu.multiprocess_func(_run_meshlab_script_on_dir_thread,
                             multi_args, debug=True,
                             verbose=True, n_threads=1)
    else:
        mu.multisubprocess_func(_run_meshlab_script_on_dir_thread,
                                multi_args, n_threads=n_threads)

