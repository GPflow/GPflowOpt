from __future__ import print_function
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
import glob
import traceback
import sys
import time
import os
import pytest

this_dir = os.path.dirname(__file__)
nbpath = os.path.join(this_dir, '../../doc/source/notebooks/')
blacklist = ['hyperopt.ipynb', 'mes_benchmark.ipynb', 'constrained_bo_mes.ipynb']
lfiles = [f for f in glob.glob(nbpath+"*.ipynb") if f not in map(lambda b: nbpath+b, blacklist)]


def _exec_notebook(notebook_filename, nbpath):
    pythonkernel = 'python' + str(sys.version_info[0])
    ep = ExecutePreprocessor(timeout=600, kernel_name=pythonkernel, interrupt_on_timeout=True)
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=nbformat.current_nbformat)
        try:
            ep.preprocess(nb, {'metadata': {'path': nbpath}})
        except CellExecutionError:
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print('-' * 60)
            assert False, 'Error executing the notebook %s. See above for error.' % notebook_filename


@pytest.mark.parametrize('notebook', lfiles)
def test_notebook(notebook):
    t = time.time()
    _exec_notebook(notebook, nbpath)
    print(notebook, 'took %g seconds.' % (time.time()-t))
