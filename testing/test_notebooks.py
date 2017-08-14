from __future__ import print_function
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
import glob
import traceback
import unittest
import sys
import time
import os
from nose.plugins.attrib import attr
from parameterized import parameterized

this_dir = os.path.dirname(__file__)
nbpath = os.path.join(this_dir, '../doc/source/notebooks/')
lfiles = [(f,) for f in glob.glob(nbpath+"*.ipynb")]


@attr('notebooks')
class TestNotebooks(unittest.TestCase):

    _multiprocess_can_split_ = True

    def _execNotebook(self, notebook_filename, nbpath):
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=nbformat.current_nbformat)
            try:
                self.ep.preprocess(nb, {'metadata': {'path': nbpath}})
            except CellExecutionError:
                print('-' * 60)
                traceback.print_exc(file=sys.stdout)
                print('-' * 60)
                self.assertTrue(False, 'Error executing the notebook %s.\
                                        See above for error.' % notebook_filename)

    def setUp(self):
        pythonkernel = 'python' + str(sys.version_info[0])
        # see http://nbconvert.readthedocs.io/en/stable/execute_api.html
        self.ep = ExecutePreprocessor(timeout=600, kernel_name=pythonkernel, interrupt_on_timeout=True)

    @parameterized.expand(lfiles)
    def test_notebook(self, notebook_filename):
        t = time.time()
        self._execNotebook(notebook_filename, nbpath)
        print(notebook_filename, 'took %g seconds.' % (time.time()-t))