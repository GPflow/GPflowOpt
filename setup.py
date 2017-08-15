#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Joachim van der Herten
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
import re

VERSIONFILE="GPflowOpt/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='GPflowOpt',
      version=verstr,
      author="Joachim van der Herten, Ivo Couckuyt",
      author_email="joachim.vanderherten@ugent.be",
      description=("Bayesian Optimization with GPflow"),
      license="Apache License 2.0",
      keywords="machine-learning bayesian-optimization tensorflow",
      url="http://github.com/gpflow/gpflowopt",
      package_data={},
      include_package_data=True,
      ext_modules=[],
      packages=["GPflowOpt", "GPflowOpt.acquisition"],
      package_dir={'GPflowOpt': 'GPflowOpt'},
      py_modules=['GPflowOpt.__init__'],
      test_suite='testing',
      install_requires=['numpy>=1.9', 'scipy>=0.16', 'GPflow>=0.3.5'],
      extras_require={'tensorflow': ['tensorflow>=1.0.0'],
                      'tensorflow with gpu': ['tensorflow-gpu>=1.0.0'],
                      'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc', 'nbsphinx', 'jupyter'],
                      'test': ['nose', 'coverage', 'six', 'parameterized', 'nbconvert', 'nbformat','jupyter',
                               'jupyter_client', 'matplotlib']
                      },
      dependency_links=['https://github.com/GPflow/GPflow/tarball/master#egg=GPflow-0.3.5'],
      classifiers=['License :: OSI Approved :: Apache Software License',
                   'Natural Language :: English',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
