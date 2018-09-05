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

import re

from pkg_resources import parse_version
from setuptools import setup

VERSIONFILE = "gpflowopt/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Dependencies of GPflowOpt
dependencies = ['numpy>=1.9', 'scipy>=0.16', 'GPflow==0.5.0']
min_tf_version = '1.0.0'

# Detect if TF is installed or outdated.
# If the right version is installed, do not list as requirement to avoid installing over e.g. tensorflow-gpu
# To avoid this, rely on importing rather than the package name (like pip).
try:
    # If tf not installed, import raises ImportError
    import tensorflow as tf

    if parse_version(tf.__version__) < parse_version(min_tf_version):
        # TF pre-installed, but below the minimum required version
        raise DeprecationWarning("TensorFlow version below minimum requirement")
except (ImportError, DeprecationWarning) as e:
    # Add TensorFlow to dependencies to trigger installation/update
    dependencies.append('tensorflow>={0}'.format(min_tf_version))

setup(name='gpflowopt',
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
      packages=["gpflowopt", "gpflowopt.acquisition"],
      package_dir={'gpflowopt': 'gpflowopt'},
      py_modules=['gpflowopt.__init__'],
      test_suite='testing',
      install_requires=dependencies,
      extras_require={
          'gpu': ['tensorflow-gpu>=1.0.0'],
          'docs': ['sphinx==1.7.8', 'sphinx_rtd_theme', 'numpydoc==0.8.0', 'nbsphinx==0.3.4', 'jupyter'],
      },
      dependency_links=['https://github.com/GPflow/GPflow/archive/0.5.0.tar.gz#egg=GPflow-0.5.0'],
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
