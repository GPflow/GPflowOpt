------------
Introduction
------------

Install
--------
1. Install TensorFlow

You will need version 1.0 or above. We find that for many users pip installation is the fastest way to get going.
``pip install tensorflow``

For alternative installations, please see instructions on the main `TensorFlow webpage <https://www.tensorflow.org/install/>`_. 

2. Install package

GPflowOpt is a pure python library for now, so you could just add it to your path (we use ``python setup.py develop``). A straightforward way to install GPflowOpt including all of its dependencies: ``pip install . --process-dependency-links``
A different way to install the package: ``python setup.py install`` (untested)

You can run the tests with ``python setup.py test``.

To build the documentation, first install extra dependencies with ``pip install .[docs]``, then proceed with ``python setup.py build_sphinx``.
