------------
Introduction
------------

`GPflowOpt <https://github.com/GPflow/GPflowOpt/>`_ is a library for Bayesian Optimization with `GPflow <https://github.com/GPflow/GPflow/>`_.
It makes use of TensorFlow for computation of acquisition functions, to offer scalability, and avoid implementation of gradients.
The package was created, and is currently maintained by `Joachim van der Herten <http://sumo.intec.ugent.be/jvanderherten>`_ and `Ivo Couckuyt <http://sumo.intec.ugent.be/icouckuy>`_

Currently the software is pre-release and under construction, hence it lacks a lot of functionality and testing. This documentation
is also incomplete and under development. The project is open source: if you feel you have some relevant skills and are interested in
contributing then please contact us on `GitHub <https://github.com/GPflow/GPflowOpt>`_ by opening an issue or pull request.

Install
--------
1. Install package

A straightforward way to install GPflowOpt is to clone its repository and run

``pip install . --process-dependency-links``

in the root folder. This also installs required dependencies including TensorFlow.
For alternative TensorFlow installations (e.g., gpu), please see the instructions on the main `TensorFlow webpage <https://www.tensorflow.org/install/>`_.

2. Development

GPflowOpt is a pure python library so you could just add it to your python path. We use

``pip  install -e . --process-dependency-links``

3. Testing and documentation

The tests require some additional dependencies that need to be installed first with
``pip install -e .[test]``. Afterwards the tests can be run with ``python setup.py test``.

Similarly, to build the documentation,
first install the extra dependencies with ``pip install -e .[docs]``.
Then proceed with ``python setup.py build_sphinx``.

Getting started
---------------

A simple example of Bayesian optimization to get up and running is provided by the
:ref:`first steps into Bayesian optimization <notebooks/firststeps.ipynb>` notebook

For more advanced use cases have a look at the other :ref:`tutorial <tutorials>` notebooks and the :ref:`api`.

Acknowledgements
-----------------
Joachim van der Herten and Ivo Couckuyt are Ghent University - imec postdoctoral fellows. Ivo Couckuyt is supported
by FWO Vlaanderen.
