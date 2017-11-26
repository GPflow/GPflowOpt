------------
Introduction
------------

`GPflowOpt <https://github.com/GPflow/GPflowOpt/>`_ is a library for Bayesian Optimization with `GPflow <https://github.com/GPflow/GPflow/>`_.
It makes use of TensorFlow for computation of acquisition functions, to offer scalability, and avoid implementation of gradients.
The package was created, and is currently maintained by `Joachim van der Herten <http://sumo.intec.ugent.be/jvanderherten>`_ and `Ivo Couckuyt <http://sumo.intec.ugent.be/icouckuy>`_

The project is open source: if you feel you have some relevant skills and are interested in
contributing then please contact us on `GitHub <https://github.com/GPflow/GPflowOpt>`_ by opening an issue or pull request.

Install
-------
1. Install package

A straightforward way to install GPflowOpt is to clone its repository and run

``pip install . --process-dependency-links``

in the root folder. This also installs required dependencies including TensorFlow.
For alternative TensorFlow installations (e.g., gpu), please see the instructions on the main `TensorFlow webpage <https://www.tensorflow.org/install/>`_.

2. Development

GPflowOpt is a pure python library so you could just add it to your python path. We use

``pip  install -e . --process-dependency-links``

3. Testing

For testing, GPflowOpt uses `nox <https://nox.readthedocs.io/en/latest/>`_ to automatically create a virtualenv and
install the additional test dependencies. To install nox:

``pip install nox-automation``

followed by

``nox``

to run all test sessions.

4. Documentation

To build the documentation, first install the extra dependencies with
``pip install -e .[docs]``. Then proceed with ``python setup.py build_sphinx``.

Getting started
---------------

A simple example of Bayesian optimization to get up and running is provided by the
:ref:`first steps into Bayesian optimization <notebooks/firststeps.ipynb>` notebook

For more advanced use cases have a look at the other :ref:`tutorial <tutorials>` notebooks and the :ref:`api`.

Citing GPflowOpt
-----------------

To cite GPflowOpt, please reference the preliminary arXiv paper. Sample Bibtex is given below:

| @ARTICLE{GPflowOpt2017,
| author = {Knudde, Nicolas and {van der Herten}, Joachim and Dhaene, Tom and Couckuyt, Ivo},
| title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
| journal = {arXiv preprint -- arXiv:1711.03845},
| year    = {2017},
| url     = {https://arxiv.org/abs/1711.03845}
| } 

Acknowledgements
-----------------
Joachim van der Herten and Ivo Couckuyt are Ghent University - imec postdoctoral fellows. Ivo Couckuyt is supported
by FWO Vlaanderen.
