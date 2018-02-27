# GPflowOpt
GPflowOpt is a python package for Bayesian Optimization using [GPflow](https://github.com/GPflow/GPflow), and uses [TensorFlow](http://www.tensorflow.org). It was [initiated](https://github.com/GPflow/GPflow/issues/397) and is currently maintained by [Joachim van der Herten](http://sumo.intec.ugent.be/members?q=jvanderherten) and [Ivo Couckuyt](http://sumo.intec.ugent.be/icouckuy). The full list of contributors (in alphabetical order) is Ivo Couckuyt, Tom Dhaene, James Hensman, Nicolas Knudde, Alexander G. de G. Matthews and Joachim van der Herten. Special thanks also to all [GPflow contributors](http://github.com/GPflow/GPflow/graphs/contributors) as this package would not be able to exist without their effort.

[![Build Status](https://travis-ci.org/GPflow/GPflowOpt.svg?branch=master)](https://travis-ci.org/GPflow/GPflowOpt)
[![Coverage Status](https://codecov.io/gh/GPflow/GPflowOpt/branch/master/graph/badge.svg)](https://codecov.io/gh/GPflow/GPflowOpt)
[![Documentation Status](https://readthedocs.org/projects/gpflowopt/badge/?version=latest)](http://gpflowopt.readthedocs.io/en/latest/?badge=latest)

# Install

The easiest way to install GPflowOpt involves cloning this repository and running
```
pip install . --process-dependency-links
```
in the source directory. This also installs all required dependencies (including TensorFlow, if needed). For more detailed installation instructions, see the [documentation](https://gpflowopt.readthedocs.io/en/latest/intro.html#install).

# Contributing
If you are interested in contributing to this open source project, contact us through an issue on this repository. For more information, see the [notes for contributors](contributing.md).

# Citing GPflowOpt

To cite GPflowOpt, please reference the preliminary arXiv paper. Sample Bibtex is given below:

```
@ARTICLE{GPflowOpt2017,
   author = {Knudde, Nicolas and {van der Herten}, Joachim and Dhaene, Tom and Couckuyt, Ivo},
    title = "{{GP}flow{O}pt: {A} {B}ayesian {O}ptimization {L}ibrary using Tensor{F}low}",
  journal = {arXiv preprint -- arXiv:1711.03845},
  year    = {2017},
  url     = {https://arxiv.org/abs/1711.03845}
}
```
