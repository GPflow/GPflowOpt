# Contributing to GPflowOpt
This file contains notes for potential contribtors to GPflowOpt, as well as some notes that may be helpful for maintainance.

## Project scope
We do welcome contributions to GPflowOpt. However, the project is deliberately of limited scope, to try to ensure a high quality codebase: if you'd like to contribute a feature, please raise discussion via a github issue.

Due to limited scope we may not be able to review and merge every feature, however useful it may be. Particularly large contributions or changes to core code are harder to justify against the scope of the project or future development plans. For these contributions like this, we suggest you publish them as a separate package that extends GPflowOpt. We can link to your project from an issue discussing the topic or within the repository. Discussing a possible contribution in an issue should give an indication to how broadly it is supported to bring it into the codebase.

## Code Style
 - Python code should follow roughly the pep8 style. We allow exceptions for, e.g., capital (single-letter) variable names to correspond to the notation of a paper (matrices, vectors, etc.). To help with this, we suggest using a plugin for your editor (we use pycharm). 
 - Practise good code as far as is reasonable. Simpler is usually better. Avoid using compicated language features. Reading the existing GPflowOpt code should give a good idea of the expected style.

## Pull requests and the master branch
All code that is destined for the master branch of GPflowOpt goes through a PR. Only a small number of people can merge PRs onto the master branch (currently Joachim van der Herten and Ivo Couckuyt).

## Tests and continuous integration
GPflowOpt is 99% covered by the testing suite. We expect changes to code to pass these tests, and for new code to be covered by new tests. Currently, tests are run by travis (python 2.7, 3.5 and 3.6), coverage is reported by codecov.

By default, all tests are run on travis except for the most expensive notebooks.

## Documentation
GPflowOpt's documentation is not comprehensive, but covers enough to get users started. We expect that new features have documentation that can help other get up to speed. The docs are mostly Jupyter notebooks that compile into html via sphinx, using nbsphinx.

## Keeping up with GPflow and tensorflow

GPflowOpt tries to keep up with the latest *released* version of GPflow (not necessarily master). Hence, GPflowOpt also adheres to the api of the tensorflow version as required by GPflow. In practice this hopefully means we will support the latest (stable) tensorflow, which is supported by GPflow. Any change in the version of GPflow or tensorflow will bump the minor version number of GPflowOpt.

Changing the version of GPflow and tensorflow that we're compatible with requires a few tasks:
 - update versions in `setup.py`
 - update versions used on travis via `.travis.yml`
 - update version ussed by readthedocs.org via `docsrequire.txt`
 - Increment the GPflowOpt version (see below). 

## Version numbering
The main purpose of versioning GPflowOpt is user convenience: to keep the number of releases down, we try to combine seversal PRs into one increment. As we work towards something that we might call 1.0, including changes to thhe GPflowOpt API. Minor version bumps (X.1) are used for updates to follow a new GPflow or TensorFlow API, or introduce incremental new features.
When incrementing the version number, the following tasks are required:
 - Update the version in `GPflowOpt/_version.py`
 - Udate the version in the `doc/source/conf.py`
 - Add a note to `RELEASE.md`
