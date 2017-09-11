# Contributing to GPflowOpt
This file contains notes for potential contributors to GPflowOpt, as well as some notes that may be helpful for maintenance. As part of the GPflow organisation, GPflowOpt follows the same philosophy and principles with regards to scope and code quality as GPflow.

## Project scope
We do welcome contributions to GPflowOpt. However, the project is deliberately of limited scope, to try to ensure a high quality codebase: if you'd like to contribute a feature, please raise discussion via a GitHub issue.

Due to limited scope we may not be able to review and merge every feature, however useful it may be. Particularly large contributions or changes to core code are harder to justify against the scope of the project or future development plans. For these contributions, we suggest you publish them as a separate package based on GPflowOpt's interfaces. We can link to your project from an issue discussing the topic or within the repository. Discussing a possible contribution in an issue should give an indication to how broadly it is supported to bring it into the codebase.

## Code Style
 - Python code should follow roughly the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. We allow exceptions for, e.g., capital (single-letter) variable names to correspond to the notation of a paper (matrices, vectors, etc.). To help with this, we suggest using a plugin for your editor (we use pycharm).
 - Practise good code as far as is reasonable. Simpler is usually better. Avoid using complicated language features. Reading the existing GPflowOpt code should give a good idea of the expected style.

## Pull requests and the master branch
All code that is destined for the master branch of GPflowOpt goes through a PR and will be reviewed. Only a small number of people can merge PRs onto the master branch (currently Joachim van der Herten and Ivo Couckuyt).

## Tests and continuous integration
GPflowOpt is 99% covered by the testing suite. We expect changes to code to pass these tests, and for new code to be covered by new tests. Currently, tests are run by travis (python 2.7, 3.5 and 3.6), coverage is reported by codecov.

By default, all tests are run on Travis except for the most expensive notebooks.

## Documentation
GPflowOpt's documentation is not comprehensive, but covers enough to get users started. We expect that new features have documentation that can help other get up to speed. The docs are mostly Jupyter notebooks that compile into html via sphinx, using nbsphinx.

## Keeping up with GPflow and TensorFlow

GPflowOpt currently tries to keep up with the GPflow master, though at some point we will start depending on the latest released version. Hence, GPflowOpt also adheres to the api of the TensorFlow version as required by GPflow. In practice this hopefully means we will support the latest (stable) TensorFlow, which is supported by GPflow. Any change in the supported version of GPflow or TensorFlow will bump the minor version number of GPflowOpt.

Changing the minimum required version of TensorFlow that we're compatible with requires a few tasks:
 - update versions in `setup.py`
 - update versions used on travis via `.travis.yml`
 - update version used by readthedocs.org via `docsrequire.txt`
 - Increment the GPflowOpt version (see below). 

## Version numbering
The main purpose of versioning GPflowOpt is user convenience: to keep the number of releases down, we try to combine several PRs into one increment.
When incrementing the version number, the following tasks are required:
 - Update the version in `GPflowOpt/_version.py`
 - Add a note to `RELEASE.md`
