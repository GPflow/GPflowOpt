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

import nox

TEST_DEPS = ('six',)
SYSTEM_TEST_DEPS = ('nbconvert', 'nbformat', 'jupyter', 'jupyter_client', 'matplotlib')


@nox.session
def unit(session):
    session.install('pytest', 'pytest-cov', *TEST_DEPS)
    session.install('-e', '.', '--process-dependency-links')

    # Run py.test against the unit tests.
    session.run(
        'py.test',
        '--cov-report=',
        '--cov-append',
        '--cov=gpflowopt',
        '--color=yes',
        '--cov-config=.coveragerc',
        'testing/unit'
    )


@nox.session
def system(session):
    session.install('pytest', 'pytest-cov', *(TEST_DEPS + SYSTEM_TEST_DEPS))
    session.install('-e', '.', '--process-dependency-links')

    # Run py.test against the unit tests.
    session.run(
        'py.test',
        '--cov-report=',
        '--cov-append',
        '--cov=gpflowopt',
        '--cov-config=.coveragerc',
        'testing/system'
    )


@nox.session
def cover(session):
    session.install('coverage', 'pytest-cov')
    session.run('coverage', 'report')
