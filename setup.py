# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils import spawn
import subprocess
import os

from setuptools import setup, find_packages

__version__ = '0.0.0'

with open('README.md') as f:
  _long_description = f.read()

if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
else:
  protoc = spawn.find_executable('protoc')


def generate_proto(source):
  if not protoc:
    return
  protoc_command = [protoc, '-I.', '--python_out=.', source]
  if subprocess.call(protoc_command) != 0:
    sys.exit(-1)


generate_proto('australis/executable.proto')
generate_proto('australis/petri.proto')

setup(
    name='australis',
    version=__version__,
    description='AOT compilation for jax (to run on c++).',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=find_packages(exclude=['examples']),
    python_requires='>=3.7',
    install_requires=[
        'jax',
        'protobuf>=3.13,<4',
    ],
    url='https://github.com/jax-ml/australis',
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    zip_safe=False,
)
