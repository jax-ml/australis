# Copyright 2022 Google LLC
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

"""The user configuration used to stage out a user's model.

Note: this might in the future be automatically code-generated from bzl rules.
"""

import numpy as np
import sys

import jax
import jax.experimental.pjit
from jax.interpreters.pxla import PartitionSpec as P
import jax.numpy as jnp

from jax.stages import Lowered

from jax.experimental.australis import exporter


def f(x):
  return x[0] + x[1]


def lower() -> Lowered:
  device_mesh = np.array(exporter.fake_devices(2, 'tpu'))
  mesh_axis_names = ('x',)
  print('device_mesh ', device_mesh, file=sys.stderr)
  fn = jax.experimental.pjit.pjit(
      f, in_axis_resources=((P('x'), P('x')),), out_axis_resources=P('x'))
  with jax.experimental.maps.Mesh(device_mesh, mesh_axis_names):
    staged: Lowered = fn.lower(
        (jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
         jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))))
  return staged


if __name__ == '__main__':
  exporter.run(lower)
