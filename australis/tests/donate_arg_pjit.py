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

"""Stage out a jitted function with multiple arguments and argument donation."""

import numpy as np

import jax
import jax.numpy as jnp
import jax.experimental.pjit
from jax.interpreters.pxla import PartitionSpec as P
from jax.stages import Lowered
from jax.experimental.australis import exporter


def f(x, y):
  return (x[0] + x[1] * y, x[0])  # Avoid untupling.


def lower() -> Lowered:
  device_mesh = np.array(exporter.fake_devices(2, 'tpu'))
  mesh_axis_names = ('x',)
  fn = jax.experimental.pjit.pjit(
      f,
      in_axis_resources=((P('x'), P('x')), None),
      out_axis_resources=P('x'),
      donate_argnums=0)
  with jax.experimental.maps.Mesh(device_mesh, mesh_axis_names):
    return fn.lower((jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
                     jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))),
                    jax.ShapeDtypeStruct((), jnp.dtype('float32')))


if __name__ == '__main__':
  exporter.run(lower)
