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

import functools
import numpy as np
import sys

import jax
import jax.experimental.pjit
from jax.interpreters.pxla import PartitionSpec as P
import jax.numpy as jnp

from jax.stages import Lowered

from jax.experimental.australis import exporter


@functools.partial(
    jax.experimental.pjit.pjit,
    in_axis_resources=((P('x'), P('x')), None),
    out_axis_resources=P('x'),
    donate_argnums=0)
def donate_arg_fn(x, y):
  return (x[0] + x[1] * y, x[0])  # Avoid untupling.


@functools.partial(
    jax.experimental.pjit.pjit,
    in_axis_resources=P('x'),
    out_axis_resources=P('x'))
def simple_fn(x):
  return 2 * x


@functools.partial(
    jax.experimental.pjit.pjit,
    in_axis_resources=((P('x'), P('x')),),
    out_axis_resources=P('x'))
def tuple_fn(x):
  return x[0] + x[1]


@functools.partial(
    jax.experimental.pjit.pjit,
    in_axis_resources=((P('x'), P('x')), None),
    out_axis_resources=(P('x'), None, P('x')))
def multi_return_fn(x, y):
  return (x[0] + x[1] * y, x[0], x[1]), y + 1, x


@functools.partial(
    jax.experimental.pjit.pjit,
    in_axis_resources=((P('x'), P('x')), None),
    out_axis_resources=P('x'))
def higher_arity_fn(x, y):
  return x[0] + x[1] * y


def lower() -> Lowered:
  device_mesh = np.array(exporter.fake_devices(2, 'tpu'))
  mesh_axis_names = ('x',)
  print('device_mesh ', device_mesh, file=sys.stderr)
  with jax.experimental.maps.Mesh(device_mesh, mesh_axis_names):
    return [
        ('simple_pjit',
         simple_fn.lower(jax.ShapeDtypeStruct((2, 2), jnp.dtype('float32')))),
        ('tuple_pjit',
         tuple_fn.lower((jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
                         jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))))),
        ('multi_return_pjit',
         multi_return_fn.lower(
             (jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
              jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))),
             jax.ShapeDtypeStruct((), jnp.dtype('float32')))),
        ('higher_arity_pjit',
         higher_arity_fn.lower(
             (jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
              jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))),
             jax.ShapeDtypeStruct((), jnp.dtype('float32')))),
        ('donate_arg_pjit',
         donate_arg_fn.lower(
             (jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
              jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))),
             jax.ShapeDtypeStruct((), jnp.dtype('float32')))),
    ]


if __name__ == '__main__':
  exporter.run(lower)
