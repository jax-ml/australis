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
"""Stage out a simple jitted function."""

import functools

import jax
import jax.numpy as jnp
from jax.stages import Lowered
from jax.experimental.australis import exporter


@jax.jit
def tuple_fn(x):
  return x[0] + x[1] * 2


@jax.jit
def multi_return(x, y):
  return (x[0] + x[1] * y, x[0], x[1]), y + 1, x


@jax.jit
def higher_arity(x, y):
  return x[0] + x[1] * y


@functools.partial(jax.jit, donate_argnums=1)
def donate_arg(x, y):
  return jnp.sum(x[0] + x[1] * y)


@jax.jit
def many_results():
  return tuple(jnp.arange(800))


@jax.jit
def many_args(args):
  carry = args[0]
  for arg in args[1:]:
    carry = carry + arg
  return carry


@jax.jit
def bfloat16_fn(x):
  return x * x


def lower() -> Lowered:
  many_args_args = jax.eval_shape(many_results)

  tmp = many_args.lower(many_args_args)
  return [
      ('bfloat16_jit',
       bfloat16_fn.lower(jax.ShapeDtypeStruct((4,), jnp.dtype('bfloat16')))),
      ('tuple_jit',
       tuple_fn.lower((jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
                       jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))))),
      ('multi_return_jit',
       multi_return.lower((jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
                           jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))),
                          jax.ShapeDtypeStruct((), jnp.dtype('float32')))),
      ('higher_arity_jit',
       higher_arity.lower((jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
                           jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))),
                          jax.ShapeDtypeStruct((), jnp.dtype('float32')))),
      ('donate_arg_jit',
       donate_arg.lower((jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
                         jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))),
                        jax.ShapeDtypeStruct((), jnp.dtype('float32')))),
      ('many_results_jit', many_results.lower()),
      ('many_args_jit', many_args.lower(many_args_args)),
  ]


if __name__ == '__main__':
  exporter.run(lower)
