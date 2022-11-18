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
"""Stage out a flax model."""

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from jax.stages import Lowered
from australis import exporter


class Sequential(nn.Module):
  layers: Sequence[nn.Module]

  @nn.compact
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


def lower() -> Lowered:
  model = Sequential(layers=[nn.Dense(16), nn.Dense(16), nn.Dense(16)])

  tx = optax.adam(0.1)

  @jax.jit
  def init():
    init_rng = jax.random.PRNGKey(0)
    params = model.init(init_rng, jnp.ones((2, 4), dtype=jnp.float32))
    return params, tx.init(params)

  init_fn = init.lower()

  @jax.jit
  def optimizer_step(params, opt_state, x):

    def fwd(params):
      return model.apply(params, x).sum()

    grads = jax.grad(fwd)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

  params, opt_state = jax.eval_shape(init)
  optimizer_step_lowered = optimizer_step.lower(
      params, opt_state, jax.ShapeDtypeStruct((2, 4), jnp.float32))

  return [
      ("flax_init", init_fn),
      ("flax_optimizer_step", optimizer_step_lowered),
  ]


if __name__ == "__main__":
  exporter.run(lower)
