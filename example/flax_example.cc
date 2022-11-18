/* Copyright 2022 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "australis/australis.h"
#include "example/flax_jit.h"
#include "australis/petri.h"

namespace aux {
namespace {

absl::StatusOr<std::tuple<PTree, PTree>> Unpack2Tuple(
    absl::StatusOr<PTree> input) {
  auto tmp = *PTree::DestructureTuple(std::move(input));
  if (tmp.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat("Wrong size: ", tmp.size()));
  }
  return std::tuple<PTree, PTree>(std::move(tmp[0]), std::move(tmp[1]));
}

TEST(Flax, OptimizerStep) {
  auto client = *Client::GetDefault();
  auto dev = client.LocalDevices()[0];
  auto init_fn = *australis::test::FlaxInit::Load(client);
  auto optimizer_step_fn = *australis::test::FlaxOptimizerStep::Load(client);

  auto [params, opt_state] = *Unpack2Tuple(init_fn());

  const float inputs[] = {0, 1, 2, 3, 4, 5, 6, 7};
  auto x = *PTree::BufferRN<float>(inputs, {2, 4}, dev);
  std::tie(params, opt_state) =
      *Unpack2Tuple(optimizer_step_fn(params, opt_state, x));

  // TODO(parkers): Properly encode the flax weights structure (or an
  // approximation of it) into the new executable proto type.
  EXPECT_EQ(
      "((((Buffer(f32[16]), Buffer(f32[4,16])), (Buffer(f32[16]), "
      "Buffer(f32[16,16])), (Buffer(f32[16]), Buffer(f32[16,16])))))",
      params.ToString());
}

}  // namespace
}  // namespace aux
