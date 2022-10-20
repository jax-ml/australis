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
#include <iostream>
#include <optional>

#include "base/init_google.h"
#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "australis/australis.h"
#include "australis/petri.h"
#include "australis/tests/jit_test_fns.h"

namespace {

using PrimitiveType = aux::PrimitiveType;
using PTree = aux::PTree;
using aux::Client;

const float inputs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

TEST(Jit, Tuple) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::TupleJit::Load(client));
  auto dev = client.LocalDevices()[0];
  ASSERT_OK_AND_ASSIGN(
      auto lhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  ASSERT_OK_AND_ASSIGN(
      auto rhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  auto tree = PTree::Tuple(std::move(lhs), std::move(rhs));

  ASSERT_OK_AND_ASSIGN(auto results, computation(tree));

  ASSERT_OK_AND_ASSIGN(auto lit, results.ToArray());

  EXPECT_EQ(3, lit.data<float>()[0]);
  EXPECT_EQ(6, lit.data<float>()[1]);
  EXPECT_EQ(9, lit.data<float>()[2]);
  EXPECT_EQ(12, lit.data<float>()[3]);
  EXPECT_EQ(15, lit.data<float>()[4]);
  EXPECT_EQ(18, lit.data<float>()[5]);
}

TEST(Jit, LargeNumberOfArguments) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto many_results_fn,
                       australis::test::ManyResultsJit::Load(client));
  ASSERT_OK_AND_ASSIGN(auto many_args_fn,
                       australis::test::ManyArgsJit::Load(client));

  for (size_t i = 0; i < 10; ++i) {
    auto results = *many_results_fn();
    int64_t result = (*results.Elements()).size();
    auto results2 = *many_args_fn(results);
    auto lit = *results2.ToArray();
    EXPECT_EQ(result * (result - 1) / 2, lit.data<int32_t>()[0]);
  }
}

TEST(Jit, BFloat16) {
  using aux::bfloat16;
  const bfloat16 inputs[] = {bfloat16(1.0f), bfloat16(2.0f), bfloat16(3.0f),
                             bfloat16(4.0f)};
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::Bfloat16Jit::Load(client));
  auto dev = client.LocalDevices()[0];

  ASSERT_OK_AND_ASSIGN(auto x, PTree::BufferRN<bfloat16>(inputs, {4}, dev));

  ASSERT_OK_AND_ASSIGN(auto lit, computation(x)->ToArray());

  EXPECT_EQ(1, lit.data<bfloat16>()[0]);
  EXPECT_EQ(4, lit.data<bfloat16>()[1]);
  EXPECT_EQ(9, lit.data<bfloat16>()[2]);
  EXPECT_EQ(16, lit.data<bfloat16>()[3]);
}

TEST(Jit, HigherArity) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::HigherArityJit::Load(client));
  auto dev = client.LocalDevices()[0];
  ASSERT_OK_AND_ASSIGN(
      auto lhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  ASSERT_OK_AND_ASSIGN(
      auto rhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  auto tuple = PTree::Tuple(std::move(lhs), std::move(rhs));
  ASSERT_OK_AND_ASSIGN(
      auto y,
      PTree::BufferRN<float>({3}, {}, dev));
  ASSERT_OK_AND_ASSIGN(auto results, computation(tuple, y));

  ASSERT_OK_AND_ASSIGN(auto lit, results.ToArray());

  EXPECT_EQ(4, lit.data<float>()[0]);
  EXPECT_EQ(8, lit.data<float>()[1]);
  EXPECT_EQ(12, lit.data<float>()[2]);
  EXPECT_EQ(16, lit.data<float>()[3]);
  EXPECT_EQ(20, lit.data<float>()[4]);
  EXPECT_EQ(24, lit.data<float>()[5]);
}

TEST(Jit, MultiReturn) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::MultiReturnJit::Load(client));
  auto dev = client.LocalDevices()[0];

  ASSERT_OK_AND_ASSIGN(
      auto lhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  ASSERT_OK_AND_ASSIGN(
      auto rhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  auto tuple = PTree::Tuple(std::move(lhs), std::move(rhs));
  ASSERT_OK_AND_ASSIGN(
      auto y,
      PTree::BufferRN<float>({3}, {}, dev));
  ASSERT_OK_AND_ASSIGN(auto results,
                       PTree::DestructureTuple(computation(tuple, y)));
  EXPECT_EQ(results.size(), 3);

  EXPECT_EQ(3, results[0].Elements()->size());
  // TODO(saeta): check results[0].
  ASSERT_OK_AND_ASSIGN(auto y_plus_one, results[1].ToArray());
  EXPECT_EQ(1, y_plus_one.data<float>().size());
  EXPECT_EQ(4, y_plus_one.data<float>()[0]);

  EXPECT_EQ(2, results[2].Elements()->size());
  // TODO(saeta): check x.
}

TEST(Jit, DonateArg) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::DonateArgJit::Load(client));
  auto dev = client.LocalDevices()[0];

  ASSERT_OK_AND_ASSIGN(
      auto lhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  ASSERT_OK_AND_ASSIGN(
      auto rhs,
      PTree::BufferRN<float>(inputs, {2, 3}, dev));
  auto tuple = PTree::Tuple(std::move(lhs), std::move(rhs));
  ASSERT_OK_AND_ASSIGN(
      auto y,
      PTree::BufferRN<float>({3}, {}, dev));

  ASSERT_OK_AND_ASSIGN(auto results, computation(tuple, std::move(y)));
  ASSERT_OK_AND_ASSIGN(auto lit, results.ToArray());

  EXPECT_EQ(1, lit.data<float>().size());
  EXPECT_EQ(84, lit.data<float>()[0]);
}

}  // namespace
