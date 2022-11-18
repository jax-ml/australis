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
#include <utility>

#include "base/init_google.h"
#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "australis/australis.h"
#include "australis/petri.h"
#include "australis/tests/pjit_test_fns.h"

namespace {

using PrimitiveType = aux::PrimitiveType;
using aux::Client;

const float inputs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

TEST(Pjit, SimplePjit) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::SimplePjit::Load(client));
  auto devices = client.LocalDevices();
  ASSERT_OK_AND_ASSIGN(auto tree, aux::PTree::ShardedBufferRN<float>(
                                      {inputs, inputs}, {1, 3}, devices));

  ASSERT_OK_AND_ASSIGN(auto result, computation(tree));
  ASSERT_EQ(2, result.num_buffers());
  ASSERT_OK_AND_ASSIGN(auto literals, result.ToArrayShards());
  ASSERT_EQ(2, literals.size());

  EXPECT_EQ(2, literals[0].data<float>()[0]);
  EXPECT_EQ(4, literals[0].data<float>()[1]);

  EXPECT_EQ(2, literals[1].data<float>()[0]);
  EXPECT_EQ(4, literals[1].data<float>()[1]);
}

TEST(Pjit, TuplePjit) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::TuplePjit::Load(client));
  auto devices = client.LocalDevices();
  ASSERT_OK_AND_ASSIGN(auto lhs, aux::PTree::ShardedBufferRN<float>(
                                     {inputs, inputs}, {1, 3}, devices));
  ASSERT_OK_AND_ASSIGN(auto rhs, aux::PTree::ShardedBufferRN<float>(
                                     {inputs, inputs}, {1, 3}, devices));
  auto tree = aux::PTree::Tuple(std::move(lhs), std::move(rhs));

  ASSERT_OK_AND_ASSIGN(auto result, computation(tree));
  ASSERT_EQ(2, result.num_buffers());
  ASSERT_OK_AND_ASSIGN(auto literals, result.ToArrayShards());

  EXPECT_EQ(2, literals[0].data<float>()[0]);
  EXPECT_EQ(4, literals[0].data<float>()[1]);
  EXPECT_EQ(6, literals[0].data<float>()[2]);

  EXPECT_EQ(2, literals[1].data<float>()[0]);
  EXPECT_EQ(4, literals[1].data<float>()[1]);
  EXPECT_EQ(6, literals[1].data<float>()[2]);
}

TEST(Pjit, HigherArityPjit) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::HigherArityPjit::Load(client));
  auto devices = client.LocalDevices();
  ASSERT_OK_AND_ASSIGN(auto lhs, aux::PTree::ShardedBufferRN<float>(
                                     {inputs, inputs}, {1, 3}, devices));
  ASSERT_OK_AND_ASSIGN(auto rhs, aux::PTree::ShardedBufferRN<float>(
                                     {inputs, inputs}, {1, 3}, devices));
  auto tuple = aux::PTree::Tuple(std::move(lhs), std::move(rhs));
  ASSERT_OK_AND_ASSIGN(
      auto y, aux::PTree::ShardedBufferRN<float>({{4}, {5}}, {}, devices));

  ASSERT_OK_AND_ASSIGN(auto result, computation(tuple, y));
  ASSERT_EQ(2, result.num_buffers());
  ASSERT_OK_AND_ASSIGN(auto literals, result.ToArrayShards());

  EXPECT_EQ(3, literals[0].data<float>().size());
  EXPECT_EQ(5, literals[0].data<float>()[0]);
  EXPECT_EQ(10, literals[0].data<float>()[1]);
  EXPECT_EQ(15, literals[0].data<float>()[2]);

  EXPECT_EQ(3, literals[1].data<float>().size());
  EXPECT_EQ(6, literals[1].data<float>()[0]);
  EXPECT_EQ(12, literals[1].data<float>()[1]);
  EXPECT_EQ(18, literals[1].data<float>()[2]);
}

TEST(Pjit, MultiReturn) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::MultiReturnPjit::Load(client));
  auto devices = client.LocalDevices();

  ASSERT_OK_AND_ASSIGN(auto lhs, aux::PTree::ShardedBufferRN<float>(
                                     {inputs, inputs}, {1, 3}, devices));
  ASSERT_OK_AND_ASSIGN(auto rhs, aux::PTree::ShardedBufferRN<float>(
                                     {inputs, inputs}, {1, 3}, devices));
  auto tuple = aux::PTree::Tuple(std::move(lhs), std::move(rhs));
  ASSERT_OK_AND_ASSIGN(
      auto y, aux::PTree::ShardedBufferRN<float>({{4}, {5}}, {}, devices));

  ASSERT_OK_AND_ASSIGN(auto results,
                       aux::PTree::DestructureTuple(computation(tuple, y)));
  ASSERT_EQ(results.size(), 3);

  EXPECT_EQ(3, results[0].Elements()->size());
  // TODO(saeta): check results[0].

  ASSERT_OK_AND_ASSIGN(auto y_plus_one, results[1].ToArrayShards());
  EXPECT_EQ(1, y_plus_one[0].data<float>().size());
  EXPECT_EQ(1, y_plus_one[1].data<float>().size());
  EXPECT_EQ(5, y_plus_one[0].data<float>()[0]);
  EXPECT_EQ(6, y_plus_one[1].data<float>()[0]);

  EXPECT_EQ(2, results[2].Elements()->size());
  // TODO(saeta): Check x.
}

TEST(Pjit, DonateArgPjit) {
  ASSERT_OK_AND_ASSIGN(auto client, Client::GetDefault());
  ASSERT_OK_AND_ASSIGN(auto computation,
                       australis::test::DonateArgPjit::Load(client));
  auto devices = client.LocalDevices();
  ASSERT_OK_AND_ASSIGN(auto lhs, aux::PTree::ShardedBufferRN<float>(
                                     {inputs, inputs}, {1, 3}, devices));
  ASSERT_OK_AND_ASSIGN(auto rhs, aux::PTree::ShardedBufferRN<float>(
                                     {inputs, inputs}, {1, 3}, devices));
  auto tuple = aux::PTree::Tuple(std::move(lhs), std::move(rhs));
  ASSERT_OK_AND_ASSIGN(
      auto y, aux::PTree::ShardedBufferRN<float>({{4}, {5}}, {}, devices));

  ASSERT_OK_AND_ASSIGN(auto result, computation(std::move(tuple), y));
  ASSERT_EQ(4, result.num_buffers());  // A tuple with 2 shards.

  ASSERT_OK_AND_ASSIGN(auto elements, result.Elements());
  ASSERT_EQ(2, elements.size());
  ASSERT_OK_AND_ASSIGN(auto lit0, elements[0].ToArrayShards());

  // Tuple element 0, shard 0
  EXPECT_EQ(3, lit0[0].data<float>().size());
  EXPECT_EQ(5, lit0[0].data<float>()[0]);
  EXPECT_EQ(10, lit0[0].data<float>()[1]);
  EXPECT_EQ(15, lit0[0].data<float>()[2]);

  // Tuple element 0, shard 1
  EXPECT_EQ(3, lit0[1].data<float>().size());
  EXPECT_EQ(6, lit0[1].data<float>()[0]);
  EXPECT_EQ(12, lit0[1].data<float>()[1]);
  EXPECT_EQ(18, lit0[1].data<float>()[2]);

  ASSERT_OK_AND_ASSIGN(auto lit1, elements[0].ToArrayShards());

  // Tuple element 1, shard 0
  EXPECT_EQ(3, lit1[0].data<float>().size());
  EXPECT_EQ(5, lit1[0].data<float>()[0]);
  EXPECT_EQ(10, lit1[0].data<float>()[1]);
  EXPECT_EQ(15, lit1[0].data<float>()[2]);

  // Tuple element 1, shard 1
  EXPECT_EQ(3, lit1[1].data<float>().size());
  EXPECT_EQ(6, lit1[1].data<float>()[0]);
  EXPECT_EQ(12, lit1[1].data<float>()[1]);
  EXPECT_EQ(18, lit1[1].data<float>()[2]);
}

}  // namespace
