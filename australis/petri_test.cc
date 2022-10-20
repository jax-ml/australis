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
#include "australis/petri.h"

#include <optional>
#include <utility>

#include "gtest/gtest.h"
#include "absl/container/btree_map.h"
#include "australis/australis.h"

namespace aux {
namespace {

TEST(PTreeTest, Constructs) {
  auto device = Client::GetDefault()->LocalDevices()[0];
  auto tree_prepared =
      *PTree::BufferRN<float>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3}, device);
  EXPECT_EQ(1, tree_prepared.num_buffers());

  auto tuple_tree = PTree::Tuple(std::move(tree_prepared));
  EXPECT_EQ(0, tree_prepared.num_buffers());
  EXPECT_EQ(1, tuple_tree.num_buffers());

  absl::btree_map<std::string, PTree> dict_input;
  dict_input.emplace("asdf", std::move(tuple_tree));
  EXPECT_EQ(0, tuple_tree.num_buffers());
  auto dict_tree = PTree::Dict(std::move(dict_input));
  EXPECT_EQ(1, dict_tree.num_buffers());
}

TEST(PTreeTest, Serialize) {
  absl::btree_map<std::string, PTree> dict_input;
  dict_input.emplace("key", PTree::Tuple(PTree::Wildcard()));

  auto dict_tree = PTree::Dict(std::move(dict_input));

  EXPECT_EQ(
      dict_tree.ToString(),
      PTree::DeserializeStructure(*dict_tree.SerializeStructure())->ToString());
}

TEST(PTreeTest, Unflatten) {
  auto device = Client::GetDefault()->LocalDevices()[0];
  absl::btree_map<std::string, PTree> dict_input;
  dict_input.emplace("key", PTree::Tuple(PTree::Wildcard()));
  dict_input.emplace(
      "key2", PTree::Tuple(PTree::Tuple(PTree::Wildcard(), PTree::Wildcard(),
                                        PTree::Wildcard()),
                           PTree::Tuple(PTree::Wildcard(), PTree::Wildcard())));

  auto pattern = PTree::Dict(std::move(dict_input));

  auto make_buffers = [device](size_t n) {
    std::vector<DeviceArray> buffers;
    for (size_t i = 0; i < n; ++i) {
      buffers.push_back(
          *DeviceArray::CreateRN<float>({3, 2, 3, 4}, {2, 2}, device));
    }
    return buffers;
  };

  for (size_t n : {0, 3, 6, 10}) {
    auto result = pattern.Unflatten(make_buffers(n));
    if (n == 6) {
      ASSERT_TRUE(result.ok()) << result.status();
    } else {
      EXPECT_FALSE(result.ok()) << result.status();
    }
  }
}

TEST(PTreeTest, InvalidPTrees) {
  PTree value;
  ASSERT_FALSE(value.Elements().ok());
  ASSERT_FALSE(value.SerializeStructure().ok());
  std::vector<const DeviceArray*> tmp;
  ASSERT_FALSE(value.FlattenTo(&tmp).ok());
  ASSERT_FALSE(value.ToArray().ok());
  ASSERT_FALSE(value.ToArrayShards().ok());
  ASSERT_FALSE(PTree::DestructureTuple(std::move(value)).ok());
}

}  // namespace
}  // namespace aux
