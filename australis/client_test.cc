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
#include "australis/client.h"

#include "gtest/gtest.h"
#include "australis/australis.h"

namespace aux {
namespace {

Array CreateFloatRange(size_t n, absl::Span<int64_t const> dims) {
  auto data = std::make_shared<std::vector<float>>();
  for (size_t i = 0; i < n; ++i) {
    data->push_back(float(i));
  }
  return Array::CreateRN<float>(*data, dims, data);
}

TEST(ArrayTest, CreateFromArray) {
  auto device = Client::GetDefault()->LocalDevices()[0];
  Array data =
      (*DeviceArray::Create({CreateFloatRange(12, {2, 2, 3})}, {device})
            ->ToArrays())[0];
  ASSERT_EQ(data.ToString(), CreateFloatRange(12, {2, 2, 3}).ToString());
}

}  // namespace
}  // namespace aux
