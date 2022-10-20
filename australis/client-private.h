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
#ifndef AUSTRALIS_CLIENT_PRIVATE_H_
#define AUSTRALIS_CLIENT_PRIVATE_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/platform/status.h"

namespace aux {
namespace internal {

template <typename T>
absl::StatusOr<T> ToAbslStatusOr(xla::StatusOr<T> s) {
  if (s.ok()) {
    return absl::StatusOr<T>(std::move(s).value());
  }
  return tensorflow::ToAbslStatus(s.status());
}

using FactoryFn = absl::StatusOr<std::shared_ptr<xla::PjRtClient>> (*)();

class BackendFactoryRegister {
 public:
  BackendFactoryRegister(const char* name, int priority, FactoryFn factory);
};

}  // namespace internal
}  // namespace aux

#endif  // AUSTRALIS_CLIENT_PRIVATE_H_
