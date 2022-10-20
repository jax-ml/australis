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
#ifndef AUSTRALIS_AUSTRALIS_COMPUTATION_H_
#define AUSTRALIS_AUSTRALIS_COMPUTATION_H_

#include <memory>
#include <string_view>
#include <utility>

#include "australis/client.h"
#include "australis/petri.h"

namespace aux {
namespace internal {

// Superclass for australis code-generated computations.
//
// DO NOT SUBCLASS ME IN HUMAN-EDITED SOURCE CODE!
class AustralisComputation {
 protected:
  static absl::StatusOr<AustralisComputation> LoadHlo(
      aux::Client client, std::string_view hlo_binary_text,
      std::string_view executable_spec, int64_t version);

  absl::StatusOr<PTree> ExecuteInternal(std::vector<const DeviceArray*> inputs);

 private:
  explicit AustralisComputation(std::unique_ptr<Executable> e, PTree out_shape)
      : executable_(std::move(e)), out_shape_(std::move(out_shape)) {}
  std::unique_ptr<aux::Executable> executable_;
  PTree out_shape_;
};

}  // namespace internal

template <typename... Args>
class TypedComputation : public internal::AustralisComputation {
 public:
  absl::StatusOr<aux::PTree> operator()(Args... args) {
    std::vector<const DeviceArray*> inputs;
    absl::Status s =
        aux::PTree::FlattenMultipleTo(&inputs, std::forward<Args>(args)...);
    if (!s.ok()) {
      return s;
    }
    return ExecuteInternal(std::move(inputs));
  }

 protected:
  TypedComputation(aux::internal::AustralisComputation computation)
      : AustralisComputation(std::move(computation)) {
    {}
  }
};

}  // namespace aux

#endif  // AUSTRALIS_AUSTRALIS_COMPUTATION_H_
