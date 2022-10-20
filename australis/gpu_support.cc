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
#include "australis/client-private.h"
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"

namespace aux {
namespace internal {
namespace {

BackendFactoryRegister _register_gpu(
    "gpu", 200, +[]() -> absl::StatusOr<std::shared_ptr<xla::PjRtClient>> {
      xla::GpuAllocatorConfig config;
      config.preallocate = false;
      return ToAbslStatusOr(xla::GetStreamExecutorGpuClient(
          true, config, nullptr, /*node_id=*/0));
    });

}  // namespace
}  // namespace internal
}  // namespace aux
