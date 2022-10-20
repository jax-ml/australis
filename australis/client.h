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
#ifndef AUSTRALIS_CLIENT_H_
#define AUSTRALIS_CLIENT_H_

#include <functional>
#include <memory>
#include <string_view>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "australis/australis.h"

namespace xla {
// TODO(parkers): Hide these somehow?.
class PjRtClient;
class PjRtDevice;
class PjRtLoadedExecutable;
class PjRtBuffer;
}  // namespace xla

namespace aux {

class Executable;
class Device;
class DeviceAssignment;

// A session for dispatching computations to an accelerator.
// Since `Client` objects are by default are globally constructed and
// live for the lifetime of the program, users should not need to worry
// about lifetime.
class Client {
 public:
  explicit Client(xla::PjRtClient* client) : client_(client) {}

  static absl::StatusOr<Client> Get(std::string_view name);
  static absl::StatusOr<Client> GetDefault();

  // Return only those devices local to this host.
  std::vector<Device> LocalDevices();

  absl::StatusOr<DeviceAssignment> DefaultDeviceAssignment(int num_replicas,
                                                           int num_partitions);

  // devices has a 'row-major' layout and must match num_replicas x
  // num_partitions.
  absl::StatusOr<DeviceAssignment> CreateDeviceAssignment(
      int num_replicas, int num_partitions, std::vector<Device> devices);

  // TODO(parkers): Add a new executable type.
  // Compiles a replicated executable with optional spmd partitioning.
  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const DeviceAssignment& devices, bool use_spmd_partitioning,
      bool tuple_args, std::string_view binary_proto);

  // Returns a string that identifies the platform (CPU/GPU/TPU).
  std::string_view platform_name() const;

 private:
  // TODO(parkers): Remove once all use-cases are fully wrapped.
  xla::PjRtClient* client() const { return client_; }

  xla::PjRtClient* client_;  // Not owned.
};

// A Device represents a single accelerator.
class Device {
 public:
  explicit Device(xla::PjRtDevice* device) : device_(device) {}

  Client client() const;

 private:
  // TODO(parkers): Remove once all use-cases are fully wrapped.
  xla::PjRtDevice* device() const { return device_; }

  friend class DeviceArray;
  friend class Client;
  xla::PjRtDevice* device_;  // Not owned.
};

class DeviceAssignment {
 private:
  friend Client;

  int num_replicas_;
  int num_partitions_;
  // 2D array Device[num_replicas, num_partitions]
  std::vector<Device> devices_;
};

class DeviceArray {
 public:
  explicit DeviceArray(
      absl::InlinedVector<std::unique_ptr<xla::PjRtBuffer>, 1> buffers);
  DeviceArray(DeviceArray&& other);
  DeviceArray& operator=(DeviceArray&& other);
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray();

  // Zero-copy version which will hold onto arrays until the values
  // are no longer needed (the resulting DeviceArray is destroyed, or depending
  // on the backend any compuation is complete). Because it is zero copy, this
  // function can return almost immediately.
  static absl::StatusOr<DeviceArray> Create(absl::Span<const Array> arrays,
                                            absl::Span<const Device> devices);

  static absl::StatusOr<DeviceArray> Create(
      absl::Span<const void* const> buffers, PrimitiveType type,
      absl::Span<int64_t const> dims,
      absl::optional<absl::Span<int64_t const>> byte_strides,
      absl::Span<const Device> devices);

  template <typename NativeT>
  static absl::StatusOr<DeviceArray> CreateRN(absl::Span<const NativeT> values,
                                              absl::Span<int64_t const> dims,
                                              Device device) {
    return Create({values.data()}, CppToPrimitiveType<NativeT>(), dims,
                  absl::nullopt, {device});
  }

  template <typename NativeT>
  static absl::StatusOr<DeviceArray> CreateRN(
      std::vector<absl::Span<const NativeT>> values,
      absl::Span<int64_t const> dims, absl::Span<const Device> devices) {
    std::vector<const void*> raw_values(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      raw_values[i] = values[i].data();
    }
    return Create(raw_values, CppToPrimitiveType<NativeT>(), dims,
                  absl::nullopt, devices);
  }

  absl::StatusOr<absl::InlinedVector<Array, 1>> ToArrays() const;

  void ToArrayAsync(
      size_t idx, std::function<void(absl::StatusOr<Array>)> on_ready) const;

  // TODO(parkers): Remove once all use-cases are fully wrapped.
  absl::InlinedVector<xla::PjRtBuffer*, 1> buffers() const;

  // TODO(parkers): Remove once all use-cases are fully wrapped.
  std::vector<std::unique_ptr<xla::PjRtBuffer>> ConsumeShards();

  size_t size() const { return buffers_.size(); }

 private:
  friend class Executable;

  DeviceArray();
  absl::InlinedVector<std::unique_ptr<xla::PjRtBuffer>, 1> buffers_;
};

// Represents a compiled computation that takes and returns a list of buffers.
// It is recommended to use typed wrappers which operate instead over PTrees.
class Executable {
 public:
  Executable(Executable&&) = default;
  Executable& operator=(Executable&&) = default;
  Executable(const Executable&) = delete;
  Executable& operator=(const Executable&) = delete;
  ~Executable();

  absl::StatusOr<std::vector<DeviceArray>> Eval(
      absl::Span<const DeviceArray*> args) const;

 private:
  friend class Client;
  explicit Executable(std::unique_ptr<xla::PjRtLoadedExecutable> executable);
  std::unique_ptr<xla::PjRtLoadedExecutable> executable_;
};

}  // namespace aux

#endif  // AUSTRALIS_CLIENT_H_
