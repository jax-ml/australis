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
#include "australis/australis_computation.h"

#include <utility>

#include "google/protobuf/io/coded_stream.h"
#include "australis/executable.pb.h"
#include "australis/petri.pb.h"

namespace aux::internal {

namespace {

using google::protobuf::io::CodedInputStream;

struct ManualExecutableSpec {
  int64_t num_replicas = 1;
  int64_t num_partitions = 1;
  bool use_spmd_partitioning = false;
  bool tuple_args = false;
  PTree out_tree;
};

bool ReadFromStream(PTree& out, CodedInputStream& stream) {
  uint32_t limit_value;
  if (!stream.ReadLittleEndian32(&limit_value)) {
    return false;
  }
  auto limit = stream.PushLimit(sizeof(uint32_t) * limit_value);

  uint32_t kind;
  if (!stream.ReadLittleEndian32(&kind)) {
    return false;
  }
  if (kind == 0) {
    out = PTree::Wildcard();
  } else if (kind == 1) {
    std::vector<PTree> items;
    while (stream.BytesUntilLimit() > 0) {
      PTree tmp;
      if (!ReadFromStream(tmp, stream)) {
        return false;
      }
      items.push_back(std::move(tmp));
    }
    out = PTree::Tuple(std::move(items));
  } else {
    return false;
  }
  stream.PopLimit(limit);
  return true;
}

bool ReadFromStream(ManualExecutableSpec& out, CodedInputStream& stream) {
  uint32_t limit_value;
  if (!stream.ReadLittleEndian32(&limit_value)) {
    return false;
  }
  auto limit = stream.PushLimit(sizeof(uint32_t) * limit_value);

  while (stream.BytesUntilLimit() > 0) {
    uint32_t tag;
    if (!stream.ReadLittleEndian32(&tag)) {
      return false;
    }
    if (tag == 0) {
      uint32_t value;
      if (!stream.ReadLittleEndian32(&value)) {
        return false;
      }
      out.num_replicas = value;
      if (!stream.ReadLittleEndian32(&value)) {
        return false;
      }
      out.num_partitions = value;
      if (!stream.ReadLittleEndian32(&value)) {
        return false;
      }
      if (value == 1) {
        out.use_spmd_partitioning = true;
      }
      if (!stream.ReadLittleEndian32(&value)) {
        return false;
      }
      if (value == 1) {
        out.tuple_args = true;
      }
    } else if (tag == 1) {
      if (!ReadFromStream(out.out_tree, stream)) {
        return false;
      }
    } else {
      return false;
    }
  }
  stream.PopLimit(limit);
  return true;
}

}  // namespace

absl::StatusOr<AustralisComputation> AustralisComputation::LoadHlo(
    aux::Client client, std::string_view hlo_binary_text,
    std::string_view executable_spec, int64_t version) {
  if (version == 0) {
    proto::ExecutableSpec spec;
    if (!spec.ParsePartialFromArray(executable_spec.data(),
                                    executable_spec.size())) {
      return absl::Status(absl::StatusCode::kFailedPrecondition,
                          "Could not parse executable spec proto.");
    }
    auto devices = client.DefaultDeviceAssignment(spec.num_replicas(),
                                                  spec.num_partitions());
    if (!devices.ok()) return devices.status();

    auto executable = client.Compile(*devices, spec.use_spmd_partitioning(),
                                     spec.tuple_args(), hlo_binary_text);
    if (!executable.ok()) return executable.status();
    return AustralisComputation(*std::move(executable),
                                PTree::FromProto(spec.out_tree()));
  } else if (version == 1) {
    CodedInputStream stream(
        reinterpret_cast<const uint8_t*>(executable_spec.data()),
        executable_spec.size());
    ManualExecutableSpec spec;
    ReadFromStream(spec, stream);

    auto devices =
        client.DefaultDeviceAssignment(spec.num_replicas, spec.num_partitions);
    if (!devices.ok()) return devices.status();

    auto executable = client.Compile(*devices, spec.use_spmd_partitioning,
                                     spec.tuple_args, hlo_binary_text);
    if (!executable.ok()) return executable.status();
    return AustralisComputation(*std::move(executable),
                                std::move(spec.out_tree));
  } else {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        "Unsupported Computation version.");
  }
}

absl::StatusOr<PTree> AustralisComputation::ExecuteInternal(
    std::vector<const DeviceArray*> inputs) {
  auto device_results = executable_->Eval(absl::MakeSpan(inputs));
  if (!device_results.ok()) return device_results.status();
  return out_shape_.Unflatten(*std::move(device_results));
}

}  // namespace aux::internal
