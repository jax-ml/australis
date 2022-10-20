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
/// A library for PyTrees.
///
/// Australis represents PyTrees in C++ with PTree's. These are designed to be
/// fast and ergonomic.

#ifndef AUSTRALIS_PETRI_H_
#define AUSTRALIS_PETRI_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "australis/australis.h"
#include "australis/client.h"

namespace aux {

namespace internal {
class PTreeStorage;
class AustralisComputation;
}  // namespace internal

namespace proto {
class PTreeProto;
};

// A data structure similar to a Python PyTree.
//
// This is a move-only type that owns the underlying data structures.
// You can think of this as a unique_ptr that holds one of the following PTree
// "kinds":
//   - Buffer: dense arrays of scalars.
//   - Tuple/List: an ordered collection of PTree's.
//   - Dict/Class: an ordered collection of name-PTree pairs.
//   - Optimized buffer: An optimized form to enable efficient dispatch.
//   - Empty: if data was moved out of `this`.
//
// This implementation was chosen (e.g. over a std::unique_ptr) to enable
// certain performance optimizations, and to maximize flexibility of library
// implementation evolution over time.
class PTree {
 public:
  // Constructors.
  template <typename... T>
  static PTree Tuple(T&&... args) {
    PTree values[] = {std::forward<T>(args)...};
    return Tuple(std::vector<PTree>{std::make_move_iterator(std::begin(values)),
                                    std::make_move_iterator(std::end(values))});
  }
  static PTree Tuple(std::vector<PTree> values);
  static PTree Tuple() { return Tuple(std::vector<PTree>()); }
  static PTree Dict(absl::btree_map<std::string, PTree> values);

  // Used in structure matching to represent any PTree.
  static PTree Wildcard();
  template <typename NativeT>
  static absl::StatusOr<PTree> BufferRN(absl::Span<const NativeT> values,
                                        absl::Span<int64_t const> dims,
                                        Device device) {
    return PTree::ShardedBuffer(DeviceArray::CreateRN(values, dims, device));
  }

  static PTree ShardedBuffer(DeviceArray array);
  static absl::StatusOr<PTree> ShardedBuffer(absl::StatusOr<DeviceArray> array);
  template <typename NativeT>
  static absl::StatusOr<PTree> ShardedBufferRN(
      std::vector<absl::Span<const NativeT>> values,
      absl::Span<int64_t const> dims, absl::Span<const Device> devices) {
    return PTree::ShardedBuffer(DeviceArray::CreateRN(values, dims, devices));
  }

  // Move-only type.
  PTree();
  PTree(PTree&& move);
  PTree& operator=(PTree&& move);
  PTree(const PTree& copy) = delete;
  PTree& operator=(const PTree& copy) = delete;
  ~PTree();

  // Returns the number of buffers (leaves) in the tree.
  size_t num_buffers() const;

  bool isEmpty() const { return !ptr_; }
  explicit operator bool() const { return !isEmpty(); }

  void swap(PTree& other) noexcept;
  friend void swap(PTree& x, PTree& y) noexcept { x.swap(y); }

  absl::StatusOr<Array> ToArray() const;

  // ToArrayAsync is not a good API; it should be changed!!
  void ToArrayAsync(
      std::function<void(absl::StatusOr<Array>)> on_ready) const;
  absl::StatusOr<std::vector<Array>> ToArrayShards() const;
  absl::StatusOr<absl::Span<const PTree>> Elements() const;

  std::string ToString() {
    std::string out;
    AppendToString(&out);
    return out;
  }

  absl::StatusOr<std::string> SerializeStructure();

  static absl::StatusOr<PTree> DeserializeStructure(std::string data);

  absl::Status FlattenTo(
      std::vector<const DeviceArray*>* flattened_output) const;

  static absl::Status FlattenMultipleTo(
      std::vector<const DeviceArray*>* flattened_output) {
    return absl::OkStatus();
  }

  template <typename... Ts>
  static absl::Status FlattenMultipleTo(
      std::vector<const DeviceArray*>* flattened_output, const PTree& arg,
      Ts&&... args) {
    auto s = arg.FlattenTo(flattened_output);
    if (!s.ok()) {
      return s;
    }
    return FlattenMultipleTo(flattened_output, std::forward<Ts>(args)...);
  }

  // Wrapping up outputs.
  absl::StatusOr<PTree> Unflatten(std::vector<DeviceArray> outputs);

  static absl::StatusOr<std::vector<PTree>> DestructureTuple(
      absl::StatusOr<PTree> ptree);

  // TODO(saeta): Implement functions to get stuff out of a PTree.
 private:
  friend class internal::PTreeStorage;
  friend class internal::AustralisComputation;

  explicit PTree(std::unique_ptr<internal::PTreeStorage> ptr);

  // Used by SerializeStructure / DeserializeStructure
  static PTree FromProto(const proto::PTreeProto& proto);
  absl::Status ToProto(proto::PTreeProto* output);

  void AppendToString(std::string* out) const;

  // If a large PTree will be used repeatedly, call this fn on the root PTree.
  //
  // TODO(saeta): Implement this optimization automatically on the first exec!
  absl::Status OptimizeFor(Device device);

  // Data storage types hidden behind a ptr.
  std::unique_ptr<internal::PTreeStorage> ptr_;
};

}  // namespace aux

#endif  // AUSTRALIS_PETRI_H_
