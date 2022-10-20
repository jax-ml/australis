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

#include <memory>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "australis/petri.pb.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace aux {

namespace {

class PTreeSequence {
 public:
  explicit PTreeSequence(std::vector<DeviceArray>&& arrays)
      : iter_(arrays.begin()), end_(arrays.end()) {}
  bool empty() const { return iter_ == end_; }
  PTree Consume() {
    auto result = PTree::ShardedBuffer(std::move(*iter_));
    ++iter_;
    return result;
  }

 private:
  std::vector<DeviceArray>::iterator iter_;
  std::vector<DeviceArray>::iterator end_;
};

}  // namespace

namespace internal {
// Storage classes for different PTree variations.
//
// Leverage compiler devirtualization for performance. :-)
class PTreeStorage {
 public:
  virtual ~PTreeStorage() {}

  virtual absl::Status FlattenTo(
      std::vector<const DeviceArray*>* flattened_output) const = 0;

  virtual size_t num_buffers() const = 0;

  static absl::StatusOr<std::vector<PTree>> DestructureTuple(
      absl::StatusOr<PTree> ptree);
  virtual absl::StatusOr<Array> ToArray() const = 0;
  virtual void ToArrayAsync(
      std::function<void(absl::StatusOr<Array>)> on_ready) const = 0;
  virtual absl::StatusOr<std::vector<Array>> ToArrayShards() const = 0;
  virtual absl::StatusOr<absl::Span<const PTree>> Elements() const = 0;

  std::string ToString() const {
    std::string out;
    AppendToString(&out);
    return out;
  }

  virtual void AppendToString(std::string* out) const = 0;

  virtual absl::Status ToProto(proto::PTreeProto* output) const = 0;

  virtual absl::StatusOr<std::vector<PTree>> DestructureTuple() {
    return absl::InvalidArgumentError(
        absl::StrCat(ToString(), " is not a tuple"));
  }

  virtual absl::StatusOr<PTree> Unflatten(PTreeSequence& seq) const = 0;

 protected:
  // Used by subclasses.
  PTreeStorage* Ptr(const PTree& tree) const { return tree.ptr_.get(); }
  void SetPtr(PTree& tree, std::unique_ptr<PTreeStorage> new_ptr) {
    tree.ptr_ = std::move(new_ptr);
  }
  PTree Make(std::unique_ptr<PTreeStorage> storage) const {
    return PTree(std::move(storage));
  }
};

namespace {

class PTreeTuple : public PTreeStorage {
 public:
  explicit PTreeTuple(std::vector<PTree> values) : elems_(std::move(values)) {}

  absl::Status FlattenTo(
      std::vector<const DeviceArray*>* flattened_output) const override {
    for (const auto& e : elems_) {
      auto s = Ptr(e)->FlattenTo(flattened_output);
      if (!s.ok()) return s;
    }
    return absl::OkStatus();
  }

  size_t num_buffers() const override {
    size_t sum = 0;
    for (const auto& e : elems_) {
      sum += Ptr(e)->num_buffers();
    }
    return sum;
  }

  absl::StatusOr<Array> ToArray() const override {
    return absl::InvalidArgumentError("Cannot get a literal out of a tuple!");
  }
  void ToArrayAsync(
      std::function<void(absl::StatusOr<Array>)> on_ready) const override {
    on_ready(
        absl::InvalidArgumentError("Cannot get a literal out of a tuple!"));
  }
  absl::StatusOr<std::vector<Array>> ToArrayShards() const override {
    return absl::InvalidArgumentError(
        "Cannot get literal shards out of a tuple!");
  }
  absl::StatusOr<absl::Span<const PTree>> Elements() const override {
    return elems_;
  }

  void AppendToString(std::string* out) const override {
    absl::StrAppend(out, "(");
    for (size_t i = 0; i < elems_.size(); ++i) {
      if (i != 0) absl::StrAppend(out, ", ");
      Ptr(elems_[i])->AppendToString(out);
    }
    absl::StrAppend(out, ")");
  }

  absl::Status ToProto(proto::PTreeProto* output) const override {
    auto* tuple_value = output->mutable_tuple_value();
    for (auto& elem : elems_) {
      auto s = (Ptr(elem)->ToProto(tuple_value->add_value()));
      if (!s.ok()) return s;
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<PTree>> DestructureTuple() override {
    return std::vector<PTree>(std::move(elems_));
  }

  absl::StatusOr<PTree> Unflatten(PTreeSequence& seq) const override {
    std::vector<PTree> out;
    out.reserve(elems_.size());
    for (const auto& elem : elems_) {
      auto tmp = Ptr(elem)->Unflatten(seq);
      if (!tmp.ok()) return tmp.status();
      out.push_back(*std::move(tmp));
    }
    return PTree::Tuple(std::move(out));
  }

 private:
  std::vector<PTree> elems_;
};

class PTreeDict : public PTreeStorage {
 public:
  explicit PTreeDict(absl::btree_map<std::string, PTree> values)
      : elems_(std::move(values)) {}

  absl::Status FlattenTo(
      std::vector<const DeviceArray*>* flattened_output) const override {
    for (const auto& e : elems_) {
      auto s = (Ptr(e.second)->FlattenTo(flattened_output));
      if (!s.ok()) return s;
    }
    return absl::OkStatus();
  }

  size_t num_buffers() const override {
    size_t sum = 0;
    for (const auto& e : elems_) {
      sum += Ptr(e.second)->num_buffers();
    }
    return sum;
  }

  absl::StatusOr<Array> ToArray() const override {
    return absl::InvalidArgumentError("Cannot get a literal out of a dict!");
  }

  void ToArrayAsync(
      std::function<void(absl::StatusOr<Array>)> on_ready) const override {
    on_ready(absl::InvalidArgumentError("Cannot get a literal out of a dict!"));
  }

  absl::StatusOr<std::vector<Array>> ToArrayShards() const override {
    return absl::InvalidArgumentError(
        "Cannot get literal shards out of a dict!");
  }

  absl::StatusOr<absl::Span<const PTree>> Elements() const override {
    return absl::InvalidArgumentError("Cannot get elements out of a dict!");
  }

  void AppendToString(std::string* out) const override {
    absl::StrAppend(out, "{");
    size_t i = 0;
    for (auto& elem : elems_) {
      if (i != 0) absl::StrAppend(out, ", ");
      absl::StrAppend(out, "\"", absl::CEscape(elem.first), "\": ");
      Ptr(elem.second)->AppendToString(out);
      ++i;
    }
    absl::StrAppend(out, "}");
  }

  absl::Status ToProto(proto::PTreeProto* output) const override {
    auto* dict_value = output->mutable_dict_value();
    for (auto& elem : elems_) {
      auto s = (Ptr(elem.second)
                    ->ToProto(&(*dict_value->mutable_value())[elem.first]));
      if (!s.ok()) return s;
    }
    return absl::OkStatus();
  }

  absl::StatusOr<PTree> Unflatten(PTreeSequence& seq) const override {
    absl::btree_map<std::string, PTree> result;
    for (auto& elem : elems_) {
      auto tmp = Ptr(elem.second)->Unflatten(seq);
      if (!tmp.ok()) return tmp.status();
      result.emplace(elem.first, *std::move(tmp));
    }
    return PTree::Dict(std::move(result));
  }

 private:
  absl::btree_map<std::string, PTree> elems_;
};

class PTreeDeviceArray : public PTreeStorage {
 public:
  explicit PTreeDeviceArray(DeviceArray data) : data_(std::move(data)) {}

  absl::Status FlattenTo(
      std::vector<const DeviceArray*>* flattened_output) const override {
    flattened_output->push_back(&data_);
    return absl::OkStatus();
  }

  size_t num_buffers() const override { return data_.size(); }

  absl::StatusOr<Array> ToArray() const override {
    if (data_.size() != 1)
      return absl::InvalidArgumentError(
          "Cannot get a literal out of a sharded buffer!");
    auto result = data_.ToArrays();
    if (!result.ok()) return result.status();
    return std::move((*result)[0]);
  }
  void ToArrayAsync(
      std::function<void(absl::StatusOr<Array>)> on_ready) const override {
    if (data_.size() != 1) {
      on_ready(absl::InvalidArgumentError(
          "Cannot get a literal out of a sharded buffer!"));
    }
    data_.ToArrayAsync(0, on_ready);
  }

  absl::StatusOr<std::vector<Array>> ToArrayShards() const override {
    auto result = data_.ToArrays();
    if (!result.ok()) return result.status();
    return std::vector<Array>(std::make_move_iterator(std::begin(*result)),
                              std::make_move_iterator(std::end(*result)));
  }

  absl::StatusOr<absl::Span<const PTree>> Elements() const override {
    return absl::InvalidArgumentError("Cannot get elements out of a Buffer!");
  }

  void AppendToString(std::string* out) const override {
    absl::StrAppend(out, "Buffer(",
                    data_.buffers()[0]->on_device_shape().ToString(), ")");
  }

  absl::Status ToProto(proto::PTreeProto* output) const override {
    return absl::InvalidArgumentError("Cannot convert a Buffer to a proto.");
  }

  absl::StatusOr<PTree> Unflatten(PTreeSequence& seq) const override {
    return absl::InvalidArgumentError("Cannot unflatten Buffer.");
  }

 private:
  DeviceArray data_;
};

class PTreeWildcard : public PTreeStorage {
 public:
  absl::StatusOr<absl::Span<const PTree>> Elements() const override {
    return absl::InvalidArgumentError("Cannot get elements out of a wildcard!");
  }
  absl::StatusOr<Array> ToArray() const override {
    return absl::InvalidArgumentError(
        "Cannot get a literal out of a wildcard!");
  }
  void ToArrayAsync(
      std::function<void(absl::StatusOr<Array>)> on_ready) const override {
    on_ready(
        absl::InvalidArgumentError("Cannot get a literal out of a wildcard!"));
  }
  size_t num_buffers() const override { return 0; }
  absl::StatusOr<std::vector<Array>> ToArrayShards() const override {
    return absl::InvalidArgumentError("Cannot flatten a wildcard!");
  }
  absl::Status FlattenTo(
      std::vector<const DeviceArray*>* flattened_output) const override {
    return absl::InvalidArgumentError(
        "Cannot flatten a Wildcard (are you trying to pass a wildcard)");
  }
  void AppendToString(std::string* out) const override {
    absl::StrAppend(out, "*");
  }
  absl::Status ToProto(proto::PTreeProto* output) const override {
    output->mutable_leaf_value();
    return absl::OkStatus();
  }
  absl::StatusOr<PTree> Unflatten(PTreeSequence& seq) const override {
    if (seq.empty()) {
      return absl::InvalidArgumentError(
          "Wrong number of arguments while unflattening.");
    }
    return seq.Consume();
  }
};

}  // namespace
}  // namespace internal

PTree PTree::Tuple(std::vector<PTree> values) {
  return PTree(absl::make_unique<internal::PTreeTuple>(std::move(values)));
}

PTree PTree::Dict(absl::btree_map<std::string, PTree> values) {
  return PTree(absl::make_unique<internal::PTreeDict>(std::move(values)));
}

PTree PTree::Wildcard() {
  return PTree(absl::make_unique<internal::PTreeWildcard>());
}

PTree PTree::ShardedBuffer(DeviceArray array) {
  return PTree(absl::make_unique<internal::PTreeDeviceArray>(std::move(array)));
}

absl::StatusOr<PTree> PTree::ShardedBuffer(absl::StatusOr<DeviceArray> array) {
  if (!array.ok()) return array.status();
  return ShardedBuffer(std::move(array).value());
}

PTree::PTree(PTree&& move) : ptr_(std::move(move.ptr_)) {}

PTree& PTree::operator=(PTree&& move) {
  ptr_ = std::move(move.ptr_);
  return *this;
}

PTree::PTree() {}
PTree::~PTree() {}

size_t PTree::num_buffers() const {
  if (ptr_) {
    return ptr_->num_buffers();
  }
  return 0;
}

void PTree::swap(PTree& other) noexcept { std::swap(ptr_, other.ptr_); }

absl::StatusOr<Array> PTree::ToArray() const {
  if (isEmpty()) {
    return absl::UnimplementedError("ToArray() called on an invalid PTree.");
  }
  return ptr_->ToArray();
}

void PTree::ToArrayAsync(
    std::function<void(absl::StatusOr<Array>)> on_ready) const {
  return ptr_->ToArrayAsync(std::move(on_ready));
}

absl::StatusOr<std::vector<Array>> PTree::ToArrayShards() const {
  if (isEmpty()) {
    return absl::UnimplementedError(
        "ToArrayShards() called on an invalid PTree.");
  }
  return ptr_->ToArrayShards();
}

absl::StatusOr<absl::Span<const PTree>> PTree::Elements() const {
  if (isEmpty()) {
    return absl::UnimplementedError("Elements() called on an invalid PTree.");
  }
  return ptr_->Elements();
}

PTree::PTree(std::unique_ptr<internal::PTreeStorage> ptr)
    : ptr_(std::move(ptr)) {}

absl::Status PTree::FlattenTo(
    std::vector<const DeviceArray*>* flattened_output) const {
  if (isEmpty()) {
    return absl::UnimplementedError("FlattenTo() called on an invalid PTree.");
  }
  return ptr_->FlattenTo(flattened_output);
}

absl::Status PTree::OptimizeFor(Device device) {
  return absl::UnimplementedError("OptimizeFor not yet implemented. Sorry!");
}

void PTree::AppendToString(std::string* out) const {
  if (isEmpty()) {
    absl::StrAppend(out, "PTree()");
  } else {
    ptr_->AppendToString(out);
  }
}

absl::Status PTree::ToProto(proto::PTreeProto* output) {
  if (isEmpty()) {
    return absl::UnimplementedError("ToProto() called on an invalid PTree.");
  }
  return ptr_->ToProto(output);
}

absl::StatusOr<PTree> PTree::Unflatten(std::vector<DeviceArray> outputs) {
  PTreeSequence seq(std::move(outputs));
  auto result = ptr_->Unflatten(seq);
  if (!seq.empty()) {
    return absl::InvalidArgumentError(
        "Wrong number of arguments while unflattening.");
  }
  return result;
}

PTree PTree::FromProto(const proto::PTreeProto& proto) {
  switch (proto.value_case()) {
    case proto::PTreeProto::kDictValue: {
      absl::btree_map<std::string, PTree> out;
      for (auto& value : proto.dict_value().value()) {
        out.emplace(value.first, PTree::FromProto(value.second));
      }
      return PTree(absl::make_unique<internal::PTreeDict>(std::move(out)));
    }
    case proto::PTreeProto::kTupleValue: {
      std::vector<PTree> out;
      for (auto& value : proto.tuple_value().value()) {
        out.emplace_back(PTree::FromProto(value));
      }
      return PTree(absl::make_unique<internal::PTreeTuple>(std::move(out)));
    }
    case proto::PTreeProto::kLeafValue: {
      return PTree(absl::make_unique<internal::PTreeWildcard>());
    }
    default:
      LOG(FATAL) << "Unreachable.";
  }
}

absl::StatusOr<std::string> PTree::SerializeStructure() {
  if (isEmpty()) {
    return absl::UnimplementedError(
        "SerializeStructure() called on an invalid PTree.");
  }
  proto::PTreeProto out;
  auto s = ToProto(&out);
  if (!s.ok()) return s;
  return out.SerializeAsString();
}

absl::StatusOr<PTree> PTree::DeserializeStructure(std::string data) {
  proto::PTreeProto ptree_proto;
  if (!ptree_proto.ParseFromString(data)) {
    return absl::InvalidArgumentError("Cannot parse tree structure");
  }
  return PTree::FromProto(ptree_proto);
}

absl::StatusOr<std::vector<PTree>> PTree::DestructureTuple(
    absl::StatusOr<PTree> ptree) {
  if (!ptree.ok()) return ptree.status();
  auto value = std::move(ptree).value();
  if (value.isEmpty()) {
    return absl::UnimplementedError(
        "DestructureTuple() called on an invalid PTree.");
  }
  return value.ptr_->DestructureTuple();
}

}  // namespace aux
