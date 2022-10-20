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
#ifndef AUSTRALIS_AUSTRALIS_H_
#define AUSTRALIS_AUSTRALIS_H_

#include <stdint.h>

#include <string>
#include <type_traits>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "third_party/eigen3/Eigen/Core"

namespace aux {

using bfloat16 = Eigen::bfloat16;

enum class PrimitiveType {
  PRIMITIVE_TYPE_INVALID = 0,
  PRED = 1,

  // Signed integral values of fixed width.
  S8 = 2,
  S16 = 3,
  S32 = 4,
  S64 = 5,

  // Unsigned integral values of fixed width.
  U8 = 6,
  U16 = 7,
  U32 = 8,
  U64 = 9,

  // Floating-point values of fixed width.
  //
  // Note: if f16s are not natively supported on the device, they will be
  // converted to f16 from f32 at arbirary points in the computation.
  F16 = 10,
  F32 = 11,
  // Truncated 16 bit floating-point format. This is similar to IEEE's 16 bit
  // floating-point format, but uses 1 bit for the sign, 8 bits for the exponent
  // and 7 bits for the mantissa.
  BF16 = 16,
  F64 = 12,

  // A tuple is a polymorphic sequence; e.g. a shape that holds different
  // sub-shapes. They are used for things like returning multiple values from a
  // computation; e.g. a computation that returns weights and biases may have a
  // signature that results in a tuple like (f32[784x2000], f32[2000])
  //
  // If a shape proto has the tuple element type, it may not have any entries
  // in the dimensions field.
  TUPLE = 13,

  // A token type threaded between side-effecting operations. Shapes of this
  // primitive type will have empty dimensions and tuple_shapes fields.
  TOKEN = 17,
};

template <typename NativeT>
PrimitiveType CppToPrimitiveType() {
  // Make the expression depend on the template parameter NativeT so
  // that this compile-time error only appears if this function is
  // instantiated with some concrete type that is not specialized
  // below.
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "Cannot map native type to primitive type.");
  return PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

#define DEFINE_CPP_TO_PRIMITIVE_MAPPING(cpp_type, enum_tag) \
  template <>                                               \
  inline PrimitiveType CppToPrimitiveType<cpp_type>() {     \
    return PrimitiveType::enum_tag;                         \
  }
DEFINE_CPP_TO_PRIMITIVE_MAPPING(bool, PRED);

DEFINE_CPP_TO_PRIMITIVE_MAPPING(int8_t, S8);
DEFINE_CPP_TO_PRIMITIVE_MAPPING(uint8_t, U8);
DEFINE_CPP_TO_PRIMITIVE_MAPPING(int16_t, S16);
DEFINE_CPP_TO_PRIMITIVE_MAPPING(uint16_t, U16);
DEFINE_CPP_TO_PRIMITIVE_MAPPING(int32_t, S32);
DEFINE_CPP_TO_PRIMITIVE_MAPPING(uint32_t, U32);
DEFINE_CPP_TO_PRIMITIVE_MAPPING(int64_t, S64);
DEFINE_CPP_TO_PRIMITIVE_MAPPING(uint64_t, U64);

DEFINE_CPP_TO_PRIMITIVE_MAPPING(double, F64);
DEFINE_CPP_TO_PRIMITIVE_MAPPING(float, F32);
DEFINE_CPP_TO_PRIMITIVE_MAPPING(bfloat16, BF16);

#undef DEFINE_CPP_TO_PRIMITIVE_MAPPING

using Shape = absl::InlinedVector<int64_t, 2>;

struct ShapedArray {
  PrimitiveType dtype;
  Shape shape;

  std::string ToString() const;
  int64_t element_count() const;
};

// Multidimensional array living in host memory with optional ownership.
// Any existing shared_ptr which owns `data` can be passed as `owner`
// which will get destroyed when all users have finished. `data` should
// be considered immutable until owner is destroyed.
class Array {
 public:
  Array(const ShapedArray& aval, const void* data,
        absl::optional<absl::Span<int64_t const>> byte_strides,
        std::shared_ptr<void> owner)
      : aval_(aval),
        byte_strides_(DefaultByteStrides(aval, byte_strides)),
        data_(data),
        owner_(owner) {}

  const ShapedArray& aval() const { return aval_; }
  absl::Span<int64_t const> byte_strides() const { return byte_strides_; }

  const void* raw_data() const { return data_; }

  template <typename T = void>
  absl::Span<const T> data() const {
    return {reinterpret_cast<const T*>(data_), aval_.element_count()};
  }

  // If unset, this should be considered a view and follow the liveness
  // rules of a string_view.
  const std::shared_ptr<void>& owner() const { return owner_; }

  template <typename NativeT>
  static Array CreateRN(const absl::Span<const NativeT> values,
                        absl::Span<int64_t const> dims,
                        std::shared_ptr<void> owner = nullptr) {
    return Array(
        {CppToPrimitiveType<NativeT>(), Shape(dims.begin(), dims.end())},
        values.data(), absl::nullopt, std::move(owner));
  }

  Array operator[](size_t idx) const;

  std::string ToString() const;

 private:
  absl::InlinedVector<int64_t, 2> DefaultByteStrides(
      const ShapedArray& aval,
      absl::optional<absl::Span<int64_t const>> byte_strides);

  ShapedArray aval_;
  absl::InlinedVector<int64_t, 2> byte_strides_;
  const void* data_;
  std::shared_ptr<void> owner_;
};

}  // namespace aux

#endif  // AUSTRALIS_AUSTRALIS_H_
