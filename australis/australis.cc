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
#include "australis/australis.h"

#include <sstream>
#include <string>
#include <string_view>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/logging.h"

namespace aux {

std::string_view PrimitiveTypeToString(PrimitiveType type) {
  switch (type) {
#define TYPE_NAME(tag)             \
  case PrimitiveType::tag: \
    return #tag
    TYPE_NAME(PRIMITIVE_TYPE_INVALID);
    TYPE_NAME(PRED);
    TYPE_NAME(S8);
    TYPE_NAME(S16);
    TYPE_NAME(S32);
    TYPE_NAME(S64);
    TYPE_NAME(U8);
    TYPE_NAME(U16);
    TYPE_NAME(U32);
    TYPE_NAME(U64);
    TYPE_NAME(F16);
    TYPE_NAME(F32);
    TYPE_NAME(BF16);
    TYPE_NAME(F64);
    TYPE_NAME(TUPLE);
    TYPE_NAME(TOKEN);
#undef TYPE_NAME
  }
}

std::ostream& operator<<(std::ostream& os, PrimitiveType type) {
  return os << PrimitiveTypeToString(type);
}

std::string ShapedArray::ToString() const {
  std::string out;
  absl::StrAppend(&out, PrimitiveTypeToString(dtype), "[");

  bool started = false;
  for (int64_t idx : shape) {
    if (started) {
      absl::StrAppend(&out, ",", idx);
    } else {
      absl::StrAppend(&out, idx);
    }
    started = true;
  }
  absl::StrAppend(&out, "]");
  return out;
}

int64_t BitWidth(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::PRED:
      return 1;
    case PrimitiveType::S8:
    case PrimitiveType::U8:
      return 8;

    case PrimitiveType::S16:
    case PrimitiveType::U16:
    case PrimitiveType::F16:
    case PrimitiveType::BF16:
      return 16;

    case PrimitiveType::U32:
    case PrimitiveType::S32:
    case PrimitiveType::F32:
      return 32;

    case PrimitiveType::U64:
    case PrimitiveType::S64:
    case PrimitiveType::F64:
      return 64;
    default:
      LOG(FATAL) << "Unhandled primitive type " << type;
  }
}

int64_t ShapedArray::element_count() const {
  int64_t out = 1;
  for (int64_t dim : shape) {
    out *= dim;
  }
  return out;
}

int64_t ByteWidth(PrimitiveType type) { return (BitWidth(type) + 7) / 8; }

absl::InlinedVector<int64_t, 2> Array::DefaultByteStrides(
    const ShapedArray& aval,
    absl::optional<absl::Span<int64_t const>> byte_strides) {
  int64_t stride = ByteWidth(aval.dtype);
  absl::InlinedVector<int64_t, 2> result(aval.shape.size());
  for (size_t i = aval.shape.size(); i > 0;) {
    --i;
    result[i] = stride;
    stride *= aval.shape[i];
  }
  return result;
}

namespace {

template <typename cpp_type>
void AppendRangeTyped(std::string* out, const void* data, int64_t byte_stride,
                      int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    if (i != 0) absl::StrAppend(out, ", ");
    absl::StrAppend(out, *reinterpret_cast<const cpp_type*>(data));
    data = reinterpret_cast<const char*>(data) + byte_stride;
  }
}

template <>
void AppendRangeTyped<bfloat16>(std::string* out, const void* data,
                                int64_t byte_stride, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    if (i != 0) absl::StrAppend(out, ", ");
    absl::StrAppend(
        out, static_cast<float>(*reinterpret_cast<const bfloat16*>(data)));
    data = reinterpret_cast<const char*>(data) + byte_stride;
  }
}

void AppendRange(std::string* out, PrimitiveType type, const void* data,
                 int64_t byte_stride, int64_t count) {
  switch (type) {
#define DEFINE_DTYPE_PRINTER(cpp_type, enum_tag) \
  case PrimitiveType::enum_tag:                  \
    return AppendRangeTyped<cpp_type>(out, data, byte_stride, count);
    DEFINE_DTYPE_PRINTER(bool, PRED);
    DEFINE_DTYPE_PRINTER(int8_t, S8);
    DEFINE_DTYPE_PRINTER(uint8_t, U8);
    DEFINE_DTYPE_PRINTER(int16_t, S16);
    DEFINE_DTYPE_PRINTER(uint16_t, U16);
    DEFINE_DTYPE_PRINTER(bfloat16, BF16);
    DEFINE_DTYPE_PRINTER(int32_t, S32);
    DEFINE_DTYPE_PRINTER(uint32_t, U32);
    DEFINE_DTYPE_PRINTER(int64_t, S64);
    DEFINE_DTYPE_PRINTER(uint64_t, U64);
    DEFINE_DTYPE_PRINTER(double, F64);
    DEFINE_DTYPE_PRINTER(float, F32);
#undef DEFINE_DTYPE_PRINTER
    default:
      absl::StrAppend(out, "??");
  }
}

void AppendIndent(std::string* out, size_t indent) {
  for (; indent--;) {
    absl::StrAppend(out, "  ");
  }
}

void AppendRecursive(std::string* out, PrimitiveType type, const void* data,
                     size_t indent, absl::Span<int64_t const> byte_strides,
                     absl::Span<int64_t const> counts) {
  if (byte_strides.empty()) {
    AppendRange(out, type, data, 0, 1);
  } else if (byte_strides.size() == 1) {
    absl::StrAppend(out, "{");
    AppendRange(out, type, data, byte_strides[0], counts[0]);
    absl::StrAppend(out, "}");
  } else {
    absl::StrAppend(out, "{\n");
    for (size_t i = 0; i < counts[0]; ++i) {
      AppendIndent(out, indent + 1);
      AppendRecursive(out, type, data, indent + 1, byte_strides.subspan(1),
                      counts.subspan(1));
      data = reinterpret_cast<const char*>(data) + byte_strides[0];
      if (i + 1 != counts[0]) {
        absl::StrAppend(out, ",\n");
      } else {
        absl::StrAppend(out, "\n");
      }
    }
    AppendIndent(out, indent);
    absl::StrAppend(out, "}");
  }
}

}  // namespace

std::string Array::ToString() const {
  std::string out = aval().ToString();
  absl::StrAppend(&out, " ");
  AppendRecursive(&out, aval_.dtype, raw_data(), 0, byte_strides(),
                  aval().shape);
  return out;
}

Array Array::operator[](size_t idx) const {
  auto new_shape = absl::Span<int64_t const>(aval_.shape).subspan(1);
  return Array(
      {aval_.dtype, Shape(new_shape.begin(), new_shape.end())},
      reinterpret_cast<const char*>(raw_data()) + idx * byte_strides_[0],
      byte_strides().subspan(1), owner_);
}

}  // namespace aux
