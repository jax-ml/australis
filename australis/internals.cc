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
#include "australis/internals.h"

#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace aux {
namespace internal {

xla::PrimitiveType PrimitiveTypeToXla(PrimitiveType type) {
#define CASE(type)          \
  case PrimitiveType::type: \
    return xla::PrimitiveType::type
  switch (type) {
    CASE(PRIMITIVE_TYPE_INVALID);
    CASE(PRED);
    CASE(S8);
    CASE(S16);
    CASE(S32);
    CASE(S64);
    CASE(U8);
    CASE(U16);
    CASE(U32);
    CASE(U64);
    CASE(F16);
    CASE(F32);
    CASE(BF16);
    CASE(F64);
    CASE(TUPLE);
    CASE(TOKEN);
  }
#undef CASE
}

PrimitiveType PrimitiveTypeFromXla(xla::PrimitiveType type) {
#define CASE(type)               \
  case xla::PrimitiveType::type: \
    return PrimitiveType::type
  switch (type) {
    CASE(PRIMITIVE_TYPE_INVALID);
    CASE(PRED);
    CASE(S8);
    CASE(S16);
    CASE(S32);
    CASE(S64);
    CASE(U8);
    CASE(U16);
    CASE(U32);
    CASE(U64);
    CASE(F16);
    CASE(F32);
    CASE(BF16);
    CASE(F64);
    CASE(TUPLE);
    CASE(TOKEN);
    default:
      return PrimitiveType::PRIMITIVE_TYPE_INVALID;
  }
#undef CASE
}

}  // namespace internal
}  // namespace aux
