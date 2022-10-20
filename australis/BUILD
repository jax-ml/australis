# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Australis is a low[ish]-level API for JAX.

load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")

licenses(["notice"])

package(
    default_visibility = [
        "//visibility:public",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "exporter",
    srcs = ["exporter.py"],
    deps = [
        ":executable_py_pb2",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

cc_library(
    name = "australis",
    srcs = ["australis.cc"],
    hdrs = ["australis.h"],
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
        "@org_tensorflow//third_party/eigen3",
    ],
)

cc_test(
    name = "australis_test",
    srcs = ["australis_test.cc"],
    deps = [
        ":australis",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "petri",
    srcs = ["petri.cc"],
    hdrs = ["petri.h"],
    deps = [
        ":australis",
        ":client",
        ":petri_cc_proto",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
    ],
)

cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":australis",
        ":client",
        ":cpu_support",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "petri_test",
    srcs = ["petri_test.cc"],
    deps = [
        ":australis",
        ":cpu_support",
        ":petri",
        "@com_google_absl//absl/container:btree",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "internals",
    srcs = ["internals.cc"],
    hdrs = ["internals.h"],
    deps = [
        ":australis",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "australis_computation",
    srcs = ["australis_computation.cc"],
    hdrs = ["australis_computation.h"],
    deps = [
        ":client",
        ":executable_cc_proto",
        ":petri",
        ":petri_cc_proto",
    ],
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = [
        "client.h",
        "client-private.h",
    ],
    deps = [
        ":australis",
        ":internals",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "cpu_support",
    srcs = ["cpu_support.cc"],
    deps = [
        ":client",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cloud_tpu_support",
    srcs = ["cloud_tpu_support.cc"],
    deps = [
        ":client",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:tpu_client",
        "@org_tensorflow//tensorflow/core/tpu:tpu_api_dlsym_initializer",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gpu_support",
    srcs = ["gpu_support.cc"],
    deps = [
        ":client",
        "@jax//jaxlib:gpu_kernels",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/service:gpu_plugin",
    ],
    alwayslink = 1,
)

proto_library(
    name = "petri_proto",
    srcs = ["petri.proto"],
)

proto_library(
    name = "executable_proto",
    srcs = ["executable.proto"],
    deps = [":petri_proto"],
)

py_proto_library(
    name = "petri_py_pb2",
    deps = [":petri_proto"],
)

cc_proto_library(
    name = "petri_cc_proto",
    deps = [":petri_proto"],
)

py_proto_library(
    name = "executable_py_pb2",
    deps = [":executable_proto"],
)

cc_proto_library(
    name = "executable_cc_proto",
    deps = [":executable_proto"],
)
