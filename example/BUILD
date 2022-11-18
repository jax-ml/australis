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

# Tests for australis.

load("@australis//:australis.bzl", "australis")

licenses(["notice"])

australis(
    name = "flax_jit",
    cc_namespace = "australis::test",
    py_deps = [],  # Internal flax deps
)

cc_test(
    name = "flax_example",
    srcs = ["flax_example.cc"],
    linkstatic = 1,
    deps = [
        ":flax_jit_cc",
        "@australis//australis",
        "@australis//australis:cpu_support",
        "@australis//australis:gpu_support",
        "@australis//australis:petri",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
