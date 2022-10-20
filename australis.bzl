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

"""BUILD file macros to make it easy to use Australis."""

def australis(
        name,
        py_file = None,
        py_deps = [],
        args = [],
        cc_namespace = "jax::experimental",
        canonicalize_function_name = True):
    """Generates a C++ library by staging out functions defined in `name`.py.

    TODO(saeta): Include an example usage here? (Or point to alternate documentation.)

    Args:
      name: The name for this build target.
      py_file: (Optional.) The name of the Python file that defines the functions
        to be exported. If not specified `$(name).py` is assumed.
      py_deps: The list of py_libraries py_file depends upon.
      hardware_platform: The hardware platform to build the target for (one of: 'cpu',
        'gpu', 'tpu1', 'tpu4', 'tpu'). Default: 'cpu'. Unsupported (coming soon):
        'gpu'. :-) 'tpu1' means 1 chip (2 physical cores), 'tpu4' ('tpu' is a synonym)
        means 4 chips (8 physical cores).
      cc_namespace: The namespace for the C++ generated code.
      canonicalize_function_name: Modify the name of the function to follow typical
        C++ naming conventions.
    """

    if py_file == None:
        py_file = name + ".py"
    exporter_target_name = name + "_exporter"
    genrule_target_name = name + "_genrule"
    library_target_name = name + "_cc"

    header_name = name + ".h"
    impl_name = name + ".cc"
    cc_embed_impl_name = name + "_cc_embed.inc"

    additional_exporter_deps = []

    native.py_binary(
        name = exporter_target_name,
        srcs = [py_file],
        main = py_file,
        deps = py_deps + additional_exporter_deps,
    )

    _australis_generate(
        name = genrule_target_name,
        exporter_name = name,
        header_out = header_name,
        impl_out = impl_name,
        cc_namespace = cc_namespace,
        cc_embed_impl_out = cc_embed_impl_name,
        exporter = ":" + exporter_target_name,
        exporter_flags = args,
        canonicalize_function_name = canonicalize_function_name,
    )

    native.cc_library(
        name = library_target_name,
        hdrs = [header_name],
        srcs = [impl_name, cc_embed_impl_name],
        data = [],
        deps = [
            Label("//australis:australis_computation"),
        ],
    )

def _australis_generate_impl(ctx):
    header_out = ctx.outputs.header_out
    impl_out = ctx.outputs.impl_out
    cc_embed_impl_out = ctx.outputs.cc_embed_impl_out

    args = ctx.actions.args()
    args.add("--impl_name", impl_out)
    args.add("--cc_embed_impl_name", cc_embed_impl_out)
    args.add("--header_name", header_out)
    args.add("--cc_namespace", ctx.attr.cc_namespace)
    args.add("--canonicalize_function_name", ctx.attr.canonicalize_function_name)
    args.add("--name", ctx.attr.exporter_name)
    args.add_all(ctx.attr.exporter_flags)

    ctx.actions.run(
        mnemonic = "AustralisExporter",
        executable = ctx.executable.exporter,
        arguments = [args],
        inputs = [],
        outputs = [header_out, impl_out, cc_embed_impl_out],
    )

_australis_generate = rule(
    implementation = _australis_generate_impl,
    attrs = {
        "exporter": attr.label(executable = True, cfg = "exec"),
        "header_out": attr.output(),
        "impl_out": attr.output(),
        "cc_embed_impl_out": attr.output(),
        "cc_namespace": attr.string(),
        "exporter_flags": attr.string_list(),
        "canonicalize_function_name": attr.bool(),
        "exporter_name": attr.string(),
    },
)
