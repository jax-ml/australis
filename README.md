# Australis

Australis is library that abstracts away the pjrt runtime API.
We want to provide a nicer C++ friendly API in order to decouple users.

To build the example:

```bash
# Because australis essentially links in jaxlib in order to access the XLA compiler,
# follow the latest prerequisite instructions for building jaxlib: https://jax.readthedocs.io/en/latest/developer.html.
# Also install jax with the proper jaxlib for your target architecture (precompiled is fine).
git clone https://github.com/jax-ml/australis.git
mkdir build
cd australis
wget https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-linux-x86_64
chmod +x bazel-5.1.1-linux-x86_64
sudo apt-get install protobuf-compiler
pip install . flax
./bazel-5.1.1-linux-x86_64 build --config=cuda --check_visibility=false //example:flax_example
```
