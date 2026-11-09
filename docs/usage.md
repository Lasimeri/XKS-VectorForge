# Build and usage

## Requirements

The native tools require C++20, CMake 3.20 or newer, and the OpenSSL crypto
library and headers. CUDA is optional for the experimental kernels. The
transitional compiler needs Python 3; inference additionally needs the packages
and compatible PyTorch CUDA installation in `requirements-inference.txt`.
No model downloads or external services are required for local execution.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For the optional CUDA library, set `-DXKSV_ENABLE_CUDA=ON` and an appropriate
`-DCMAKE_CUDA_ARCHITECTURES` for the installed GPU. Select an inference Python
interpreter with `XKSV_PYTHON=/path/to/python`. The default CTest suite is
native; `-DXKSV_LEGACY_PYTHON_TESTS=ON` adds the transitional Python tests.
GPU integration tests additionally require `-DXKSV_TEST_CUDA_PYTHON=/path/to/python`.

## Inventory

```sh
./build/xksv-native inspect /path/to/model > inventory.jsonl
./build/xksv-native inspect /path/to/model --stats > inventory-with-stats.jsonl
```

Each JSON line contains a shard and tensor name, dtype, shape, contiguous
strides, role, source offset and byte size. `--stats` scans BF16 data and adds
exponent and bit histograms. On a sharded repository, the index determines the
shard set. Processing statistics reads all BF16 bytes and can take time.

## Compile and verify

```sh
./xksv compile /path/to/model --output model.xksv --codec dfloat11 --workers 4
./build/xksv-native verify model.xksv --source /path/to/model
```

The existing compiler is transitional Python and requires its built C++ library.
Keep the source unchanged until compilation and verification finish. A `.partial`
file and journal are kept beside the requested output. If interrupted, use the
same source and options with `--resume`. Existing output is never overwritten.
The full verification rereads and compares reconstructed tensor bytes, so allow
for substantial disk activity. Without `--source`, native verification checks
stored checksums rather than comparing an independent source. Checksums detect
corruption; they do not establish model authenticity.

`--codec dfloat11-parallel` selects the experimental independent-substream
encoding. `--autotune` measures GPU decoder alternatives subject to a size cap;
it requires a compatible CUDA/PyTorch environment and does not tune inference.
These encodings are inspired by DFloat11 but are not compatible with its file
format. Non-BF16 tensors are stored without changing their bits.

## Run and compare

```sh
XKSV_PYTHON=/path/to/python ./xksv run model.xksv --device cuda \
  --stream --prompt 'Hello' --max-new-tokens 16
XKSV_PYTHON=/path/to/python ./xksv run model.xksv --device cuda \
  --stream --prompt 'Hello' --max-new-tokens 1 \
  --compare-original /path/to/model --compare-source-streaming --compare-layers
```

Streaming is a bounded-residency fallback, not fused compressed execution. The
comparison's source-streaming option uses the same Transformers and Accelerate
hooks for the source and archive; it is not an unmodified `from_pretrained`
reference. `--compare-layers` records observed decoder-layer equality and
`--compare-step` adds a recurrent decode comparison. These results apply only
to the tested prompt, backend, weights and numerical configuration. Use
`--match-backend-dtype` for the tested Qwen backend's automatic execution cast.

## Benchmark

```sh
XKSV_PYTHON=/path/to/python ./xksv bench model.xksv \
  --compare-original /path/to/model --prefill 512,2048 \
  --batches 1 --generate 1,128 --repeats 3 --output results.json
```

Use `--stream --source-streaming` if resident weights do not fit. Measure
warmup, timings, memory, input shapes and numerical results on the target
hardware. Prefill uses cold prompts, without a prefix-cache speedup. Report
variability and peak memory as well as compression ratio. Comparing a compressed
file's size alone does not demonstrate faster inference.

## Troubleshooting

* Missing shared library: build the project in `build/`, or set `XKSV_LIBRARY`
  to the built `libxksv_native.so`.
* Unsupported architecture or unexpected parameter names: the fallback is
  deliberately restricted; preserve the checkpoint and add a tested adapter.
* GPU out of memory: use `--stream` and lower the prompt or batch size. A
  single module and runtime scratch must still fit. Multi-GPU is not supported.
* Interrupted compilation: retain the `.partial` and `.journal` files and
  rerun with `--resume`; source changes or changed codec options are rejected.
* Source-byte verification failure: treat the archive as unusable until the
  source, hardware and storage have been investigated.
