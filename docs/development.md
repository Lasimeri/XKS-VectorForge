# Development and reproducibility

The C++ sources in `src/` implement BF16 entropy coding, reversible tile
layout, the C ABI, native Safetensors inspection, compilation and archive verification.
`cuda/` contains experimental decoders and fused-kernel prototypes. The
transitional compiler, autotuner and runtime live in `python/`; the runtime has
not yet been ported to native code.
The `xksv` entry point selects C++ for inspect, compile and verify, and the
transitional implementation for run and bench. `build/xksv-native` is the
native binary. nlohmann/json 3.12.0 is
vendored under `third_party/nlohmann/` with its MIT license; OpenSSL is a system
dependency. The project's own code uses the root MIT license.

## Tests

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Native tests exercise all 65,536 BF16 bit patterns, reversible layouts, scalar
and dispatched paths, Safetensors descriptor validation, and archive corruption.
The optional Python tests cover checkpoint compilation, resumption and
integration; the optional GPU tests require a CUDA device and PyTorch. Do not
interpret unit-test success as validation of an untested model or backend.

When changing an encoding, exercise raw fallback, all supported dtypes, empty and
edge tensors, NaNs, signed zeros, independent block access, malformed metadata,
checksum failures, source comparison and interrupted output handling. Do not
change format version 1 incompatibly. Inference changes must separately report
weight-byte preservation and observed numerical behavior, with fixed backend,
dtype, hardware, prompts and generation settings.

## Benchmark discipline

Keep source checkpoints and artifacts outside Git. Record the exact model,
compiler revision, command, GPU/driver, dependency versions, warmup, prompt
length, batch size, decode length, synchronization, repeat count, memory
measurement scope, and numeric comparison. Preserve raw results and disclose
missing metrics. A repeated-token synthetic benchmark is not representative of
all workloads. See [validation.md](validation.md) and the JSON results in this
directory for the present measurements and limitations.

## Contributing

Submit focused changes with tests that exercise observable behavior. Maintain
bounded memory for large checkpoints and reject malformed sizes, offsets,
headers, and codec streams. Keep source files unchanged and never overwrite an
existing output artifact. Document compatibility, dependency and numerical
changes. If a bug could expose secrets or affect model integrity, avoid posting
sensitive checkpoint contents or credentials in a public issue.
