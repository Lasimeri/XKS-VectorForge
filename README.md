# XKS VectorForge

An experimental AOT compiler for lossless LLM weights. The current prototype
compiles Safetensors checkpoints into indexed `.xksv` artifacts, verifies source
bytes, and runs supported text models through a Transformers fallback. It is
not yet the requested production compressed-weight inference engine: the
resident fallback reconstructs weights into ordinary GPU or CPU tensors at load
time. The streaming fallback reduces resident weight memory by decoding modules
on demand. An opt-in experimental Llama GEMV path executes compressed slabs
directly, but fused GEMM and a production optimized backend are not implemented.

Documentation: [design and status](docs/design.md), [build and usage](docs/usage.md),
[format](docs/format.md), [validation and benchmarks](docs/validation.md),
[source audit](docs/source-audit.md), and [development](docs/development.md).

## Build

C++20, CMake, and OpenSSL libcrypto are required for the native tools. Optional CUDA requires the CUDA
compiler. The native migration currently supplies `build/xksv-native inspect`
and `build/xksv-native verify` in C++; compilation and inference still use the transitional
Python implementation. The transitional implementation requires Python 3, and inference also requires PyTorch, Transformers, and Accelerate. The validated
versions are listed in `requirements-inference.txt`; install a PyTorch wheel
matching the local CUDA environment. No network access is required for normal
operation.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
# Optional CUDA prototype; choose the architecture for your GPU:
cmake -S . -B build -DXKSV_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

## Use

```sh
./xksv inspect /path/to/model
# C++-only checkpoint inventory (optionally add --stats):
./build/xksv-native inspect /path/to/model
./xksv compile /path/to/model --output model.xksv --codec dfloat11
# Optional independently coded substreams for the parallel CUDA decoder:
./xksv compile /path/to/model --output model-parallel.xksv --codec dfloat11-parallel
# Measure GPU block decoders and select between compressed formats:
XKSV_PYTHON=/path/to/python-with-torch-and-transformers ./xksv compile /path/to/model --output model-tuned.xksv --autotune
./xksv verify model.xksv --source /path/to/model
# C++-only block, tensor, and source-byte verification:
./build/xksv-native verify model.xksv --source /path/to/model
# After interruption, continue using the same source and output:
./xksv compile /path/to/model --output model.xksv --resume
XKSV_PYTHON=/path/to/python-with-torch-and-transformers ./xksv run model.xksv --device cuda --prompt 'Hello' --max-new-tokens 16
# Module-level compressed-file streaming fallback:
XKSV_PYTHON=/path/to/python-with-torch-and-transformers ./xksv run model.xksv --device cuda --stream --prompt 'Hello'
XKSV_PYTHON=/path/to/python-with-torch-and-transformers ./xksv bench model.xksv --compare-original /path/to/model --prefill 512,2048,8192 --batches 1,4,8 --generate 1,128,512 --repeats 3 --output results.json
# Use bounded-memory source and compiled loading for a model larger than VRAM:
XKSV_PYTHON=/path/to/python-with-torch-and-transformers ./xksv bench model.xksv --compare-original /path/to/model --source-streaming --stream --prefill 512 --batches 1 --generate 1 --repeats 1
```

`--gpu-decode` selects experimental CUDA decoding for loading. The default
codec uses a serial decoder; `dfloat11-parallel` uses independent substreams
and a parallel decoder, with larger metadata. See [block timings](docs/benchmark-parallel-decode.json). The optional `xksv_cuda_gemv` and `xksv_cuda_gemm` functions fuse BF16 decoding
and FP32 computation as single-thread experiments. `--stream
--experimental-fused-gemv` opts supported Llama linear layers into fused
batch-one execution; larger batches use an ordinary reconstructed-matrix
fallback. This mode is not bit-exact and is not an optimized kernel. Run its tests
with a Python environment containing PyTorch: `python tests/test_cuda.py`.

`--autotune` benchmarks representative serial and parallel CUDA block decodes,
then selects the faster compressed encoding when the parallel block is at most
15% larger. It groups similar block shapes and exponent-size distributions,
records the GPU and measurements, and falls back to raw bytes if neither codec
compresses. This is a block-loading tuner; it does not tune GEMV/GEMM, prefill,
or end-to-end generation. Resume a tuned build with `--autotune --resume` on
the same GPU model and compute capability.

The compiler uses a bounded multi-threaded work queue (configure `--workers`)
and checkpoints progress after each tensor. Resume rechecks committed
blocks and source bytes before continuing; incomplete work uses adjacent
`.partial` and `.journal` files.

The compiler reads tensor slabs with bounded memory, packs BF16 into reversible
16x16 tiles, Huffman-codes their exponents, and stores sign/mantissa bits
unchanged. Each slab chooses compressed or raw storage and has a random-access
index and SHA-256 of its original bytes. Non-BF16 tensors remain unchanged.
The final artifact is verified against source tensors and atomically published
without overwriting existing files. Model and tokenizer configuration files are
embedded. This codec is inspired by DFloat11 but is not format-compatible.

The [format description](docs/format.md) records the serialized layout.
The C++ inventory reader vendors nlohmann/json 3.12.0 under its MIT license.
The legacy Python integration tests are optional; enable them with
`-DXKSV_LEGACY_PYTHON_TESTS=ON` when configuring CMake.

The verifier independently checks blocks and whole-tensor hashes; use `--source`
to compare the actual source bytes and tensor descriptors. The format and
parser are experimental and have not undergone a security audit. File
integrity hashes do not authenticate a model or establish its trustworthiness.

## Validation and limitations

The tested Qwen3.5-0.8B artifact derives from [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
and retains its [Apache 2.0 model license](artifacts/Qwen3.5-0.8B.LICENSE).
Model files are excluded from this code repository.
The project code uses the root MIT license.

A local Qwen3.6-derived GrugCap-27B checkpoint was also compiled to
an `.xksv` artifact. All 1,199 indexed tensors were compared to their
source bytes during compilation. This is a scale test of a related 64-layer
architecture, not a test of the original Qwen3.8-27B checkpoint. See
[validation details](docs/validation.md).
For one short prompt, all 64 decoder layers and prefill logits matched a
source-streaming reference byte-for-byte, and generated token IDs matched.
The compiled streaming fallback was substantially slower in that single test.

The CPU tests cover all BF16 bit patterns, edge tiles, corruption, mixed dtypes,
and source comparison. A pretrained SmolLM2-135M was compiled and run on an RTX
3090 Ti: tested last-token logits and greedy output tokens matched the original
Transformers load exactly for the tested prompts. See
[prefill benchmark data](docs/benchmark-smollm2-prefill.json),
[decode benchmark data](docs/benchmark-smollm2-decode.json), and
[validation details](docs/validation.md).
Qwen3.5-0.8B's 488 source tensors reconstructed exactly, and a byte-level
comparison matched all 24 text decoder layers, prefill logits, and one decode
step on a tested prompt when
`--match-backend-dtype` was set. That option deliberately reproduces the
Transformers `dtype=auto` cast of 36 FP32 source tensors to BF16 **for execution**;
the archive preserves their original FP32 bits. Without it, source dtypes are
retained and the tested logits differed from Transformers' cast baseline.
Fifteen MTP draft-head tensors remain in the archive but are unused by this
Transformers generation path. Those results do not establish bit-identical inference for other models,
backends, hardware, or numerical settings.

The ordinary fallback reconstructs all weights at load time. A separate
`--stream` fallback uses Accelerate hooks to decode and transfer each module's
weights on demand, and releases them after use. Its startup validates archive
metadata and it checks each block and tensor when fetched; use `verify` for an
independent full-file check. For models too large for a conventional reference
load, `--compare-original /path/to/source --compare-source-streaming` compares
against the original Safetensors weights loaded module by module through the
same Transformers/Accelerate execution path. This reference is not an
unmodified `from_pretrained` load. Streaming is automatically selected
when the estimated reconstructed weights leave insufficient GPU headroom. It
repeatedly decodes weights on every token, is slow, requires a PyTorch and
Accelerate environment, and its memory estimate does not guarantee sufficient
scratch or KV-cache space. A single tensor must fit on the device. The fallback
is currently enabled only for Llama and Qwen3.5 text models. An initial Qwen
streaming test failed because DeltaNet reads a convolution weight directly; the
fallback now preloads that module, and the tested Qwen prompt's layer values
and generated IDs match the original backend. See
[resident validation](docs/qwen35-resident-validation.json) and
[streaming validation](docs/qwen35-stream-validation.json) and
[byte-level validation](docs/qwen35-bytewise-validation.json). Other missing
work includes GGUF, model dependency graph extraction, end-to-end autotuning, multi-GPU, optimized fused
GEMM, direct compressed-weight inference, robust hostile-file validation, and
the full benchmark matrix. Original Qwen3.8-27B execution remains unverified.
The CUDA GEMV is not a throughput
optimization, and no inference speedup is claimed.
