# Design and implementation status

XKS VectorForge explores offline layout preparation and lossless compression
of whole-model Safetensors checkpoints. Its storage guarantee is exact
reconstruction of the original tensor bytes. That guarantee does not imply
that different inference kernels, reduction orders, or dtype choices produce
bit-identical activations. The implementation is a research prototype, not a
production compressed-weight serving engine.

## Data path

1. The checkpoint reader validates the Safetensors shard index, tensor names,
   descriptors, byte ranges, dtypes and contiguous source extents. It includes
   tensors outside the language-model backbone, such as vision and auxiliary
   weights. An unindexed extra file in an indexed repository is not silently
   treated as a checkpoint shard.
2. The compiler reads bounded blocks, rearranges eligible BF16 slabs into
   16-by-16 tiles, and entropy-codes exponent symbols. Sign and mantissa bits
   are preserved verbatim. Each slab uses the selected encoding only if it is
   smaller than raw bytes. Other dtypes remain raw.
3. The archive records assets, original headers and descriptors, independently
   addressed block offsets, layout dimensions and checksums. Compilation uses
   a partial file and a per-tensor journal. A final source comparison precedes
   atomic, non-overwriting publication.
4. The fallback runtime extracts model configuration, reconstructs weights and
   hands ordinary tensors to Transformers. In streaming mode, Accelerate hooks
   reconstruct module weights on demand and release them after use. Static
   layouts are inverted before these ordinary backend operators run.

The current C++ CLI implements inventory and verification. The compiler,
autotuner, benchmark harness and Transformers integration are transitional
Python components that call C++/CUDA numerical code. The migration to a fully
native C/C++ toolchain and an established native inference backend is unfinished.

## Codec and layout

The default block stores a canonical Huffman code over BF16 exponent bytes and
one byte per value holding its sign and seven mantissa bits. The block's
original dimensions drive inverse tiling. Edge tiles contain only real elements,
so layout conversion does not change tensor shapes. Raw fallback handles
incompressible blocks. Every block and whole tensor has a SHA-256 digest over
original-order bytes. See [format.md](format.md) for serialization details.

The experimental parallel variant encodes up to 32 independent substreams,
allowing GPU parallelism at the cost of per-stream metadata. Its decoder and
single-thread fused GEMV/GEMM demonstrations are experiments; they are not a
DFloat11 file format implementation or optimized production kernels. NaNs,
infinities, signed zeros and subnormals are preserved at the bit level.

## Runtime profiles

* **Resident fallback:** verifies the archive, reconstructs all supported model
  tensors and loads them into Transformers. This is suitable when memory is
  sufficient; it does not preserve compressed GPU residency.
* **Module streaming:** keeps the archive on storage and reconstructs each
  module's weights when requested by Accelerate. It reduces resident GPU weight
  memory, but repeatedly decompresses and transfers weights and can be slow.
  One module's required tensors and activation/KV scratch still need to fit.
* **Experimental fused Llama GEMV:** directly consumes compressed linear slabs
  for selected batch-one Llama operations. The implementation is not optimized
  and can change floating-point results. It is disabled by default.

The streaming loader is currently restricted to Llama and Qwen3.5-family text
execution. The backend's dtype configuration affects numerical comparisons.
For the tested Qwen3.5 checkpoint, `--match-backend-dtype` reproduces the
Transformers automatic dtype cast during execution; archived source bytes remain
unchanged. Vision inference, draft-head execution, multi-GPU scheduling and a
compressed-native backend have not been validated.

## Tuning and measurement

The optional block tuner benchmarks representative CUDA serial and parallel
block decoders, choosing the parallel representation only within a size cap.
It does not optimize end-to-end generation or choose prefill and decode layouts.
Tests in [validation.md](validation.md) and the accompanying JSON record actual
hardware and measured workloads. They do not establish speedups for other
hardware, models, batch sizes or context lengths. GPU memory numbers reported by
PyTorch do not include every possible allocation in the process or system.

## Remaining work

A complete native compiler and resume mechanism, architecture adapters and
model graph analysis, production-grade compressed GEMV/GEMM, memory-residency
planning, comprehensive profiler-driven tuning, GGUF interoperability, model
security hardening, multi-GPU execution and full original Qwen3.8 validation
remain to be implemented. No learned or low-rank approximation is used.
