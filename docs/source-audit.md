# Source audit and scope

Observed 2026-11-08. Reference snapshots are not vendored.

* XKS-Memory commit `2777ba2ddf5cc72997f63c7169104964a78941e5`:
  `src/vector/kernels.c` and `src/semantic/linalg.c` contain scalar and
  runtime-dispatched AVX2/FMA implementations; `src/vector/store.c` validates
  and mmaps persisted numerical data. Its vector backend targets retrieval,
  while `rsvd.c` implements truncated SVD. Neither retrieval nor truncated SVD
  is appropriate for lossless model weight compilation.
* DFloat11 commit `457733886ce6ebc6d8dda1621fad1ffa2661e028`:
  `dfloat11/dfloat11_utils.py` counts BF16 exponents, builds Huffman codes,
  limits code lengths for its GPU implementation, builds hierarchical lookup
  tables, and encodes thread-level stream boundaries. `dfloat11/decode.cu`
  reconstructs BF16 with lookup tables, offsets, and sign/mantissa bytes.
  The present CPU block codec is a separate design and does not implement
  DFloat11's thread offsets, lookup layout, or model integration.
* [ZipServ](https://arxiv.org/abs/2603.17435) describes hardware-aware lossless
  compression and fused execution; neither its format nor its kernels are
  implemented here.
* [Qwen3.8-27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
  identifies `Qwen3_5ForConditionalGeneration`, BF16 text weights, 64 text
  layers with 48 linear and 16 full attention layers, 5120 hidden dimensions,
  vision and MTP components. The [checkpoint index](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/model.safetensors.index.json)
  lists 1,199 tensor names, 18 shards, and 55,562,855,904 bytes of tensor data.
  Individual tensor contents have not been inspected or downloaded.
* llama.cpp commit `7d7f0287530bc8981690f361d8c1627dbae1c99a` contains
  `QWEN35` architecture and conversion code. This alone does not establish
  compatibility with this project's future compressed layout or exact inference.
  [SGLang's model guide](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx)
  documents Qwen3.8 support. The vLLM `qwen3_5.py` implementation at commit
  `fd0c471b3a4c2446c44b2ba8b4699a42e181d7b2` maps the separate DeltaNet
  checkpoint projections into fused QKVZ and BA parameters. SGLang's
  `qwen3_5.py` at `4003bf1077debcca3fccfa6e0d7b56ed584aa727` uses
  corresponding packed projection loaders and supports a fused projection.
  Neither backend is integrated here; Transformers provides a fallback.

The current prototype implements a streaming Safetensors reader, a versioned
indexed archive, checksums, source reconstruction verification, tensor-level
resume, a Transformers fallback loader, and experimental serial CUDA decoding
and decode/GEMV/GEMM kernels. Model-based tests and measurements have been run
on SmolLM2-135M, but not yet on the target 27B model. Required next steps include
full graph and Qwen hybrid integration, compressed-weight inference, optimized
GPU kernels, offload, autotuning, and complete performance profiling.
The RTX 3090 Ti in the development environment has 24 GB VRAM, insufficient to
hold the uncompressed checkpoint; the feasibility and performance of streaming
have not been measured.

An initial Qwen3.5-0.8B streaming experiment failed numerical validation.
DeltaNet directly reads its child convolution weight, bypassing its loading
hook; preloading the entire DeltaNet module fixed the observed mismatch. The
resident and streaming fallbacks passed layer-by-layer comparison on one prompt
when execution dtype was matched to Transformers. Streaming is restricted to
Llama and Qwen3.5 text models and remains experimental.
