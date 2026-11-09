# Validation record

Development host: RTX 3090 Ti, 24,564 MiB reported VRAM, NVIDIA driver
615.78.08, CUDA compiler 13.4.92, PyTorch 2.13.0+cu130, Transformers 5.6.0,
GCC 16.2.1, Linux. Other graphical applications were running. Results are
observations, not speedup claims.

## Safetensors and codec

CPU tests exhaustively round-trip all 65,536 BF16 bit patterns, exercise
irregular tile edges, single-symbol coding, malformed streams, mixed source
dtypes, raw fallback, corrupted data, interrupted compilation, changed source
and serial versus parallel deterministic output. CUDA tests compare independent
block decoding bit-for-bit with the CPU input and compare fused FP32 GEMV/GEMM
experiments numerically within tolerance. Those kernels use one thread and are
not integrated into model inference. [Kernel timing data](benchmark-cuda-kernels.json)
shows the fused prototype was slower than the separate path for the two tested
block sizes; these measurements include allocation, transfer and launch overhead.
A separate [parallel decoder benchmark](benchmark-parallel-decode.json) measured
2.45 ms versus 24.7 ms for a 16×4096 block, at 95,687 versus 86,995 encoded
bytes. For a 16×256 block, parallel metadata made the encoded block larger than
the raw source, so the compiler chooses raw storage for that case. These are
individual-block measurements, not model-inference speedups. The experimental
fused model path is not currently implemented for GrugCap's architecture.

## GrugCap-27B scale test

The local GrugCap checkpoint is a Qwen3.6-derived merge using the 64-layer
`qwen3_5` architecture. It is not the original Qwen3.8-27B checkpoint. Its
indexed 1,199 tensors comprise 55,562,855,904 bytes. All were reconstructed
and compared byte-for-byte with the source during compilation; the resulting
`artifacts/GrugCap-27B.xksv` is 37,001,698,739 bytes including its metadata
and assets. The separate unindexed auxiliary Safetensors file is not part of
the indexed checkpoint. The runtime does not use the 15 MTP draft-head tensors
for normal Transformers generation, although they are stored and verified.

The C++ compiler also compiled the full checkpoint, resumed after a deliberate
interruption, and verified all 1,199 tensors and 55,562,855,904 source bytes.
Its 37,001,811,104-byte artifact was removed after verification to preserve disk
space; the earlier verified artifact remains in use and is being uploaded.

On an RTX 3090 Ti, a five-token prompt and one generated token matched the
source-streaming reference byte-for-byte for all 64 decoder layer outputs and
final prefill logits, and the generated token IDs matched. Both sides used the
same Transformers/Accelerate hooks and source dtypes; this is not an
unmodified `from_pretrained` comparison. In this one synchronized run, the
compiled streaming prefill took 317.60 s versus 28.84 s for streamed source
weights, and generation including prefill took 311.84 s versus 28.64 s. PyTorch
reported peak GPU allocations of 2.72 GB in each phase, and peak process RSS
was 4.37 GB. These are single-run measurements of a short prompt, not a
throughput study. See [the raw result](grugcap-27b-validation.json).

### Native llama.cpp bridge

One cold native C/C++ run used the same compiled GrugCap archive through the
experimental FUSE bridge and llama.cpp commit `f5b9bd39b56c7a7839a9795a100b6a00b84ac961`.
The Qwen3.8 GGUF supplied metadata and tokenizer information only; model weights
were decoded from the lossless GrugCap archive. Twenty layers were offloaded to
the RTX 3090 Ti, with eight CPU threads, a 512-token context, a five-token
prompt, greedy sampling, two generated tokens, and no warmup. The result was
`The capital of France is Paris.` The first generated token matched the
earlier source-streaming test; no native-to-Transformers logit comparison was
performed.

The runner reported 63.026 s for five prefill tokens (0.079 tokens/s) and
40.128 s for one decode evaluation (0.0249 tokens/s). It reported 103.164 s
total for prompt processing and generation. Loading before prompt processing
took approximately 263 s by wall-clock timestamps; this is separate from that
103 s. Samples taken every five seconds observed a maximum of 19,113 MiB total
GPU memory in use, including other processes, and 15,835,648 KiB runner RSS.
The bridge and system page cache are not included in runner RSS. This is one
short request on a busy local machine, not a steady-state throughput study or a
controlled comparison of the native and Python backends. The bridge decodes and
pages weights on demand and does not use fused compressed-weight GEMM.

## Qwen3.5-0.8B

The official checkpoint had 488 tensors and 1,746,882,752 tensor bytes. The
compiled artifact is 1,207,244,026 bytes, including assets and index. Every
tensor reconstructed bit-for-bit against the original source. The artifact is
available at `artifacts/Qwen3.5-0.8B.xksv`.

With the same Transformers dtype configuration (`--match-backend-dtype`), a
five-token prompt produced matching prefill logit values, all 24 decoder layer
values, one subsequent recurrent decode's logits, and the same generated
IDs for both resident and streaming tests. A later streaming run compared
the bytes of all 24 layers, prefill logits and one decode step directly and
matched the original backend; see [byte-level validation](qwen35-bytewise-validation.json).
A second comparison using the original Safetensors weights streamed through the
same backend and hooks also matched prefill and decode logits byte-for-byte; see
[source-streaming validation](qwen35-source-stream-validation.json). The backend normally casts 36 FP32
source tensors to BF16. Source-dtype execution preserves those FP32 values but
its tested logits differed from the backend's default cast configuration; the
weight archive is lossless in either mode. Fifteen MTP draft-head tensors were
verified in the archive but are not used by this Transformers generation path.
Vision execution has not been tested.

In one two-token request, streaming allocated a peak 531,006,976 GPU bytes in
PyTorch versus 1,760,767,488 bytes in a separate resident request. The streamed
request's generation including prefill took 23.47 s. These are single runs with
different request lengths and are not a controlled latency comparison. The
streaming implementation decodes and transfers weights on each module invocation.

Synchronized resident comparisons for 512 and 2048-token prompts and 1 and 16
output tokens, batch 1, are in [benchmark-qwen35.json](benchmark-qwen35.json).
Each case has one measured run after warmup; this is insufficient to establish
variability. Both paths use regular reconstructed Transformers weights during
inference, so similar throughput is expected.

PCIe volume, hardware bandwidth, per-layer kernel timing,
profiling, full benchmark repetitions, optimized fused inference, and original
Qwen3.8-27B execution remain unmeasured. Results from other models cannot
establish original Qwen3.8-27B correctness or performance.
