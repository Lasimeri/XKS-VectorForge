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
individual-block measurements, not model-inference speedups. The opt-in fused
GEMV path ran 210 linear layers of SmolLM2-135M: prefill logits matched, but
its first decode logits differed by up to 0.3125 and the two-token request
including prefill took 48.18 seconds. Generated IDs matched for that single
prompt. See [fused model validation](smollm2-fused-validation.json); this is not
strict execution and not an acceleration. With the parallel-substream artifact,
the fused path applied to 30 linear layers and the remaining weights used the
regular streaming fallback. A separate one-token-prompt check happened to match
the original logits and token exactly, but atomic accumulation can change
reduction order; this one result does not establish bit-identical execution.
See [parallel fused model validation](smollm2-fused-parallel-validation.json).

## GrugCap-27B scale test

The local GrugCap checkpoint is a Qwen3.6-derived merge using the 64-layer
`qwen3_5` architecture. It is not the original Qwen3.8-27B checkpoint. Its
indexed 1,199 tensors comprise 55,562,855,904 bytes. All were reconstructed
and compared byte-for-byte with the source during compilation; the resulting
`artifacts/GrugCap-27B.xksv` is 37,001,698,739 bytes including its metadata
and assets. The separate unindexed auxiliary Safetensors file is not part of
the indexed checkpoint. The runtime does not use the 15 MTP draft-head tensors
for normal Transformers generation, although they are stored and verified.

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

## SmolLM2-135M

The 269,030,016 tensor bytes compiled to a 188,262,645-byte artifact including
assets and index. A checkpoint interruption and resume produced the exact same
artifact SHA-256 as the uninterrupted compilation. An experimental
32-substream build occupied 266,449,432 bytes including assets and index; for
this small model its metadata and raw fallbacks largely erase compression.
A size-constrained autotuned build occupied 188,276,048 bytes and selected
the single-codebook codec for every block: no eligible block met the 15%
parallel-size cap, so no GPU timing profile was needed for that model. A
synthetic larger-slab test confirms that eligible candidates are timed and the
result recorded.
The experimental GPU loading path reconstructed the parallel-substream artifact and matched
both prefill and decode logits on a tested prompt; see
[parallel GPU validation](smollm2-parallel-gpu-validation.json). Its roughly
4.4-second load time is a single run and mostly reflects raw fallbacks. For batch sizes 1, 4 and 8,
512/2048/8192-token prefill and 1-token output were tested. Separately, 128 and
512-token generation were tested at each of those batch sizes with 512-token
prompts. Tested final prefill logit values and generated IDs matched the original
backend. These historical logit comparisons used float32 value equality;
they did not check the sign bit of zero. Results are in [prefill data](benchmark-smollm2-prefill.json) and
[decode data](benchmark-smollm2-decode.json). Each case has one measured run
after a prompt-shape warmup; prompts consist of repeated fixed token IDs and
prefix caching is not used. GPU operations are synchronized around timings.
Peak GPU allocation is PyTorch's cumulative peak for each model phase, not
system-wide VRAM usage. The JSON includes individual timing values.

PCIe volume, hardware bandwidth, peak host RAM, per-layer kernel timing,
profiling, full benchmark repetitions, optimized fused inference, and original
Qwen3.8-27B execution remain unmeasured. Results from other models cannot
establish original Qwen3.8-27B correctness or performance.
