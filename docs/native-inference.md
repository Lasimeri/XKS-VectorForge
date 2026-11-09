# Experimental native GrugCap inference

The optional `xksv-llama-bridge` supplies decoded GrugCap weights to an
installed llama.cpp build through a read-only FUSE filesystem. It exposes a
virtual GGUF file; the 53.9 GB uncompressed view is generated on demand and is
not written to disk. llama.cpp executes the model in C/C++ and may offload
layers to CUDA. This is a research bridge, not the requested optimized fused
compressed-weight backend.

The current bridge supports the tested 64-layer GrugCap/Qwen3.5 text architecture
only. It requires a compatible Qwen3.8-27B GGUF **as a metadata and tokenizer
template**. The reference file's weight values are never used. The bridge
checks architecture parameters, the model's defined vocabulary and BPE merges,
tensor names and tensor sizes before mounting. It reads source weights from the
`.xksv` archive and checks each decoded block's SHA-256. Vision and draft-head
execution are not supported by this bridge.

The llama.cpp architecture uses a different layout for DeltaNet value heads.
The bridge applies the corresponding reversible head permutations while
serving weights. It also constructs the normalization and recurrent constants
expected by that backend from the archived BF16 values. The original archive is
unchanged. These execution transformations and llama.cpp's arithmetic do not
establish bit-identical inference relative to Transformers. Compare outputs for
the actual workload and backend settings before relying on them.

## Build

Linux, FUSE 3 headers and userspace tools, C++20, OpenSSL, and a built llama.cpp
checkout with shared `ggml-base` are required. The integration was developed
against llama.cpp commit `f5b9bd39b56c7a7839a9795a100b6a00b84ac961`; other
revisions are not guaranteed to work. CUDA support depends on how llama.cpp was
built.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DXKSV_ENABLE_LLAMA_BRIDGE=ON \
  -DXKSV_LLAMA_ROOT=/path/to/llama.cpp
cmake --build build -j
```

If llama.cpp was built outside its source tree, set
`-DXKSV_LLAMA_BUILD_DIR=/path/to/llama-build` as well.

## Run

In one terminal, mount the archive and keep the process running:

```sh
mkdir -p /tmp/xksv-mount
./build/xksv-llama-bridge /path/to/GrugCap-27B.xksv \
  /path/to/compatible-Qwen3.8-27B.gguf /tmp/xksv-mount
```

In a second terminal, run the C/C++ backend:

```sh
/path/to/llama.cpp/build/bin/llama-completion \
  -m /tmp/xksv-mount/model.gguf \
  -p 'The capital of France is' -n 2 -c 512 -t 8 \
  -ngl 20 --fit off --no-warmup --temp 0 --no-conversation
```

Adjust GPU layers to the actual free VRAM. The number of layers that fit
varies with other allocations, context size, and backend version. The operating
system can cache or evict decoded pages, so host RAM and storage bandwidth can
have a large effect. A cold start may touch a large fraction of the model;
generation can reread and decode weights when they are not cached. Do not
interpret the virtual file size as resident memory or as compressed GPU
residency. The bridge uses a bounded 128 MiB shared decoded-block cache plus
per-thread references. Unmount with `fusermount3 -u /tmp/xksv-mount` after the runner exits.

The template provides metadata and tokenizer rules, so matching tensor shapes
alone is insufficient. The bridge rejects unsupported head dimensions, layer
schedules, rope settings, vocabulary and merge rules. It currently relies on an
external template rather than compiling all llama.cpp execution metadata into
`.xksv`; removing that requirement is future work.
