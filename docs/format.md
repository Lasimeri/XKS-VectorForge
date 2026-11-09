# XKSV prototype format, version 1

The prototype uses little-endian integers. A file begins with the 8-byte magic
`XKSV 00 01 0d 0a` and a 64-bit absolute offset to a UTF-8 JSON index at the end
of the file. Bytes between the 16-byte header and index are asset and tensor
blocks; the index supplies their absolute offsets and lengths. The reader
requires blocks to cover that region without overlap or holes. The index holds
original Safetensors headers, shapes, dtypes, tensor offsets, SHA-256 hashes,
source-shard hashes for new builds, and an ordered block list for every tensor.
The original header and tensor descriptors are cross-checked during verification.
Model/tokenizer assets are stored as independently hashed blocks. Asset hashes
provide integrity checks, not authenticity. This format is experimental and
not compatible with DFloat11's serialized format.

Each BF16 slab is arranged as row-major 16-by-16 tiles, with edge tiles containing
only real elements. A block may instead store the source bytes unchanged if its
encoded size is no smaller. An encoded slab stores:

* `uint64` element count and `uint64` exponent bit count;
* 256 canonical Huffman code lengths, one byte per exponent;
* one byte per element containing the original sign and mantissa bits;
* an MSB-first packed exponent stream, padded only at the final byte.

An experimental `bf16-huffman-parallel32` block instead stores up to 32
independently coded contiguous substreams of the tiled slab. Its `XKSP2`
header records the slab dimensions and substream lengths. Each substream has
its own lengths, sign/mantissa bytes and exponent stream as above. This adds
metadata but allows parallel GPU decoding. The compiler falls back to raw
bytes when the selected encoding is no smaller. The default remains the
single-codebook format.

Blocks are independently decodable. After decoding, the tile layout is inverted
to obtain original source bytes and checked against the block hash. Non-BF16
blocks are raw and retain their original byte representation. Tensor-level hashes
cover original contiguous tensor bytes. The compiler uses a journal and a partial
file in the output directory. It fsyncs data and the journal at tensor boundaries;
resume validates the source shard hashes and all committed blocks before reuse.
After full source verification it links the completed file to the requested name
without replacing an existing file, and removes the partial and journal files.

There is no authentication, stable compatibility promise, graph representation,
GPU kernel profile, or general-purpose hardened parser in this version. Tensor
and block metadata is JSON and is read into memory. This is suitable for the
current prototype, not yet the intended production 27B streaming format.
