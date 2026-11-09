#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
namespace xksv {
// Independent block. Bitstream is MSB first; code lengths define canonical codes.
struct Block {
  std::uint64_t count = 0, bit_count = 0;
  std::array<std::uint8_t, 256> lengths{};
  std::vector<std::uint8_t> sign_mantissa, exponents;
};
Block encode_bf16(std::span<const std::uint16_t> words);
std::vector<std::uint16_t> decode_bf16(const Block& block);
}
