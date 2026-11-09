#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
namespace xksv {
// Row-major matrix -> row-major tiles. Edge tiles contain only real elements.
std::vector<std::uint16_t> tile(std::span<const std::uint16_t> input, std::size_t rows, std::size_t cols, std::size_t tile_rows, std::size_t tile_cols);
std::vector<std::uint16_t> untile(std::span<const std::uint16_t> input, std::size_t rows, std::size_t cols, std::size_t tile_rows, std::size_t tile_cols);
}
