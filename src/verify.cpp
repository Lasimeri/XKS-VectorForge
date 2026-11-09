#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int xksv_unpack(const void *, size_t, size_t, size_t, void **,
                           size_t *);
extern "C" int xksv_unpack_parallel(const void *, size_t, void **, size_t *);
extern "C" void xksv_free(void *);
namespace fs = std::filesystem;
using json = nlohmann::json;
json inventory_native(const fs::path &, bool);
namespace {
constexpr uint64_t chunk = 1 << 20;
uint64_t num(const json &j) {
  if (!j.is_number_integer() ||
      (!j.is_number_unsigned() && j.get<int64_t>() < 0))
    throw std::runtime_error("invalid integer in archive");
  return j.get<uint64_t>();
}
uint64_t mul(uint64_t a, uint64_t b) {
  if (b && a > UINT64_MAX / b)
    throw std::runtime_error("size overflow");
  return a * b;
}
std::string read_at(std::ifstream &f, uint64_t offset, uint64_t size) {
  if (size > std::numeric_limits<std::streamsize>::max() ||
      offset > std::numeric_limits<std::streamoff>::max())
    throw std::runtime_error("file range too large");
  std::string data(size, '\0');
  f.clear();
  f.seekg(offset);
  if (!f || (size && !f.read(data.data(), size)))
    throw std::runtime_error("truncated file");
  return data;
}
uint64_t u64(const char *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= uint64_t(static_cast<unsigned char>(p[i])) << (i * 8);
  return v;
}
json parse(const std::string &text) {
  std::map<int, std::set<std::string>> keys;
  return json::parse(text,
                     [&](int depth, json::parse_event_t event, json &value) {
                       if (event == json::parse_event_t::object_start)
                         keys[depth + 1].clear();
                       if (event == json::parse_event_t::key &&
                           !keys[depth].insert(value.get<std::string>()).second)
                         throw std::runtime_error("duplicate JSON key");
                       return true;
                     });
}
struct Hash {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(),
                                                              EVP_MD_CTX_free};
  Hash() {
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("SHA-256 initialization failed");
  }
  void update(const std::string &s) {
    if (EVP_DigestUpdate(ctx.get(), s.data(), s.size()) != 1)
      throw std::runtime_error("SHA-256 failed");
  }
  std::string finish() {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int n = 0;
    if (EVP_DigestFinal_ex(ctx.get(), digest, &n) != 1)
      throw std::runtime_error("SHA-256 failed");
    std::ostringstream out;
    for (unsigned i = 0; i < n; ++i)
      out << std::hex << std::setw(2) << std::setfill('0')
          << unsigned(digest[i]);
    return out.str();
  }
};
std::string hash(const std::string &s) {
  Hash h;
  h.update(s);
  return h.finish();
}
std::string base64(const std::string &s) {
  if (s.size() % 4 || s.size() > 150000000)
    throw std::runtime_error("invalid stored header encoding");
  if (s.empty())
    return {};
  std::string out((s.size() / 4) * 3, '\0');
  int n = EVP_DecodeBlock(reinterpret_cast<unsigned char *>(out.data()),
                          reinterpret_cast<const unsigned char *>(s.data()),
                          s.size());
  if (n < 0)
    throw std::runtime_error("invalid stored header encoding");
  if (s.back() == '=')
    --n;
  if (s.size() > 1 && s[s.size() - 2] == '=')
    --n;
  out.resize(n);
  return out;
}
void plain_name(const std::string &s) {
  fs::path p(s);
  if (s.empty() || p.filename() != p || s == "." || s == "..")
    throw std::runtime_error("invalid stored filename");
}
const std::map<std::string, uint64_t> sizes = {
    {"BOOL", 1}, {"U8", 1},  {"I8", 1},  {"F8_E4M3", 1}, {"F8_E5M2", 1},
    {"I16", 2},  {"U16", 2}, {"F16", 2}, {"BF16", 2},    {"I32", 4},
    {"U32", 4},  {"F32", 4}, {"I64", 8}, {"U64", 8},     {"F64", 8}};
} // namespace
void verify_native(const fs::path &path, const fs::path &source,
                   bool allow_partial) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open archive");
  auto file_size = fs::file_size(path);
  if (file_size < 16)
    throw std::runtime_error("truncated archive");
  auto prefix = read_at(f, 0, 16);
  if (std::memcmp(prefix.data(), "XKSV\0\1\r\n", 8))
    throw std::runtime_error("invalid archive magic");
  auto limit = u64(prefix.data() + 8);
  if (limit < 16 || limit > file_size || file_size - limit > 1000000000)
    throw std::runtime_error("invalid archive index offset");
  auto index = parse(read_at(f, limit, file_size - limit));
  if (index.value("version", 0) != 1 ||
      index.value("format", std::string()) != "safetensors")
    throw std::runtime_error("unsupported archive");
  std::vector<std::pair<uint64_t, uint64_t>> spans;
  auto span = [&](uint64_t off, uint64_t n, uint64_t max) {
    if (off < 16 || off > limit || n > limit - off || n > max)
      throw std::runtime_error("invalid archive extent");
    spans.emplace_back(off, off + n);
  };
  std::set<std::string> assets, shards, names, source_shards;
  uint64_t tensors = 0, bytes = 0;
  if (!source.empty())
    for (const auto &shard : inventory_native(source, false))
      source_shards.insert(shard.at("name").get<std::string>());
  for (const auto &asset : index.at("assets")) {
    auto name = asset.at("name").get<std::string>();
    plain_name(name);
    if (!assets.insert(name).second)
      throw std::runtime_error("duplicate asset");
    auto off = num(asset.at("offset")), n = num(asset.at("size"));
    span(off, n, 100000000);
    auto data = read_at(f, off, n);
    if (hash(data) != asset.at("sha256").get<std::string>())
      throw std::runtime_error("asset checksum failed: " + name);
    if (!source.empty()) {
      auto p = source / name;
      if (!fs::is_regular_file(p) || fs::file_size(p) != n)
        throw std::runtime_error("source asset differs: " + name);
      std::ifstream in(p, std::ios::binary);
      if (read_at(in, 0, n) != data)
        throw std::runtime_error("source asset differs: " + name);
    }
  }
  for (const auto &shard : index.at("shards")) {
    auto name = shard.at("name").get<std::string>();
    plain_name(name);
    if (!shards.insert(name).second)
      throw std::runtime_error("duplicate shard");
    auto header = base64(shard.at("header").get<std::string>());
    auto stored = parse(header);
    if (!stored.is_object())
      throw std::runtime_error("invalid shard header");
    std::ifstream src;
    uint64_t source_size = 0;
    if (!source.empty()) {
      auto p = fs::is_directory(source) ? source / name : source;
      src.open(p, std::ios::binary);
      if (!src)
        throw std::runtime_error("missing source shard: " + name);
      source_size = fs::file_size(p);
      if (source_size < 8 || u64(read_at(src, 0, 8).data()) != header.size() ||
          read_at(src, 8, header.size()) != header)
        throw std::runtime_error("source header differs: " + name);
    }
    std::set<std::string> seen;
    std::vector<std::pair<uint64_t, uint64_t>> source_ranges;
    for (const auto &t : shard.at("tensors")) {
      auto tensor_name = t.at("name").get<std::string>();
      if (!names.insert(tensor_name).second ||
          !seen.insert(tensor_name).second || !stored.contains(tensor_name) ||
          tensor_name == "__metadata__")
        throw std::runtime_error("invalid tensor name");
      auto dtype = t.at("dtype").get<std::string>();
      auto type = sizes.find(dtype);
      if (type == sizes.end() || !t.at("shape").is_array())
        throw std::runtime_error("invalid tensor type");
      uint64_t elements = 1;
      for (const auto &dim : t.at("shape"))
        elements = mul(elements, num(dim));
      auto start = num(t.at("start")), length = num(t.at("size"));
      if (length != mul(elements, type->second) || start > UINT64_MAX - length)
        throw std::runtime_error("invalid tensor size");
      const auto &d = stored.at(tensor_name);
      if (d.at("dtype") != dtype || d.at("shape") != t.at("shape") ||
          d.at("data_offsets") != json::array({start, start + length}))
        throw std::runtime_error("tensor descriptor mismatch: " + tensor_name);
      if (length)
        source_ranges.emplace_back(start, start + length);
      if (!source.empty() && (header.size() + 8 > source_size ||
                              start > source_size - header.size() - 8 ||
                              length > source_size - header.size() - 8 - start))
        throw std::runtime_error("source tensor outside file");
      uint64_t processed = 0;
      Hash digest;
      for (const auto &b : t.at("blocks")) {
        auto off = num(b.at("offset")), n = num(b.at("size")),
             raw_size = num(b.at("raw_size"));
        span(off, n, chunk);
        if (raw_size > chunk || processed > length ||
            raw_size > length - processed)
          throw std::runtime_error("invalid block size");
        auto codec = b.at("codec").get<std::string>();
        auto encoded = read_at(f, off, n);
        std::string data;
        if (codec == "raw") {
          if (n != raw_size)
            throw std::runtime_error("invalid raw block");
          data = std::move(encoded);
        } else if (codec == "bf16-huffman-tile16" ||
                   codec == "bf16-huffman-parallel32") {
          auto rows = num(b.at("rows")), cols = num(b.at("cols"));
          if (dtype != "BF16" || !rows || !cols ||
              mul(mul(rows, cols), 2) != raw_size)
            throw std::runtime_error("invalid compressed block shape");
          if (codec == "bf16-huffman-parallel32" &&
              (n < 24 || u64(encoded.data() + 8) != rows ||
               u64(encoded.data() + 16) != cols))
            throw std::runtime_error("parallel block shape mismatch");
          void *out = nullptr;
          size_t out_size = 0;
          int rc = codec == "bf16-huffman-tile16"
                       ? xksv_unpack(encoded.data(), encoded.size(), rows, cols,
                                     &out, &out_size)
                       : xksv_unpack_parallel(encoded.data(), encoded.size(),
                                              &out, &out_size);
          std::unique_ptr<void, decltype(&xksv_free)> owned(out, xksv_free);
          if (rc || out_size != raw_size)
            throw std::runtime_error("block decoding failed: " + tensor_name);
          data.assign(static_cast<char *>(out), out_size);
        } else
          throw std::runtime_error("unknown block codec");
        if (hash(data) != b.at("sha256").get<std::string>())
          throw std::runtime_error("block checksum failed: " + tensor_name);
        if (!source.empty() &&
            data != read_at(src, 8 + header.size() + start + processed,
                            data.size()))
          throw std::runtime_error("source bytes differ: " + tensor_name);
        digest.update(data);
        processed += data.size();
      }
      if (processed != length ||
          digest.finish() != t.at("sha256").get<std::string>())
        throw std::runtime_error("tensor checksum failed: " + tensor_name);
      ++tensors;
      if (bytes > UINT64_MAX - length)
        throw std::runtime_error("byte count overflow");
      bytes += length;
    }
    size_t expected = stored.size() - (stored.contains("__metadata__") ? 1 : 0);
    if (!allow_partial && seen.size() != expected)
      throw std::runtime_error("stored tensor set mismatch");
    std::sort(source_ranges.begin(), source_ranges.end());
    uint64_t end = 0;
    for (auto [begin, next] : source_ranges) {
      if (begin != end)
        throw std::runtime_error("invalid source tensor coverage");
      end = next;
    }
    if (!allow_partial && !source.empty() &&
        end != source_size - header.size() - 8)
      throw std::runtime_error("source file has unreferenced data");
  }
  if (!source.empty() && shards != source_shards)
    throw std::runtime_error("source shard set differs");
  std::sort(spans.begin(), spans.end());
  uint64_t end = 16;
  for (auto [begin, next] : spans) {
    if (begin != end)
      throw std::runtime_error("invalid archive data coverage");
    end = next;
  }
  if (end != limit)
    throw std::runtime_error("unreferenced archive data");
  std::cout << "verified " << tensors << " tensors, " << bytes << " bytes"
            << (source.empty() ? " against stored checksums"
                               : " against source")
            << '\n';
}
