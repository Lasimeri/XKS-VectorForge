#define FUSE_USE_VERSION 31
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <fuse3/fuse.h>
#include <ggml.h>
#include <gguf.h>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
extern "C" int xksv_unpack(const void *, size_t, size_t, size_t, void **,
                           size_t *);
extern "C" int xksv_unpack_parallel(const void *, size_t, void **, size_t *);
extern "C" void xksv_free(void *);
using json = nlohmann::json;
struct Block {
  uint64_t start, raw, offset, size, rows, cols;
  std::string codec, sha;
};
struct Source {
  std::string name;
  uint64_t size;
  std::vector<Block> blocks;
};
enum class Perm { none, qkv, z, heads, conv, out };
struct Tensor {
  std::string name;
  const Source *source;
  uint64_t offset, size, cols, rows;
  bool f32, plus_one, neg_exp;
  Perm perm;
};
static int archive_fd = -1;
static std::vector<char> metadata;
static std::vector<Tensor> tensors;
static uint64_t virtual_size = 0;
static std::map<std::string, Source> sources;
static void read_exact(uint64_t pos, void *data, size_t n) {
  auto *p = static_cast<char *>(data);
  while (n) {
    auto q = pread(archive_fd, p, n, pos);
    if (q <= 0)
      throw std::runtime_error("archive read failed");
    p += q;
    pos += q;
    n -= q;
  }
}
static std::string digest(const std::string &s) {
  unsigned char d[32];
  unsigned n;
  auto *ctx = EVP_MD_CTX_new();
  if (!ctx || EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(ctx, s.data(), s.size()) != 1 ||
      EVP_DigestFinal_ex(ctx, d, &n) != 1) {
    if (ctx)
      EVP_MD_CTX_free(ctx);
    throw std::runtime_error("SHA256 failure");
  }
  EVP_MD_CTX_free(ctx);
  static const char h[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (unsigned i = 0; i < n; i++) {
    out += h[d[i] >> 4];
    out += h[d[i] & 15];
  }
  return out;
}
static std::shared_ptr<const std::string> block_data(const Source &source,
                                                     size_t i) {
  struct Cache {
    const Source *src = nullptr;
    size_t index = 0;
    std::shared_ptr<const std::string> data;
  };
  using Key = std::pair<const Source *, size_t>;
  struct Entry {
    std::shared_ptr<const std::string> data;
    std::list<Key>::iterator order;
  };
  static std::mutex cache_mutex;
  static std::map<Key, Entry> shared;
  static std::list<Key> lru;
  static size_t shared_bytes = 0;
  constexpr size_t limit = 128 * 1024 * 1024;
  static thread_local Cache cache;
  if (cache.src == &source && cache.index == i)
    return cache.data;
  const Key key{&source, i};
  {
    std::lock_guard lock(cache_mutex);
    auto found = shared.find(key);
    if (found != shared.end()) {
      lru.splice(lru.begin(), lru, found->second.order);
      cache = {&source, i, found->second.data};
      return cache.data;
    }
  }
  const auto &b = source.blocks[i];
  std::string encoded(b.size, '\0');
  read_exact(b.offset, encoded.data(), encoded.size());
  std::string out;
  if (b.codec == "raw")
    out = std::move(encoded);
  else {
    void *p = nullptr;
    size_t n = 0;
    int rc = b.codec == "bf16-huffman-tile16"
                 ? xksv_unpack(encoded.data(), encoded.size(), b.rows, b.cols,
                               &p, &n)
             : b.codec == "bf16-huffman-parallel32"
                 ? xksv_unpack_parallel(encoded.data(), encoded.size(), &p, &n)
                 : -1;
    std::unique_ptr<void, decltype(&xksv_free)> hold(p, xksv_free);
    if (rc || n != b.raw)
      throw std::runtime_error("decode failure: " + source.name);
    out.assign(static_cast<char *>(p), n);
  }
  if (out.size() != b.raw || digest(out) != b.sha)
    throw std::runtime_error("block checksum mismatch: " + source.name);
  auto decoded = std::make_shared<const std::string>(std::move(out));
  {
    std::lock_guard lock(cache_mutex);
    auto found = shared.find(key);
    if (found != shared.end()) {
      lru.splice(lru.begin(), lru, found->second.order);
      decoded = found->second.data;
    } else if (decoded->size() <= limit) {
      while (!lru.empty() && decoded->size() > limit - shared_bytes) {
        auto old = shared.find(lru.back());
        shared_bytes -= old->second.data->size();
        shared.erase(old);
        lru.pop_back();
      }
      lru.push_front(key);
      shared.emplace(key, Entry{decoded, lru.begin()});
      shared_bytes += decoded->size();
    }
  }
  cache = {&source, i, std::move(decoded)};
  return cache.data;
}
static void source_copy(const Source &s, uint64_t pos, char *dst, size_t n) {
  if (pos > s.size || n > s.size - pos)
    throw std::runtime_error("source range error: " + s.name);
  while (n) {
    auto it = std::upper_bound(
        s.blocks.begin(), s.blocks.end(), pos,
        [](uint64_t v, const Block &b) { return v < b.start; });
    if (it == s.blocks.begin())
      throw std::runtime_error("missing block");
    --it;
    auto i = size_t(it - s.blocks.begin());
    auto data = block_data(s, i);
    auto off = pos - it->start;
    auto take = std::min<size_t>(n, data->size() - off);
    if (!take)
      throw std::runtime_error("empty block");
    memcpy(dst, data->data() + off, take);
    dst += take;
    pos += take;
    n -= take;
  }
}
static uint64_t reorder(uint64_t head) { return (head % 16) * 3 + head / 16; }
static uint64_t source_row(const Tensor &t, uint64_t row) {
  if (t.perm == Perm::qkv || t.perm == Perm::conv) {
    if (row < 4096)
      return row;
    row -= 4096;
    return 4096 + reorder(row / 128) * 128 + row % 128;
  }
  if (t.perm == Perm::z)
    return reorder(row / 128) * 128 + row % 128;
  if (t.perm == Perm::heads)
    return reorder(row);
  return row;
}
static uint64_t source_element(const Tensor &t, uint64_t elem) {
  auto row = elem / t.cols, col = elem % t.cols;
  if (t.perm == Perm::heads && t.rows == 1)
    col = reorder(col);
  else
    row = source_row(t, row);
  if (t.perm == Perm::out)
    col = reorder(col / 128) * 128 + col % 128;
  return row * t.cols + col;
}
static void tensor_read(const Tensor &t, uint64_t pos, char *dst, size_t n) {
  if (pos > t.size || n > t.size - pos)
    throw std::runtime_error("tensor range error");
  if (t.f32) {
    while (n) {
      uint64_t e = pos / 4;
      auto idx = source_element(t, e);
      unsigned char raw[2];
      source_copy(*t.source, idx * 2, reinterpret_cast<char *>(raw), 2);
      uint32_t bits = uint32_t(raw[0]) << 16 | uint32_t(raw[1]) << 24;
      float f;
      memcpy(&f, &bits, 4);
      if (t.plus_one)
        f += 1.0f;
      if (t.neg_exp)
        f = -std::exp(f);
      char v[4];
      memcpy(v, &f, 4);
      size_t off = pos % 4, take = std::min(n, 4 - off);
      memcpy(dst, v + off, take);
      dst += take;
      pos += take;
      n -= take;
    }
    return;
  }
  while (n) {
    uint64_t e = pos / 2, byte = pos % 2, row = e / t.cols, col = e % t.cols;
    auto src = source_element(t, e) * 2 + byte;
    uint64_t segment = (t.cols - col) * 2 - byte;
    if (t.perm == Perm::out)
      segment = std::min<uint64_t>(segment, (128 - col % 128) * 2 - byte);
    size_t take = std::min<uint64_t>(n, segment);
    source_copy(*t.source, src, dst, take);
    dst += take;
    pos += take;
    n -= take;
  }
}
static int get_attr(const char *path, struct stat *s, struct fuse_file_info *) {
  memset(s, 0, sizeof(*s));
  if (!strcmp(path, "/")) {
    s->st_mode = S_IFDIR | 0555;
    s->st_nlink = 2;
    return 0;
  }
  if (!strcmp(path, "/model.gguf")) {
    s->st_mode = S_IFREG | 0444;
    s->st_nlink = 1;
    s->st_size = virtual_size;
    return 0;
  }
  return -ENOENT;
}
static int open_file(const char *path, struct fuse_file_info *f) {
  if (strcmp(path, "/model.gguf"))
    return -ENOENT;
  if ((f->flags & O_ACCMODE) != O_RDONLY)
    return -EACCES;
  return 0;
}
static int read_file(const char *path, char *dst, size_t n, off_t off,
                     struct fuse_file_info *) {
  if (strcmp(path, "/model.gguf"))
    return -ENOENT;
  if (off < 0)
    return -EINVAL;
  uint64_t pos = off;
  if (pos >= virtual_size)
    return 0;
  n = std::min<uint64_t>(n, virtual_size - pos);
  size_t ret = n;
  try {
    while (n) {
      if (pos < metadata.size()) {
        size_t take = std::min<uint64_t>(n, metadata.size() - pos);
        memcpy(dst, metadata.data() + pos, take);
        dst += take;
        pos += take;
        n -= take;
        continue;
      }
      auto it = std::upper_bound(
          tensors.begin(), tensors.end(), pos,
          [](uint64_t p, const Tensor &t) { return p < t.offset; });
      if (it != tensors.begin()) {
        auto prev = std::prev(it);
        if (pos < prev->offset + prev->size) {
          size_t take = std::min<uint64_t>(n, prev->offset + prev->size - pos);
          tensor_read(*prev, pos - prev->offset, dst, take);
          dst += take;
          pos += take;
          n -= take;
          continue;
        }
      }
      uint64_t next = it == tensors.end() ? virtual_size : it->offset;
      size_t take = std::min<uint64_t>(n, next - pos);
      if (!take)
        throw std::runtime_error("invalid GGUF range");
      memset(dst, 0, take);
      dst += take;
      pos += take;
      n -= take;
    }
  } catch (const std::exception &e) {
    std::cerr << "bridge read: " << e.what() << '\n';
    return -EIO;
  }
  return int(ret);
}
static std::string source_name(const std::string &name, Perm &perm, bool &plus,
                               bool &exp) {
  perm = Perm::none;
  plus = false;
  exp = false;
  if (name == "output.weight")
    return "lm_head.weight";
  if (name == "token_embd.weight")
    return "model.language_model.embed_tokens.weight";
  if (name == "output_norm.weight") {
    plus = true;
    return "model.language_model.norm.weight";
  }
  if (name.rfind("blk.", 0) != 0)
    throw std::runtime_error("unknown tensor: " + name);
  auto dot = name.find('.', 4);
  if (dot == std::string::npos)
    throw std::runtime_error("bad tensor name");
  int layer = std::stoi(name.substr(4, dot - 4));
  if (layer >= 64)
    return {};
  auto suffix = name.substr(dot + 1);
  std::string prefix =
      "model.language_model.layers." + std::to_string(layer) + ".";
  static const std::map<std::string, std::string> map = {
      {"attn_norm.weight", "input_layernorm.weight"},
      {"post_attention_norm.weight", "post_attention_layernorm.weight"},
      {"ffn_down.weight", "mlp.down_proj.weight"},
      {"ffn_gate.weight", "mlp.gate_proj.weight"},
      {"ffn_up.weight", "mlp.up_proj.weight"},
      {"attn_q.weight", "self_attn.q_proj.weight"},
      {"attn_k.weight", "self_attn.k_proj.weight"},
      {"attn_v.weight", "self_attn.v_proj.weight"},
      {"attn_output.weight", "self_attn.o_proj.weight"},
      {"attn_q_norm.weight", "self_attn.q_norm.weight"},
      {"attn_k_norm.weight", "self_attn.k_norm.weight"},
      {"attn_qkv.weight", "linear_attn.in_proj_qkv.weight"},
      {"attn_gate.weight", "linear_attn.in_proj_z.weight"},
      {"ssm_a", "linear_attn.A_log"},
      {"ssm_alpha.weight", "linear_attn.in_proj_a.weight"},
      {"ssm_beta.weight", "linear_attn.in_proj_b.weight"},
      {"ssm_conv1d.weight", "linear_attn.conv1d.weight"},
      {"ssm_dt.bias", "linear_attn.dt_bias"},
      {"ssm_norm.weight", "linear_attn.norm.weight"},
      {"ssm_out.weight", "linear_attn.out_proj.weight"}};
  auto i = map.find(suffix);
  if (i == map.end())
    throw std::runtime_error("unknown tensor: " + name);
  if (suffix == "attn_norm.weight" || suffix == "post_attention_norm.weight" ||
      suffix == "attn_q_norm.weight" || suffix == "attn_k_norm.weight")
    plus = true;
  if (suffix == "ssm_a")
    exp = true;
  if (suffix == "attn_qkv.weight")
    perm = Perm::qkv;
  else if (suffix == "attn_gate.weight")
    perm = Perm::z;
  else if (suffix == "ssm_conv1d.weight")
    perm = Perm::conv;
  else if (suffix == "ssm_a" || suffix == "ssm_dt.bias" ||
           suffix == "ssm_alpha.weight" || suffix == "ssm_beta.weight")
    perm = Perm::heads;
  else if (suffix == "ssm_out.weight")
    perm = Perm::out;
  return prefix + i->second;
}
static std::string asset(const json &index, const std::string &name) {
  for (auto &a : index.at("assets"))
    if (a.at("name") == name) {
      std::string data(a.at("size").get<size_t>(), '\0');
      read_exact(a.at("offset").get<uint64_t>(), data.data(), data.size());
      if (digest(data) != a.at("sha256").get<std::string>())
        throw std::runtime_error("asset checksum mismatch");
      return data;
    }
  throw std::runtime_error("missing asset: " + name);
}
static void setup(const char *archive, const char *reference) {
  archive_fd = open(archive, O_RDONLY | O_CLOEXEC);
  if (archive_fd < 0)
    throw std::runtime_error("cannot open archive");
  struct stat st{};
  if (fstat(archive_fd, &st))
    throw std::runtime_error("cannot stat archive");
  char h[16];
  read_exact(0, h, 16);
  if (memcmp(h, "XKSV\0\1\r\n", 8))
    throw std::runtime_error("bad archive magic");
  uint64_t idx = 0;
  for (int i = 0; i < 8; i++)
    idx |= uint64_t((unsigned char)h[8 + i]) << (8 * i);
  if (idx < 16 || idx > uint64_t(st.st_size) ||
      uint64_t(st.st_size) - idx > 1000000000)
    throw std::runtime_error("bad archive index");
  std::string text(st.st_size - idx, '\0');
  read_exact(idx, text.data(), text.size());
  auto index = json::parse(text);
  for (auto &shard : index.at("shards"))
    for (auto &t : shard.at("tensors")) {
      if (t.at("dtype") != "BF16")
        continue;
      Source s;
      s.name = t.at("name").get<std::string>();
      s.size = t.at("size").get<uint64_t>();
      uint64_t pos = 0;
      for (auto &b : t.at("blocks")) {
        Block x{pos,
                b.at("raw_size").get<uint64_t>(),
                b.at("offset").get<uint64_t>(),
                b.at("size").get<uint64_t>(),
                b.at("rows").get<uint64_t>(),
                b.at("cols").get<uint64_t>(),
                b.at("codec").get<std::string>(),
                b.at("sha256").get<std::string>()};
        if (x.offset > idx || x.size > idx - x.offset || x.raw > 1048576 ||
            x.size > 1048576)
          throw std::runtime_error("invalid archive block");
        pos += x.raw;
        s.blocks.push_back(x);
      }
      if (pos != s.size || !sources.emplace(s.name, std::move(s)).second)
        throw std::runtime_error("invalid tensor blocks");
    }
  auto cfg = json::parse(asset(index, "config.json"));
  auto config = cfg.at("text_config");
  if (cfg.value("model_type", std::string()) != "qwen3_5" ||
      !config.value("attn_output_gate", false) ||
      config.value("vocab_size", 0) != 248320 ||
      config.value("hidden_size", 0) != 5120 ||
      config.value("num_hidden_layers", 0) != 64 ||
      config.value("linear_num_key_heads", 0) != 16 ||
      config.value("linear_num_value_heads", 0) != 48 ||
      config.value("linear_key_head_dim", 0) != 128 ||
      config.value("linear_value_head_dim", 0) != 128)
    throw std::runtime_error("unsupported Qwen layout");
  auto tokenizer = json::parse(asset(index, "tokenizer.json"));
  ggml_context *g = nullptr;
  gguf_init_params p{true, &g};
  auto old = gguf_init_from_file(reference, p);
  if (!old)
    throw std::runtime_error("cannot load GGUF reference metadata");
  auto key = [&](const char *name) {
    auto id = gguf_find_key(old, name);
    if (id < 0)
      throw std::runtime_error(std::string("reference metadata is missing ") +
                               name);
    return id;
  };
  auto check_u32 = [&](const char *name, uint32_t expected) {
    auto id = key(name);
    if (gguf_get_kv_type(old, id) != GGUF_TYPE_UINT32 ||
        gguf_get_val_u32(old, id) != expected)
      throw std::runtime_error(std::string("reference metadata differs: ") +
                               name);
  };
  auto arch = key("general.architecture");
  if (gguf_get_kv_type(old, arch) != GGUF_TYPE_STRING ||
      std::string(gguf_get_val_str(old, arch)) != "qwen35")
    throw std::runtime_error("reference architecture is not qwen35");
  auto block_count = key("qwen35.block_count");
  auto nextn_count = key("qwen35.nextn_predict_layers");
  if (gguf_get_kv_type(old, block_count) != GGUF_TYPE_UINT32 ||
      gguf_get_kv_type(old, nextn_count) != GGUF_TYPE_UINT32 ||
      gguf_get_val_u32(old, block_count) < gguf_get_val_u32(old, nextn_count) ||
      gguf_get_val_u32(old, block_count) - gguf_get_val_u32(old, nextn_count) !=
          64)
    throw std::runtime_error("reference block count differs");
  check_u32("qwen35.embedding_length",
            config.at("hidden_size").get<uint32_t>());
  check_u32("qwen35.feed_forward_length",
            config.at("intermediate_size").get<uint32_t>());
  check_u32("qwen35.context_length",
            config.at("max_position_embeddings").get<uint32_t>());
  check_u32("qwen35.attention.head_count",
            config.at("num_attention_heads").get<uint32_t>());
  check_u32("qwen35.attention.head_count_kv",
            config.at("num_key_value_heads").get<uint32_t>());
  check_u32("qwen35.attention.key_length",
            config.at("head_dim").get<uint32_t>());
  check_u32("qwen35.attention.value_length",
            config.at("head_dim").get<uint32_t>());
  check_u32("qwen35.ssm.conv_kernel",
            config.at("linear_conv_kernel_dim").get<uint32_t>());
  check_u32("qwen35.ssm.state_size",
            config.at("linear_key_head_dim").get<uint32_t>());
  check_u32("qwen35.ssm.group_count",
            config.at("linear_num_key_heads").get<uint32_t>());
  check_u32("qwen35.ssm.time_step_rank",
            config.at("linear_num_value_heads").get<uint32_t>());
  check_u32("qwen35.ssm.inner_size",
            config.at("linear_value_head_dim").get<uint32_t>() *
                config.at("linear_num_value_heads").get<uint32_t>());
  check_u32("qwen35.full_attention_interval",
            config.at("full_attention_interval").get<uint32_t>());
  auto rope = config.at("rope_parameters");
  if (!rope.value("mrope_interleaved", false) ||
      rope.value("rope_type", std::string()) != "default")
    throw std::runtime_error("unsupported source rope configuration");
  check_u32("qwen35.rope.dimension_count",
            uint32_t(config.at("head_dim").get<double>() *
                     rope.at("partial_rotary_factor").get<double>()));
  auto check_f32 = [&](const char *name, float expected) {
    auto id = key(name);
    if (gguf_get_kv_type(old, id) != GGUF_TYPE_FLOAT32 ||
        gguf_get_val_f32(old, id) != expected)
      throw std::runtime_error(std::string("reference metadata differs: ") +
                               name);
  };
  check_f32("qwen35.rope.freq_base", rope.at("rope_theta").get<float>());
  check_f32("qwen35.attention.layer_norm_rms_epsilon",
            config.at("rms_norm_eps").get<float>());
  auto sections = key("qwen35.rope.dimension_sections");
  if (gguf_get_kv_type(old, sections) != GGUF_TYPE_ARRAY ||
      gguf_get_arr_type(old, sections) != GGUF_TYPE_INT32 ||
      gguf_get_arr_n(old, sections) != 4)
    throw std::runtime_error("invalid reference rope sections");
  auto section_data =
      static_cast<const int32_t *>(gguf_get_arr_data(old, sections));
  for (int i = 0; i < 3; ++i)
    if (section_data[i] != rope.at("mrope_section")[i].get<int32_t>())
      throw std::runtime_error("reference rope sections differ");
  if (section_data[3] != 0)
    throw std::runtime_error("unsupported reference rope sections");
  const auto &layers = config.at("layer_types");
  if (layers.size() != 64)
    throw std::runtime_error("invalid layer schedule");
  for (size_t i = 0; i < 64; ++i)
    if (layers[i].get<std::string>() !=
        (i % 4 == 3 ? "full_attention" : "linear_attention"))
      throw std::runtime_error("unsupported layer schedule");
  auto tokens = gguf_find_key(old, "tokenizer.ggml.tokens"),
       merges = gguf_find_key(old, "tokenizer.ggml.merges");
  if (tokens < 0 || merges < 0 || gguf_get_arr_n(old, tokens) != 248320)
    throw std::runtime_error("invalid reference tokenizer");
  auto tokenizer_model = key("tokenizer.ggml.model");
  auto tokenizer_pre = key("tokenizer.ggml.pre");
  if (gguf_get_kv_type(old, tokenizer_model) != GGUF_TYPE_STRING ||
      gguf_get_kv_type(old, tokenizer_pre) != GGUF_TYPE_STRING ||
      std::string(gguf_get_val_str(old, tokenizer_model)) != "gpt2" ||
      std::string(gguf_get_val_str(old, tokenizer_pre)) != "qwen35")
    throw std::runtime_error("unsupported reference tokenizer rules");
  for (auto it = tokenizer["model"]["vocab"].begin();
       it != tokenizer["model"]["vocab"].end(); ++it) {
    size_t id = it.value().get<size_t>();
    if (id >= 248320 || it.key() != gguf_get_arr_str(old, tokens, id))
      throw std::runtime_error("tokenizer vocabulary differs from reference");
  }
  for (auto &t : tokenizer["added_tokens"]) {
    size_t id = t["id"].get<size_t>();
    if (id >= 248320 ||
        t["content"].get<std::string>() != gguf_get_arr_str(old, tokens, id))
      throw std::runtime_error("added tokens differ from reference");
  }
  const auto &ms = tokenizer["model"]["merges"];
  if (ms.size() != gguf_get_arr_n(old, merges))
    throw std::runtime_error("tokenizer merges count differs");
  for (size_t i = 0; i < ms.size(); ++i) {
    auto m = ms[i].is_array() ? ms[i][0].get<std::string>() + " " +
                                    ms[i][1].get<std::string>()
                              : ms[i].get<std::string>();
    if (m != gguf_get_arr_str(old, merges, i))
      throw std::runtime_error("tokenizer merges differ from reference");
  }
  auto out = gguf_init_empty();
  gguf_set_kv(out, old);
  gguf_set_val_u32(out, "qwen35.block_count", 64);
  gguf_set_val_u32(out, "qwen35.nextn_predict_layers", 0);
  gguf_set_val_u32(out, "general.file_type", 32);
  gguf_set_val_str(out, "general.name", "GrugCap-27B-XKSV");
  for (int64_t i = 0; i < gguf_get_n_tensors(old); ++i) {
    std::string name = gguf_get_tensor_name(old, i);
    Perm perm;
    bool plus, exp;
    auto srcname = source_name(name, perm, plus, exp);
    if (srcname.empty())
      continue;
    auto si = sources.find(srcname);
    if (si == sources.end())
      throw std::runtime_error("missing source tensor: " + srcname);
    auto *t = ggml_get_tensor(g, name.c_str());
    uint64_t count = ggml_nelements(t);
    if (count * 2 != si->second.size)
      throw std::runtime_error("tensor size mismatch: " + name);
    bool f32 = t->type == GGML_TYPE_F32;
    gguf_add_tensor(out, t);
    gguf_set_tensor_type(out, name.c_str(),
                         f32 ? GGML_TYPE_F32 : GGML_TYPE_BF16);
    tensors.push_back(
        {name, &si->second, 0, count * (f32 ? 4 : 2), uint64_t(t->ne[0]),
         uint64_t(t->ne[1] * t->ne[2] * t->ne[3]), f32, plus, exp, perm});
  }
  auto meta_size = gguf_get_meta_size(out);
  metadata.resize(meta_size);
  gguf_get_meta_data(out, metadata.data());
  for (size_t i = 0; i < tensors.size(); i++)
    tensors[i].offset = meta_size + gguf_get_tensor_offset(out, i);
  virtual_size = meta_size + gguf_get_tensor_offset(out, tensors.size() - 1) +
                 gguf_get_tensor_size(out, tensors.size() - 1);
  std::cerr << "bridge ready: " << tensors.size() << " tensors, virtual GGUF "
            << virtual_size << " bytes\n";
  gguf_free(out);
  gguf_free(old);
  ggml_free(g);
}
int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: bridge ARCHIVE REFERENCE_GGUF MOUNTPOINT\n";
      return 2;
    }
    setup(argv[1], argv[2]);
    fuse_operations op{};
    op.getattr = get_attr;
    op.open = open_file;
    op.read = read_file;
    char *args[] = {argv[0],
                    argv[3],
                    const_cast<char *>("-f"),
                    const_cast<char *>("-o"),
                    const_cast<char *>("ro"),
                    nullptr};
    return fuse_main(5, args, &op, nullptr);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
