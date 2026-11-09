#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;
json inventory_native(const fs::path &, bool);
void verify_native(const fs::path &, const fs::path &, bool);
extern "C" int xksv_pack(const void *, size_t, size_t, size_t, void **,
                         size_t *);
extern "C" int xksv_pack_parallel(const void *, size_t, size_t, size_t, void **,
                                  size_t *);
extern "C" void xksv_free(void *);
namespace {
constexpr uint64_t chunk = 1 << 20;
void error(const std::string &message) {
  throw std::runtime_error(message + ": " + std::strerror(errno));
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
std::string sha(const std::string &s) {
  Hash h;
  h.update(s);
  return h.finish();
}
std::string read(std::ifstream &f, uint64_t size) {
  std::string data(size, '\0');
  if (size && !f.read(data.data(), size))
    throw std::runtime_error("unexpected end of source");
  return data;
}
std::string hash_file(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open source shard: " + p.string());
  Hash h;
  std::vector<char> buffer(chunk);
  while (f) {
    f.read(buffer.data(), buffer.size());
    auto n = f.gcount();
    if (n)
      h.update(std::string(buffer.data(), n));
  }
  if (!f.eof())
    throw std::runtime_error("source shard read failed");
  return h.finish();
}
std::string encode64(const std::string &s) {
  if (s.size() > 100000000)
    throw std::runtime_error("header too large");
  std::string out(4 * ((s.size() + 2) / 3), '\0');
  if (!s.empty())
    EVP_EncodeBlock(reinterpret_cast<unsigned char *>(out.data()),
                    reinterpret_cast<const unsigned char *>(s.data()),
                    s.size());
  return out;
}
void fsync_path(const fs::path &p, bool directory = false) {
  int fd =
      open(p.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0));
  if (fd < 0)
    error("open for sync failed");
  if (fsync(fd) < 0) {
    int e = errno;
    close(fd);
    errno = e;
    error("fsync failed");
  }
  close(fd);
}
void write_all(int fd, const char *data, size_t size) {
  while (size) {
    auto n = ::write(fd, data, size);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      error("write failed");
    data += n;
    size -= n;
  }
}
void journal_write(const fs::path &path, const json &j) {
  auto pattern =
      (path.parent_path() / ("." + path.filename().string() + ".XXXXXX"))
          .string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back(0);
  int fd = mkstemp(name.data());
  if (fd < 0)
    error("cannot create journal temporary file");
  try {
    auto data = j.dump();
    write_all(fd, data.data(), data.size());
    if (fsync(fd) < 0)
      error("journal fsync failed");
    if (close(fd) < 0) {
      fd = -1;
      error("journal close failed");
    }
    fd = -1;
    if (rename(name.data(), path.c_str()) < 0)
      error("journal rename failed");
    fsync_path(path.parent_path(), true);
  } catch (...) {
    if (fd >= 0)
      close(fd);
    unlink(name.data());
    throw;
  }
}
void put64(std::ostream &out, uint64_t n) {
  for (int i = 0; i < 8; ++i)
    out.put(char(n >> (i * 8)));
  if (!out)
    throw std::runtime_error("output write failed");
}
void header(std::fstream &out, uint64_t pos) {
  out.clear();
  out.seekp(0);
  out.write("XKSV\0\1\r\n", 8);
  put64(out, pos);
}
void durable(std::fstream &out, const fs::path &p) {
  out.flush();
  if (!out)
    throw std::runtime_error("output flush failed");
  fsync_path(p);
}
void write(std::fstream &out, const std::string &s) {
  if (!s.empty())
    out.write(s.data(), s.size());
  if (!out)
    throw std::runtime_error("output write failed");
}
uint64_t offset(std::fstream &out) {
  auto p = out.tellp();
  if (p < 0)
    throw std::runtime_error("cannot get output offset");
  return uint64_t(p);
}
std::string path_file(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), {});
}
json clean_tensor(const json &t) {
  json result = t;
  result.erase("shard");
  result.erase("statistics");
  return result;
}
struct Encoded {
  std::string data, codec, checksum;
  size_t raw_size;
  uint64_t rows, cols;
};
class Workers {
  std::vector<std::thread> threads;
  std::queue<std::function<void()>> tasks;
  std::mutex mutex;
  std::condition_variable ready;
  bool stop = false;

public:
  explicit Workers(unsigned count) {
    try {
      for (unsigned i = 0; i < count; ++i)
        threads.emplace_back([this] {
          while (true) {
            std::function<void()> task;
            {
              std::unique_lock lock(mutex);
              ready.wait(lock, [&] { return stop || !tasks.empty(); });
              if (stop && tasks.empty())
                return;
              task = std::move(tasks.front());
              tasks.pop();
            }
            task();
          }
        });
    } catch (...) {
      {
        std::lock_guard lock(mutex);
        stop = true;
      }
      ready.notify_all();
      for (auto &t : threads)
        t.join();
      throw;
    }
  }
  ~Workers() {
    {
      std::lock_guard lock(mutex);
      stop = true;
    }
    ready.notify_all();
    for (auto &t : threads)
      t.join();
  }
  std::future<Encoded> submit(std::string raw, uint64_t rows, uint64_t cols,
                              const std::string &codec) {
    auto task = std::make_shared<std::packaged_task<Encoded()>>(
        [raw = std::move(raw), rows, cols, codec]() mutable {
          Encoded result;
          result.raw_size = raw.size();
          result.rows = rows;
          result.cols = cols;
          result.checksum = sha(raw);
          result.codec = "raw";
          if (rows) {
            void *ptr = nullptr;
            size_t n = 0;
            int rc =
                codec == "dfloat11"
                    ? xksv_pack(raw.data(), raw.size(), rows, cols, &ptr, &n)
                    : xksv_pack_parallel(raw.data(), raw.size(), rows, cols,
                                         &ptr, &n);
            std::unique_ptr<void, decltype(&xksv_free)> owned(ptr, xksv_free);
            if (rc)
              throw std::runtime_error("native packing failed");
            if (n < raw.size()) {
              result.data.assign(static_cast<char *>(ptr), n);
              result.codec = codec == "dfloat11" ? "bf16-huffman-tile16"
                                                 : "bf16-huffman-parallel32";
            }
          }
          if (result.codec == "raw")
            result.data = std::move(raw);
          return result;
        });
    auto future = task->get_future();
    {
      std::lock_guard lock(mutex);
      tasks.emplace([task] { (*task)(); });
    }
    ready.notify_one();
    return future;
  }
};
const std::array<const char *, 19> asset_names = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "tokenizer.model",
    "vocab.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
    "NOTICE",
    "README.md",
    "adapter_config.json",
    "model.safetensors.index.json"};
} // namespace
void compile_native(const fs::path &source, const fs::path &output_arg,
                    const std::string &codec, bool resume, unsigned workers) {
  auto inventory = inventory_native(source, false);
  auto output = fs::absolute(output_arg);
  auto parent = output.parent_path();
  if (!fs::is_directory(parent))
    throw std::runtime_error("output directory does not exist");
  if (fs::exists(output))
    throw std::runtime_error("output already exists");
  for (const auto &s : inventory)
    if (fs::weakly_canonical(s.at("path").get<std::string>()) ==
        fs::weakly_canonical(output))
      throw std::runtime_error("output would overwrite source");
  auto partial = fs::path(output.string() + ".partial"),
       journal = fs::path(output.string() + ".journal");
  json index;
  uint64_t position = 0;
  if (resume) {
    if (!fs::is_regular_file(partial) || !fs::is_regular_file(journal))
      throw std::runtime_error("resume requires partial file and journal");
    auto state = json::parse(path_file(journal));
    index = state.at("index");
    position = state.at("position").get<uint64_t>();
    if (position < 16 || fs::file_size(partial) < position ||
        index.value("codec_preference", std::string("dfloat11")) != codec)
      throw std::runtime_error("invalid or incompatible resume journal");
    if (index.at("shards").size() != inventory.size())
      throw std::runtime_error("source shard set changed");
    bool unfinished = false;
    for (size_t i = 0; i < inventory.size(); ++i) {
      const auto &src = inventory[i];
      const auto &saved = index["shards"][i];
      if (saved.at("name") != src.at("name") ||
          saved.at("header").get<std::string>() !=
              encode64(src.at("header").get<std::string>()) ||
          saved.value("source_sha256", std::string()) !=
              hash_file(src.at("path").get<std::string>()))
        throw std::runtime_error("source shard changed");
      const auto &old = saved.at("tensors");
      const auto &now = src.at("tensors");
      if (old.size() > now.size() || (unfinished && !old.empty()))
        throw std::runtime_error("invalid journal tensor sequence");
      for (size_t j = 0; j < old.size(); ++j)
        for (const auto *key : {"name", "dtype", "shape", "start", "size"})
          if (old[j].at(key) != now[j].at(key))
            throw std::runtime_error("source tensor descriptor changed");
      if (old.size() != now.size())
        unfinished = true;
    }
    // The journal records a complete tensor boundary. Temporarily append an
    // index so the normal verifier can check committed data before reuse.
    fs::resize_file(partial, position);
    {
      std::fstream out(partial,
                       std::ios::in | std::ios::out | std::ios::binary);
      if (!out)
        throw std::runtime_error("cannot reopen partial output");
      out.seekp(position);
      write(out, index.dump());
      header(out, position);
      durable(out, partial);
    }
    try {
      verify_native(partial, source, true);
    } catch (...) {
      fs::resize_file(partial, position);
      std::fstream out(partial,
                       std::ios::in | std::ios::out | std::ios::binary);
      header(out, 0);
      durable(out, partial);
      throw;
    }
    fs::resize_file(partial, position);
    std::fstream out(partial, std::ios::in | std::ios::out | std::ios::binary);
    header(out, 0);
    durable(out, partial);
  } else {
    if (fs::exists(partial) || fs::exists(journal))
      throw std::runtime_error("partial compilation exists; use --resume");
    index = {{"version", 1},
             {"format", "safetensors"},
             {"codec_preference", codec},
             {"shards", json::array()},
             {"assets", json::array()}};
    // Hash each source before creating an output so resume detects any changes.
    for (const auto &s : inventory)
      index["shards"].push_back(
          {{"name", s.at("name")},
           {"header", encode64(s.at("header").get<std::string>())},
           {"source_sha256", hash_file(s.at("path").get<std::string>())},
           {"tensors", json::array()}});
    int fd = open(partial.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0)
      error("cannot create partial output");
    close(fd);
    std::fstream out(partial, std::ios::in | std::ios::out | std::ios::binary);
    if (!out)
      throw std::runtime_error("cannot open partial output");
    header(out, 0);
    out.seekp(16);
    if (fs::is_directory(source))
      for (auto name : asset_names) {
        fs::path p = source / name;
        if (!fs::is_regular_file(p))
          continue;
        if (fs::file_size(p) > 100000000)
          throw std::runtime_error("asset too large: " + std::string(name));
        auto data = path_file(p);
        index["assets"].push_back({{"name", name},
                                   {"offset", offset(out)},
                                   {"size", data.size()},
                                   {"sha256", sha(data)}});
        write(out, data);
      }
    durable(out, partial);
    position = offset(out);
    journal_write(journal, {{"index", index}, {"position", position}});
  }
  Workers pool(workers);
  std::fstream out(partial, std::ios::in | std::ios::out | std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot open partial output");
  out.seekp(position);
  for (size_t si = 0; si < inventory.size(); ++si) {
    const auto &src = inventory[si];
    auto &saved = index["shards"][si];
    std::ifstream in(src.at("path").get<std::string>(), std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot open source shard");
    auto data_start = uint64_t(8 + src.at("header").get<std::string>().size());
    for (size_t ti = saved.at("tensors").size(); ti < src.at("tensors").size();
         ++ti) {
      const auto &desc = src["tensors"][ti];
      json t = clean_tensor(desc);
      t["blocks"] = json::array();
      Hash tensor_hash;
      uint64_t remaining = t.at("size").get<uint64_t>();
      in.clear();
      in.seekg(data_start + t.at("start").get<uint64_t>());
      if (!in)
        throw std::runtime_error("source seek failed");
      uint64_t cols = 0;
      const auto &shape = t.at("shape");
      if (t.at("dtype") == "BF16") {
        if (shape.size() == 2 && shape[1].get<uint64_t>() &&
            shape[1].get<uint64_t>() <= chunk / 2)
          cols = shape[1].get<uint64_t>();
        else
          cols = std::min<uint64_t>(4096, remaining / 2);
      }
      std::deque<std::future<Encoded>> pending;
      auto flush = [&] {
        auto block = pending.front().get();
        pending.pop_front();
        t["blocks"].push_back({{"offset", offset(out)},
                               {"size", block.data.size()},
                               {"raw_size", block.raw_size},
                               {"codec", block.codec},
                               {"rows", block.codec == "raw" ? 0 : block.rows},
                               {"cols", block.codec == "raw" ? 0 : block.cols},
                               {"sha256", block.checksum}});
        write(out, block.data);
      };
      while (remaining) {
        uint64_t rows = cols ? std::min({uint64_t(16), chunk / (cols * 2),
                                         remaining / (cols * 2)})
                             : 0;
        if (cols && !rows) {
          cols = std::min<uint64_t>(4096, remaining / 2);
          rows = 1;
        }
        auto length = rows ? rows * cols * 2 : std::min(remaining, chunk);
        auto raw = read(in, length);
        remaining -= length;
        tensor_hash.update(raw);
        pending.push_back(pool.submit(std::move(raw), rows, cols, codec));
        if (pending.size() >= workers * 2)
          flush();
      }
      while (!pending.empty())
        flush();
      t["sha256"] = tensor_hash.finish();
      saved["tensors"].push_back(t);
      durable(out, partial);
      position = offset(out);
      journal_write(journal, {{"index", index}, {"position", position}});
      std::cerr << "compiled " << t.at("name").get<std::string>() << " ("
                << t.at("size") << " bytes)\n";
    }
  }
  auto index_offset = offset(out);
  write(out, index.dump());
  header(out, index_offset);
  durable(out, partial);
  out.close();
  verify_native(partial, source, false);
  if (link(partial.c_str(), output.c_str()) < 0)
    error("cannot publish output without overwrite");
  if (unlink(partial.c_str()) < 0 || unlink(journal.c_str()) < 0)
    error("cannot remove compilation temporary file");
  fsync_path(parent, true);
  std::cout << "created " << output << ": artifact " << fs::file_size(output)
            << " bytes\n";
}
