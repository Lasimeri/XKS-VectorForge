#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;
void verify_native(const fs::path &, const fs::path &, bool = false);
void compile_native(const fs::path &, const fs::path &, const std::string &, bool, unsigned);
namespace {
constexpr uint64_t max_header = 100000000;
const std::map<std::string, uint64_t> sizes = {{"BOOL",1},{"U8",1},{"I8",1},{"F8_E4M3",1},{"F8_E5M2",1},{"I16",2},{"U16",2},{"F16",2},{"BF16",2},{"I32",4},{"U32",4},{"F32",4},{"I64",8},{"U64",8},{"F64",8}};
uint64_t number(const json &j, const std::string &what) {
    if (!j.is_number_integer() || (j.is_number_integer() && !j.is_number_unsigned() && j.get<int64_t>() < 0)) throw std::runtime_error("invalid " + what);
    return j.get<uint64_t>();
}
uint64_t multiply(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max()/b) throw std::runtime_error("tensor size overflow");
    return a*b;
}
void read_exact(std::istream &f, char *data, size_t size) {
    if (size && !f.read(data, static_cast<std::streamsize>(size))) throw std::runtime_error("unexpected end of file");
}
std::string role(const std::string &n) {
    auto has=[&](const char *s){return n.find(s)!=std::string::npos;};
    if(has("embed_tokens")||has("word_embeddings")) return "token_embedding";
    if(has("lm_head")||has("output_projection")) return "language_model_head";
    if(has("visual")||has("vision")) return "vision_component";
    if(has("linear_attn")) { if(has("conv")) return "deltanet_convolution"; if(has("in_proj")) return "deltanet_input_projection"; if(has("out_proj")) return "deltanet_output_projection"; return "deltanet_state_parameter"; }
    if(has("self_attn")||has(".attention.")) return "attention_parameter";
    if(has("expert")) return "moe_expert";
    if(has("router")) return "moe_router";
    if(has(".mlp.")||has(".feed_forward.")) return "feed_forward_parameter";
    if(has("norm")) return "normalization_parameter";
    return "other";
}
json parse_json(const std::string &text) {
    std::map<int,std::set<std::string>> keys;
    return json::parse(text,[&](int depth, json::parse_event_t event, json &value) {
        if(event == json::parse_event_t::object_start) keys[depth+1].clear();
        if(event == json::parse_event_t::key && !keys[depth].insert(value.get<std::string>()).second) throw std::runtime_error("duplicate JSON key");
        return true;
    });
}
json stats(std::ifstream &f, uint64_t pos, uint64_t size) {
    std::array<uint64_t,256> hist{}; std::array<uint64_t,16> bits{};
    std::vector<unsigned char> buf(1<<20); f.clear(); f.seekg(static_cast<std::streamoff>(pos));
    while(size) { size_t n=static_cast<size_t>(std::min<uint64_t>(size,buf.size())); read_exact(f,reinterpret_cast<char*>(buf.data()),n); size-=n;
        for(size_t i=0;i<n;i+=2) { uint16_t w=uint16_t(buf[i]) | (uint16_t(buf[i+1])<<8); ++hist[(w>>7)&255]; for(int b=0;b<16;++b) bits[b]+=(w>>b)&1; }
    }
    uint64_t count=0; for(auto n:hist) count+=n; double entropy=0;
    if(count) for(auto n:hist) if(n) { double p=double(n)/count; entropy-=p*std::log2(p); }
    return {{"exponent_histogram",hist},{"bit_ones",bits},{"exponent_entropy_bits",entropy},{"ideal_payload_bytes",double(count)*(8+entropy)/8}};
}
json inspect_impl(const fs::path &source, bool with_stats) {
    std::vector<fs::path> files; std::map<std::string,std::string> mapping, actual;
    json shards=json::array();
    const fs::path index=source/"model.safetensors.index.json";
    if(fs::is_directory(source) && fs::is_regular_file(index)) {
        std::ifstream f(index); std::string text((std::istreambuf_iterator<char>(f)),{}); auto j=parse_json(text);
        if(!j.contains("weight_map") || !j["weight_map"].is_object()) throw std::runtime_error("invalid shard index");
        std::set<std::string> names;
        for(auto it=j["weight_map"].begin();it!=j["weight_map"].end();++it) {
            if(!it.value().is_string()) throw std::runtime_error("invalid shard name");
            auto name=it.value().get<std::string>(); fs::path p(name);
            if(p.filename()!=p || p.extension()!=".safetensors") throw std::runtime_error("invalid shard name");
            mapping[it.key()]=name; names.insert(name);
        }
        for(auto &name:names) files.push_back(source/name);
    } else if(fs::is_directory(source)) {
        for(auto &entry:fs::directory_iterator(source)) if(entry.path().extension()==".safetensors") files.push_back(entry.path());
        std::sort(files.begin(),files.end());
    } else files.push_back(source);
    if(files.empty()) throw std::runtime_error("no Safetensors shards found");
    std::set<std::string> all_names;
    for(auto &path:files) {
        if(!fs::is_regular_file(path)||path.extension()!=".safetensors") throw std::runtime_error("invalid Safetensors shard: "+path.string());
        uint64_t file_size=fs::file_size(path); if(file_size<8) throw std::runtime_error("truncated header");
        std::ifstream f(path,std::ios::binary); std::array<unsigned char,8> prefix{}; read_exact(f,reinterpret_cast<char*>(prefix.data()),8);
        uint64_t hlen=0; for(unsigned i=0;i<8;++i) hlen|=uint64_t(prefix[i])<<(8*i);
        if(hlen>max_header || hlen>file_size-8) throw std::runtime_error("invalid header length");
        std::string header(hlen,'\0'); read_exact(f,header.data(),header.size()); json root=parse_json(header);
        if(!root.is_object()) throw std::runtime_error("invalid Safetensors header");
        std::vector<json> tensors; std::vector<std::pair<uint64_t,uint64_t>> ranges;
        for(auto it=root.begin();it!=root.end();++it) {
            auto name=it.key(); if(name=="__metadata__") continue;
            if(!all_names.insert(name).second) throw std::runtime_error("duplicate tensor: "+name);
            const auto &d=it.value(); if(!d.is_object() || !d.contains("dtype") || !d["dtype"].is_string() || !d.contains("shape") || !d["shape"].is_array() || !d.contains("data_offsets") || !d["data_offsets"].is_array() || d["data_offsets"].size()!=2) throw std::runtime_error("invalid descriptor: "+name);
            auto dtype=d["dtype"].get<std::string>(); auto sz=sizes.find(dtype); if(sz==sizes.end()) throw std::runtime_error("unsupported dtype: "+dtype);
            std::vector<uint64_t> shape, strides; uint64_t count=1;
            for(const auto &dim:d["shape"]) { auto n=number(dim,"dimension"); shape.push_back(n); count=multiply(count,n); }
            uint64_t stride=1; for(auto i=shape.rbegin();i!=shape.rend();++i) { strides.push_back(stride); stride=multiply(stride,*i); } std::reverse(strides.begin(),strides.end());
            uint64_t start=number(d["data_offsets"][0],"offset"), end=number(d["data_offsets"][1],"offset");
            if(end<start || end>file_size-8-hlen || end-start!=multiply(count,sz->second)) throw std::runtime_error("invalid tensor extent: "+name);
            json t={{"shard",path.filename().string()},{"name",name},{"dtype",dtype},{"shape",shape},{"strides",strides},{"role",role(name)},{"start",start},{"size",end-start}};
            tensors.push_back(std::move(t)); if(end>start) ranges.emplace_back(start,end); actual[name]=path.filename().string();
        }
        std::sort(ranges.begin(),ranges.end()); uint64_t cursor=0; for(auto [start,end]:ranges) { if(start!=cursor) throw std::runtime_error("overlapping or noncontiguous tensor data"); cursor=end; }
        if(cursor!=file_size-8-hlen) throw std::runtime_error("unreferenced tensor data");
        std::sort(tensors.begin(),tensors.end(),[](const json &a,const json &b){return a["start"].get<uint64_t>()<b["start"].get<uint64_t>();});
        for(auto &t:tensors) if(with_stats) t["statistics"]=t["dtype"]=="BF16" ? stats(f,8+hlen+t["start"].get<uint64_t>(),t["size"].get<uint64_t>()) : json(nullptr);
        shards.push_back({{"name",path.filename().string()},{"path",path.string()},{"header",header},{"tensors",tensors}});
    }
    if(!mapping.empty() && mapping!=actual) throw std::runtime_error("shard index does not match tensor names and files");
    return shards;
}
}
json inventory_native(const fs::path &source, bool with_stats) { return inspect_impl(source,with_stats); }
int main(int argc,char **argv) {
    try {
        if(argc>=3 && std::string(argv[1])=="inspect" && (argc==3 || (argc==4 && std::string(argv[3])=="--stats"))) { for(const auto &s:inventory_native(argv[2],argc==4)) for(const auto &t:s.at("tensors")) std::cout<<t.dump()<<'\n'; return 0; }
        if(argc>=3 && std::string(argv[1])=="verify" && (argc==3 || (argc==5 && std::string(argv[3])=="--source"))) { verify_native(argv[2],argc==5 ? fs::path(argv[4]) : fs::path()); return 0; }
        if(argc>=3 && std::string(argv[1])=="compile") {
            fs::path output; std::string codec="dfloat11"; bool resume=false; unsigned workers=4;
            for(int i=3;i<argc;++i) { std::string arg=argv[i];
                if(arg=="--output" && i+1<argc) output=argv[++i];
                else if(arg=="--codec" && i+1<argc) codec=argv[++i];
                else if(arg=="--workers" && i+1<argc) { std::string value=argv[++i]; size_t used=0; auto n=std::stoul(value,&used); if(used!=value.size() || n<1 || n>32) throw std::runtime_error("workers must be between 1 and 32"); workers=unsigned(n); }
                else if(arg=="--resume") resume=true;
                else throw std::runtime_error("unknown or incomplete compile option: "+arg);
            }
            if(output.empty() || (codec!="dfloat11" && codec!="dfloat11-parallel")) throw std::runtime_error("compile requires --output and a supported --codec");
            compile_native(argv[2],output,codec,resume,workers); return 0;
        }
        std::cerr<<"usage: xksv-native inspect CHECKPOINT [--stats]\n       xksv-native compile CHECKPOINT --output ARCHIVE [--codec dfloat11|dfloat11-parallel] [--workers N] [--resume]\n       xksv-native verify ARCHIVE [--source CHECKPOINT_DIR]\n"; return 2;
    } catch(const std::exception &e) { std::cerr<<"xksv-native: "<<e.what()<<'\n'; return 1; }
}
