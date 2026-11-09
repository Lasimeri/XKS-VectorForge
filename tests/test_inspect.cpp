#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
namespace fs=std::filesystem;
static void shard(const fs::path &p,const std::string &header,const std::string &body) {
    std::ofstream f(p,std::ios::binary); uint64_t n=header.size();
    for(int i=0;i<8;++i) f.put(char(n>>(i*8)));
    f<<header<<body;
}
static int run(const char *exe,const fs::path &path,bool stats=false) {
    auto pid=fork(); if(pid<0) throw std::runtime_error("fork failed");
    if(!pid) { int fd=open("/dev/null",O_WRONLY); dup2(fd,1); dup2(fd,2); close(fd);
        if(stats) execl(exe,exe,"inspect",path.c_str(),"--stats",nullptr); else execl(exe,exe,"inspect",path.c_str(),nullptr); _exit(127); }
    int status=0; if(waitpid(pid,&status,0)<0 || !WIFEXITED(status)) throw std::runtime_error("inspect did not exit"); return WEXITSTATUS(status);
}
static int verify(const char *exe,const fs::path &archive,const fs::path &source) {
    auto pid=fork(); if(pid<0) throw std::runtime_error("fork failed");
    if(!pid) { int fd=open("/dev/null",O_WRONLY); dup2(fd,1); dup2(fd,2); close(fd);
        execl(exe,exe,"verify",archive.c_str(),"--source",source.c_str(),nullptr); _exit(127); }
    int status=0; if(waitpid(pid,&status,0)<0 || !WIFEXITED(status)) throw std::runtime_error("verify did not exit"); return WEXITSTATUS(status);
}
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    std::array<char,32> temp{}; std::string pattern="/tmp/xksv-native-test-XXXXXX"; std::copy(pattern.begin(),pattern.end(),temp.begin());
    auto dir=mkdtemp(temp.data()); if(!dir) return 2; fs::path p(dir);
    try {
        auto file=p/"model.safetensors";
        shard(file,R"({"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",std::string("\0\x80\x80\x7f",4));
        if(run(argv[1],file) || run(argv[1],file,true)) throw std::runtime_error("valid file rejected");
        shard(file,R"({"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",std::string(4,'\0'));
        if(!run(argv[1],file)) throw std::runtime_error("duplicate key accepted");
        shard(file,R"({"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"y":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}})",std::string(4,'\0'));
        if(!run(argv[1],file)) throw std::runtime_error("overlap accepted");
        shard(file,R"({"x":{"dtype":"BF16","shape":[18446744073709551615,2],"data_offsets":[0,0]}})","");
        if(!run(argv[1],file)) throw std::runtime_error("overflow accepted");
        shard(file,R"({"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",std::string(4,'\0'));
        {std::ofstream f(p/"model.safetensors.index.json"); f<<R"({"weight_map":{"weight":"../model.safetensors"}})";}
        if(!run(argv[1],p)) throw std::runtime_error("index path traversal accepted");
        {std::ofstream f(p/"model.safetensors.index.json"); f<<R"({"weight_map":{"weight":"model.safetensors"}})";}
        if(run(argv[1],p)) throw std::runtime_error("valid shard index rejected");
        auto body=std::string("\0\x80\x80\x7f",4);
        shard(file,R"({"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",body);
        auto archive=p/"model.xksv";
        {
            std::ofstream f(archive,std::ios::binary);
            f.write("XKSV\0\1\r\n",8); f.put(char(20)); for(int i=1;i<8;++i) f.put(0); f.write(body.data(),body.size());
            f<<R"({"version":1,"format":"safetensors","assets":[],"shards":[{"name":"model.safetensors","header":"eyJ3ZWlnaHQiOnsiZHR5cGUiOiJCRjE2Iiwic2hhcGUiOlsyXSwiZGF0YV9vZmZzZXRzIjpbMCw0XX19","tensors":[{"name":"weight","dtype":"BF16","shape":[2],"start":0,"size":4,"sha256":"3cace997725d7033d64fe0a4f2f83bda7c3c48ce1e99a791546734795fc813bb","blocks":[{"offset":16,"size":4,"raw_size":4,"codec":"raw","sha256":"3cace997725d7033d64fe0a4f2f83bda7c3c48ce1e99a791546734795fc813bb"}]}]}]})";
        }
        if(verify(argv[1],archive,p)) throw std::runtime_error("valid archive rejected");
        {std::fstream f(archive,std::ios::in|std::ios::out|std::ios::binary); f.seekp(16); f.put(1);}
        if(!verify(argv[1],archive,p)) throw std::runtime_error("corrupted archive accepted");
        fs::remove_all(p); return 0;
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; fs::remove_all(p); return 1; }
}
