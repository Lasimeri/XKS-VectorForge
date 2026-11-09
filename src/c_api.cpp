#include "xksv/codec.hpp"
#include "xksv/layout.hpp"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <vector>
#include <cstdint>
namespace {
void put64(std::vector<std::uint8_t>& v, std::uint64_t n) { for(int i=0;i<8;i++) v.push_back((n>>(8*i))&255); }
std::uint64_t get64(const std::uint8_t* p) { std::uint64_t n=0; for(int i=0;i<8;i++) n|=std::uint64_t(p[i])<<(8*i); return n; }
int copy(const std::vector<std::uint8_t>& v, void** out, std::size_t* size) {
 *size=v.size(); *out=std::malloc(v.size() ? v.size() : 1); if(!*out) return -1; std::memcpy(*out,v.data(),v.size()); return 0;
}
}
extern "C" {
void xksv_free(void* p) { std::free(p); }
int xksv_pack(const void* data, std::size_t n, std::size_t rows, std::size_t cols, void** out, std::size_t* out_size) {
 try {
  if(n%2 || rows==0 || cols>std::numeric_limits<std::size_t>::max()/rows || rows*cols!=n/2) return -1;
  auto p=static_cast<const std::uint8_t*>(data); std::vector<std::uint16_t> words(n/2);
  for(std::size_t i=0;i<words.size();i++) words[i]=p[2*i] | (std::uint16_t(p[2*i+1])<<8);
  auto tiled=xksv::tile(words,rows,cols,16,16); auto b=xksv::encode_bf16(tiled);
  std::vector<std::uint8_t> v; v.reserve(272+b.sign_mantissa.size()+b.exponents.size());
  put64(v,b.count); put64(v,b.bit_count); v.insert(v.end(),b.lengths.begin(),b.lengths.end()); v.insert(v.end(),b.sign_mantissa.begin(),b.sign_mantissa.end()); v.insert(v.end(),b.exponents.begin(),b.exponents.end());
  return copy(v,out,out_size);
 } catch(...) {return -1;}
}
int xksv_unpack(const void* data, std::size_t n, std::size_t rows, std::size_t cols, void** out, std::size_t* out_size) {
 try {
  if(n<272 || !rows || cols>std::numeric_limits<std::size_t>::max()/rows) return -1;
  auto p=static_cast<const std::uint8_t*>(data); auto count=get64(p); auto bits=get64(p+8);
  if(count!=rows*cols || count>n-272 || bits/8+(bits%8!=0)!=n-272-count) return -1;
  xksv::Block b; b.count=count; b.bit_count=bits; std::memcpy(b.lengths.data(),p+16,256);
  b.sign_mantissa.assign(p+272,p+272+count); b.exponents.assign(p+272+count,p+n);
  auto words=xksv::untile(xksv::decode_bf16(b),rows,cols,16,16); std::vector<std::uint8_t> v; v.reserve(count*2);
  for(auto w:words) { v.push_back(w&255); v.push_back(w>>8); }
  return copy(v,out,out_size);
 } catch(...) {return -1;}
}
}
// Experimental slab: independently coded substreams for parallel GPU decoding.
// Header: magic, rows, cols, chunk count; then uint64 segment sizes and segments.
extern "C" int xksv_pack_parallel(const void* data,std::size_t n,std::size_t rows,std::size_t cols,void** out,std::size_t* out_size) {
 try {
  if(n%2 || !rows || rows>16 || !cols || cols>std::numeric_limits<std::size_t>::max()/rows || rows*cols!=n/2 || rows*cols>std::numeric_limits<std::size_t>::max()/32) return -1;
  const auto* p=static_cast<const std::uint8_t*>(data); std::vector<std::uint16_t> words(n/2);
  for(std::size_t i=0;i<words.size();++i) words[i]=p[i*2] | (std::uint16_t(p[i*2+1])<<8);
  auto packed=xksv::tile(words,rows,cols,16,16);
  std::size_t chunks=std::min<std::size_t>(32,packed.size());
  std::vector<std::vector<std::uint8_t>> encoded; encoded.reserve(chunks);
  for(std::size_t i=0;i<chunks;++i) {
   auto start=packed.size()*i/chunks, end=packed.size()*(i+1)/chunks;
   auto b=xksv::encode_bf16(std::span<const std::uint16_t>(packed).subspan(start,end-start));
   std::vector<std::uint8_t> v; put64(v,b.count); put64(v,b.bit_count);
   v.insert(v.end(),b.lengths.begin(),b.lengths.end());
   v.insert(v.end(),b.sign_mantissa.begin(),b.sign_mantissa.end());
   v.insert(v.end(),b.exponents.begin(),b.exponents.end()); encoded.push_back(std::move(v));
  }
  std::vector<std::uint8_t> v{'X','K','S','P','2',0,0,0}; put64(v,rows);put64(v,cols);put64(v,chunks);
  for(auto& segment:encoded) put64(v,segment.size());
  for(auto& segment:encoded) v.insert(v.end(),segment.begin(),segment.end());
  return copy(v,out,out_size);
 } catch(...) {return -1;}
}
extern "C" int xksv_unpack_parallel(const void* data,std::size_t n,void** out,std::size_t* out_size) {
 try {
  const auto* p=static_cast<const std::uint8_t*>(data);
  if(n<32 || std::memcmp(p,"XKSP2\0\0\0",8)) return -1;
  auto rows=get64(p+8),cols=get64(p+16),chunks=get64(p+24);
  if(!rows || rows>16 || !cols || !chunks || chunks>32 || cols>std::numeric_limits<std::size_t>::max()/rows || chunks>rows*cols || rows*cols>n || rows*cols>std::numeric_limits<std::size_t>::max()/32 || n<32+8*chunks) return -1;
  std::size_t pos=32+8*chunks; std::vector<std::uint16_t> packed; packed.reserve(rows*cols);
  for(std::size_t i=0;i<chunks;++i) {
   auto length=get64(p+32+8*i); if(length<272 || length>n-pos) return -1;
   auto count=get64(p+pos), bits=get64(p+pos+8);
   if(count!=(rows*cols*(i+1)/chunks)-(rows*cols*i/chunks) || count>length-272 || bits/8+(bits%8!=0)!=length-272-count) return -1;
   xksv::Block b; b.count=count;b.bit_count=bits;std::memcpy(b.lengths.data(),p+pos+16,256);
   b.sign_mantissa.assign(p+pos+272,p+pos+272+count);b.exponents.assign(p+pos+272+count,p+pos+length);
   auto words=xksv::decode_bf16(b);packed.insert(packed.end(),words.begin(),words.end());pos+=length;
  }
  if(pos!=n) return -1;
  auto words=xksv::untile(packed,rows,cols,16,16);std::vector<std::uint8_t> result;result.reserve(words.size()*2);
  for(auto w:words){result.push_back(w&255);result.push_back(w>>8);} return copy(result,out,out_size);
 } catch(...) {return -1;}
}
