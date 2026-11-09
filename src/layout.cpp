#include "xksv/layout.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define XKSV_X86_DISPATCH 1
#endif
namespace xksv {
namespace {
using Copy = void (*)(std::uint16_t*,const std::uint16_t*,std::size_t);
void scalar_copy(std::uint16_t* out,const std::uint16_t* in,std::size_t count) {
 for(std::size_t i=0;i<count;++i) out[i]=in[i];
}
#ifdef XKSV_X86_DISPATCH
__attribute__((target("avx2"))) void avx2_copy(std::uint16_t* out,const std::uint16_t* in,std::size_t count) {
 std::size_t i=0;
 for(;count-i>=16;i+=16) {
  auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in+i));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),value);
 }
 for(;i<count;++i) out[i]=in[i];
}
#endif
Copy dispatch() {
#ifdef XKSV_X86_DISPATCH
 if(!std::getenv("XKSV_FORCE_SCALAR") && __builtin_cpu_supports("avx2")) return avx2_copy;
#endif
 return scalar_copy;
}
std::vector<std::uint16_t> transform(std::span<const std::uint16_t> input,std::size_t rows,std::size_t cols,std::size_t tr,std::size_t tc,bool inverse) {
 if(!tr || !tc || (cols && rows>std::numeric_limits<std::size_t>::max()/cols) || input.size()!=rows*cols) throw std::invalid_argument("invalid matrix dimensions");
 std::vector<std::uint16_t> output(input.size()); std::size_t packed=0; auto copy=dispatch();
 for(std::size_t r=0;r<rows;) {
  auto nr=std::min(tr,rows-r);
  for(std::size_t c=0;c<cols;) {
   auto nc=std::min(tc,cols-c);
   for(std::size_t i=0;i<nr;++i) {
    auto original=(r+i)*cols+c;
    if(inverse) copy(output.data()+original,input.data()+packed,nc);
    else copy(output.data()+packed,input.data()+original,nc);
    packed+=nc;
   }
   c+=nc;
  }
  r+=nr;
 }
 return output;
}
}
std::vector<std::uint16_t> tile(std::span<const std::uint16_t> x,std::size_t r,std::size_t c,std::size_t tr,std::size_t tc){return transform(x,r,c,tr,tc,false);}
std::vector<std::uint16_t> untile(std::span<const std::uint16_t> x,std::size_t r,std::size_t c,std::size_t tr,std::size_t tc){return transform(x,r,c,tr,tc,true);}
}
