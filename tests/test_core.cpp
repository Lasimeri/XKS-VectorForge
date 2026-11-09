#include "xksv/codec.hpp"
#include "xksv/layout.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
void check(bool v){if(!v) throw std::runtime_error("test failed");}
int main(){
 std::vector<std::uint16_t> all(65536); std::iota(all.begin(),all.end(),0);
 check(xksv::decode_bf16(xksv::encode_bf16(all))==all);
 std::mt19937 gen(12345); std::shuffle(all.begin(),all.end(),gen);
 for(std::size_t n: {0u,1u,2u,255u,4096u,65536u}) { std::vector<std::uint16_t> v(all.begin(),all.begin()+n); check(xksv::decode_bf16(xksv::encode_bf16(v))==v); }
 std::vector<std::uint16_t> same(1000,0x8000); auto b=xksv::encode_bf16(same); check(xksv::decode_bf16(b)==same);
 b.bit_count--; bool caught=false; try { (void)xksv::decode_bf16(b); } catch(const std::runtime_error&) {caught=true;} check(caught);
 for(std::size_t rows: {0u,1u,7u,32u,33u}) for(std::size_t cols:{0u,1u,9u,32u,35u}) {std::vector<std::uint16_t> v(rows*cols); std::iota(v.begin(),v.end(),0); auto t=xksv::tile(v,rows,cols,8,16); check(xksv::untile(t,rows,cols,8,16)==v); check(xksv::decode_bf16(xksv::encode_bf16(t))==t);}
 std::cout<<"codec and layout tests passed\n";
}
