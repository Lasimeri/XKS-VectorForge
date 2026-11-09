#include "xksv/codec.hpp"
#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
namespace xksv {
namespace {
using Codes = std::array<std::string, 256>;
Codes canonical(const std::array<std::uint8_t, 256>& lens) {
  std::vector<std::pair<int,int>> ordered;
  for (int i=0;i<256;++i) if(lens[i]) ordered.emplace_back(lens[i],i);
  std::sort(ordered.begin(),ordered.end());
  Codes result{};
  std::string code;
  for (const auto& [length,symbol]:ordered) {
    if (code.empty()) code.assign(length,'0');
    else {
      int i=static_cast<int>(code.size())-1;
      while (i>=0 && code[i]=='1') code[i--]='0';
      if(i<0) throw std::runtime_error("oversubscribed Huffman codebook");
      code[i]='1';
      code.append(length-code.size(),'0');
    }
    result[symbol]=code;
  }
  return result;
}
struct Node { std::uint64_t freq; int symbol, left, right; };
}
Block encode_bf16(std::span<const std::uint16_t> words) {
  Block out; out.count=words.size();
  std::array<std::uint64_t,256> freq{};
  for(auto w:words) ++freq[(w>>7)&255];
  std::vector<Node> nodes;
  using Item=std::pair<std::uint64_t,int>;
  std::priority_queue<Item,std::vector<Item>,std::greater<Item>> queue;
  for(int i=0;i<256;++i) if(freq[i]) { int n=nodes.size(); nodes.push_back({freq[i],i,-1,-1}); queue.emplace(freq[i],n); }
  while(queue.size()>1) {
    auto [fa,a]=queue.top(); queue.pop(); auto [fb,b]=queue.top(); queue.pop();
    if(fa>std::numeric_limits<std::uint64_t>::max()-fb) throw std::overflow_error("frequency overflow");
    int n=nodes.size(); nodes.push_back({fa+fb,-1,a,b}); queue.emplace(fa+fb,n);
  }
  if(!queue.empty()) {
    std::vector<std::pair<int,int>> todo{{queue.top().second,0}};
    while(!todo.empty()) { auto [i,depth]=todo.back(); todo.pop_back();
      if(nodes[i].symbol>=0) { if(depth>255) throw std::runtime_error("Huffman tree too deep"); out.lengths[nodes[i].symbol]=static_cast<std::uint8_t>(std::max(depth,1)); }
      else { todo.emplace_back(nodes[i].left,depth+1); todo.emplace_back(nodes[i].right,depth+1); }
    }
  }
  auto codes=canonical(out.lengths); out.sign_mantissa.reserve(words.size());
  for(auto w:words) {
    out.sign_mantissa.push_back(static_cast<std::uint8_t>(((w>>8)&128)|(w&127)));
    for(char bit:codes[(w>>7)&255]) { if((out.bit_count&7)==0) out.exponents.push_back(0); if(bit=='1') out.exponents.back() |= static_cast<std::uint8_t>(0x80u>>(out.bit_count&7)); ++out.bit_count; }
  }
  return out;
}
std::vector<std::uint16_t> decode_bf16(const Block& b) {
  if(b.count!=b.sign_mantissa.size() || b.bit_count/8+(b.bit_count%8!=0)!=b.exponents.size()) throw std::runtime_error("invalid block lengths");
  auto codes=canonical(b.lengths);
  struct DNode { int child[2]={-1,-1}; int symbol=-1; };
  std::vector<DNode> tree(1);
  for(int s=0;s<256;++s) if(!codes[s].empty()) { int n=0; for(char c:codes[s]) {
    if(tree[n].symbol!=-1) throw std::runtime_error("invalid prefix code");
    int bit=c-'0'; int next=tree[n].child[bit]; if(next<0) { next=tree.size(); tree[n].child[bit]=next; tree.emplace_back(); } n=next;
  } if(tree[n].symbol!=-1 || tree[n].child[0]!=-1 || tree[n].child[1]!=-1) throw std::runtime_error("invalid codebook"); tree[n].symbol=s; }
  std::vector<std::uint16_t> result; result.reserve(b.sign_mantissa.size()); std::uint64_t pos=0;
  for(auto sm:b.sign_mantissa) { int n=0; while(tree[n].symbol<0) { if(pos>=b.bit_count) throw std::runtime_error("truncated bitstream"); int bit=(b.exponents[pos/8]>>(7-(pos%8)))&1; ++pos; n=tree[n].child[bit]; if(n<0) throw std::runtime_error("invalid bitstream"); }
    result.push_back(static_cast<std::uint16_t>(((sm&128u)<<8)|(tree[n].symbol<<7)|(sm&127u)));
  }
  if(pos!=b.bit_count) throw std::runtime_error("trailing exponent bits");
  return result;
}
}
