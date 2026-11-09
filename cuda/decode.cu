#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>
namespace {
struct Node { int child[2]; int symbol; };
std::uint64_t get64(const std::uint8_t* p) {std::uint64_t x=0; for(int i=0;i<8;++i)x|=std::uint64_t(p[i])<<(8*i); return x;}
__global__ void decode_kernel(const std::uint8_t* sm, const std::uint8_t* bits, std::uint64_t nbits, const Node* tree,
                              std::size_t rows, std::size_t cols, std::uint16_t* out, int* error) {
 if(blockIdx.x || threadIdx.x) return;
 std::uint64_t pos=0; std::size_t idx=0;
 for(std::size_t r=0;r<rows;r+=16) for(std::size_t c=0;c<cols;c+=16)
  for(std::size_t i=0;i<16 && r+i<rows;++i) for(std::size_t j=0;j<16 && c+j<cols;++j) {
   int node=0;
   while(tree[node].symbol<0) {
    if(pos>=nbits) {*error=1;return;}
    int bit=(bits[pos/8]>>(7-(pos%8)))&1; ++pos; node=tree[node].child[bit];
    if(node<0) {*error=1;return;}
   }
   auto value=sm[idx++]; out[(r+i)*cols+c+j]=std::uint16_t(((value&128u)<<8)|(tree[node].symbol<<7)|(value&127u));
  }
 if(pos!=nbits) *error=1;
}
__global__ void gemv_kernel(const std::uint8_t* sm, const std::uint8_t* bits, std::uint64_t nbits, const Node* tree,
                            std::size_t rows, std::size_t cols, std::size_t batch, const float* x, float* out, int* error) {
 if(blockIdx.x || threadIdx.x) return;
 for(std::size_t r=0;r<rows*batch;++r) out[r]=0.0f;
 std::uint64_t pos=0; std::size_t idx=0;
 for(std::size_t r=0;r<rows;r+=16) for(std::size_t c=0;c<cols;c+=16)
  for(std::size_t i=0;i<16 && r+i<rows;++i) for(std::size_t j=0;j<16 && c+j<cols;++j) {
   int node=0;
   while(tree[node].symbol<0) {
    if(pos>=nbits) {*error=1;return;}
    int bit=(bits[pos/8]>>(7-(pos%8)))&1; ++pos; node=tree[node].child[bit];
    if(node<0) {*error=1;return;}
   }
   auto v=sm[idx++]; auto word=std::uint32_t(((v&128u)<<8)|(tree[node].symbol<<7)|(v&127u));
   for(std::size_t k=0;k<batch;++k) out[(r+i)*batch+k]+=__uint_as_float(word<<16)*x[(c+j)*batch+k];
  }
 if(pos!=nbits) *error=1;
}
}
static int launch(const void* input,std::size_t size,std::size_t rows,std::size_t cols,void* output,const float* x,std::size_t batch) {
 const auto* p=static_cast<const std::uint8_t*>(input);
 if(!p || !output || size<272 || !rows || !cols || !batch || cols>std::numeric_limits<std::size_t>::max()/rows || batch>std::numeric_limits<std::size_t>::max()/rows || batch>std::numeric_limits<std::size_t>::max()/cols) return 1;
 auto n=get64(p), nbits=get64(p+8);
 if(n!=rows*cols || n>size-272 || nbits/8+(nbits%8!=0)!=size-272-n) return 1;
 std::vector<std::pair<int,int>> ordered;
 for(int i=0;i<256;++i) if(p[16+i]) ordered.emplace_back(p[16+i],i);
 std::sort(ordered.begin(),ordered.end());
 std::vector<Node> tree(1,Node{{-1,-1},-1}); std::string code;
 for(auto [len,sym]:ordered) {
  if(code.empty()) code.assign(len,'0');
  else {int i=static_cast<int>(code.size())-1; while(i>=0 && code[i]=='1')code[i--]='0'; if(i<0)return 1; code[i]='1'; code.append(len-code.size(),'0');}
  int cur=0;
  for(char ch:code) {
   if(tree[cur].symbol>=0)return 1; int bit=ch-'0'; int next=tree[cur].child[bit];
   if(next<0){next=static_cast<int>(tree.size());tree[cur].child[bit]=next;tree.push_back(Node{{-1,-1},-1});} cur=next;
  }
  if(tree[cur].symbol>=0 || tree[cur].child[0]>=0 || tree[cur].child[1]>=0)return 1;
  tree[cur].symbol=sym;
 }
 std::uint8_t* d_data=nullptr; Node* d_tree=nullptr; int* d_error=nullptr; int error=1; cudaError_t status;
 auto payload=size-272;
 status=cudaMalloc(&d_data,payload ? payload : 1);
 if(status==cudaSuccess) status=cudaMalloc(&d_tree,tree.size()*sizeof(Node));
 if(status==cudaSuccess) status=cudaMalloc(&d_error,sizeof(int));
 if(status==cudaSuccess) status=cudaMemcpy(d_data,p+272,payload,cudaMemcpyHostToDevice);
 if(status==cudaSuccess) status=cudaMemcpy(d_tree,tree.data(),tree.size()*sizeof(Node),cudaMemcpyHostToDevice);
 if(status==cudaSuccess) status=cudaMemset(d_error,0,sizeof(int));
 if(status==cudaSuccess) {
  if(x) gemv_kernel<<<1,1>>>(d_data,d_data+n,nbits,d_tree,rows,cols,batch,x,static_cast<float*>(output),d_error);
  else decode_kernel<<<1,1>>>(d_data,d_data+n,nbits,d_tree,rows,cols,static_cast<std::uint16_t*>(output),d_error);
  status=cudaGetLastError();
 }
 if(status==cudaSuccess) status=cudaMemcpy(&error,d_error,sizeof(int),cudaMemcpyDeviceToHost);
 if(d_error) cudaFree(d_error); if(d_tree) cudaFree(d_tree); if(d_data) cudaFree(d_data);
 return status==cudaSuccess && error==0 ? 0 : 1;
}

extern "C" int xksv_cuda_decode(const void* input,std::size_t size,std::size_t rows,std::size_t cols,void* output) {
 return launch(input,size,rows,cols,output,nullptr,1);
}
// Experimental FP32 accumulation; arithmetic ordering is not promised bit-identical.
extern "C" int xksv_cuda_gemv(const void* input,std::size_t size,std::size_t rows,std::size_t cols,const float* x,float* output) {
 if(!x) return 1;
 return launch(input,size,rows,cols,output,x,1);
}

// X and output are row-major [cols, batch] and [rows, batch].
extern "C" int xksv_cuda_gemm(const void* input,std::size_t size,std::size_t rows,std::size_t cols,std::size_t batch,const float* x,float* output) {
 if(!x) return 1;
 return launch(input,size,rows,cols,output,x,batch);
}
