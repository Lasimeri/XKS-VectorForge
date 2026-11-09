#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>
namespace {
struct Node {int child[2];int symbol;};
struct Chunk {std::uint64_t sm,bits,nbits,count,start;int tree;};
std::uint64_t get64(const std::uint8_t* p){std::uint64_t v=0;for(int i=0;i<8;i++)v|=std::uint64_t(p[i])<<(8*i);return v;}
__global__ void decode(const std::uint8_t* data,const Node* trees,const Chunk* chunks,std::size_t nchunks,std::size_t rows,std::size_t cols,void* out,const float* x,int* error) {
 auto tid=threadIdx.x;if(tid>=nchunks)return;auto chunk=chunks[tid];std::uint64_t pos=0;
 for(std::uint64_t i=0;i<chunk.count;i++){
  int node=chunk.tree;
  while(trees[node].symbol<0){
   if(pos>=chunk.nbits){atomicExch(error,1);return;}
   int bit=(data[chunk.bits+pos/8]>>(7-(pos%8)))&1;pos++;node=trees[node].child[bit];
   if(node<0){atomicExch(error,1);return;}
  }
  auto sm=data[chunk.sm+i];auto packed=chunk.start+i;
  auto tile=packed/(rows*16);auto width=min(std::size_t(16),cols-tile*16);auto local=packed-tile*rows*16;
  auto row=local/width,col=tile*16+local%width;
  if(row>=rows || col>=cols){atomicExch(error,1);return;}
  auto word=std::uint16_t(((sm&128u)<<8)|(trees[node].symbol<<7)|(sm&127u));
  if(x) atomicAdd(static_cast<float*>(out)+row,__uint_as_float(std::uint32_t(word)<<16)*x[col]);
  else static_cast<std::uint16_t*>(out)[row*cols+col]=word;
 }
 if(pos!=chunk.nbits)atomicExch(error,1);
}
}
static int launch_parallel(const void* input,std::size_t n,void* output,const float* x){
 auto p=static_cast<const std::uint8_t*>(input);if(!p || !output || n<32 || std::memcmp(p,"XKSP2\0\0\0",8))return 1;
 auto rows=get64(p+8),cols=get64(p+16),count=get64(p+24);
 if(!rows||rows>16||!cols||!count||count>32||cols>std::numeric_limits<std::size_t>::max()/rows||count>rows*cols||rows*cols>n||rows*cols>std::numeric_limits<std::size_t>::max()/32||n<32+8*count)return 1;
 std::vector<Node> trees;std::vector<Chunk> chunks;std::size_t pos=32+8*count;
 for(std::size_t k=0;k<count;k++){
  auto len=get64(p+32+8*k);if(len<272||len>n-pos)return 1;
  auto elements=get64(p+pos),bits=get64(p+pos+8);auto start=rows*cols*k/count;
  if(elements!=rows*cols*(k+1)/count-start||elements>len-272||bits/8+(bits%8!=0)!=len-272-elements)return 1;
  std::vector<std::pair<int,int>> ordered;for(int i=0;i<256;i++)if(p[pos+16+i])ordered.emplace_back(p[pos+16+i],i);
  std::sort(ordered.begin(),ordered.end());std::string code;int root=trees.size();trees.push_back(Node{{-1,-1},-1});
  for(auto [size,sym]:ordered){
   if(code.empty())code.assign(size,'0');else{int i=static_cast<int>(code.size())-1;while(i>=0&&code[i]=='1')code[i--]='0';if(i<0)return 1;code[i]='1';code.append(size-code.size(),'0');}
   int node=root;for(char c:code){if(trees[node].symbol>=0)return 1;int bit=c-'0',next=trees[node].child[bit];if(next<0){next=trees.size();trees[node].child[bit]=next;trees.push_back(Node{{-1,-1},-1});}node=next;}
   if(trees[node].symbol>=0||trees[node].child[0]>=0||trees[node].child[1]>=0)return 1;trees[node].symbol=sym;
  }
  chunks.push_back(Chunk{pos+272,pos+272+elements,bits,elements,start,root});pos+=len;
 }
 if(pos!=n)return 1;
 std::uint8_t* d_data=nullptr;Node* d_tree=nullptr;Chunk* d_chunks=nullptr;int* d_error=nullptr;int error=1;
 auto status=cudaMalloc(&d_data,n);
 if(status==cudaSuccess)status=cudaMalloc(&d_tree,trees.size()*sizeof(Node));
 if(status==cudaSuccess)status=cudaMalloc(&d_chunks,chunks.size()*sizeof(Chunk));
 if(status==cudaSuccess)status=cudaMalloc(&d_error,sizeof(int));
 if(status==cudaSuccess)status=cudaMemcpy(d_data,p,n,cudaMemcpyHostToDevice);
 if(status==cudaSuccess)status=cudaMemcpy(d_tree,trees.data(),trees.size()*sizeof(Node),cudaMemcpyHostToDevice);
 if(status==cudaSuccess)status=cudaMemcpy(d_chunks,chunks.data(),chunks.size()*sizeof(Chunk),cudaMemcpyHostToDevice);
 if(status==cudaSuccess)status=cudaMemset(d_error,0,sizeof(int));
 if(status==cudaSuccess && x)status=cudaMemset(output,0,rows*sizeof(float));
 if(status==cudaSuccess){decode<<<1,32>>>(d_data,d_tree,d_chunks,count,rows,cols,output,x,d_error);status=cudaGetLastError();}
 if(status==cudaSuccess)status=cudaMemcpy(&error,d_error,sizeof(int),cudaMemcpyDeviceToHost);
 if(d_error)cudaFree(d_error);if(d_chunks)cudaFree(d_chunks);if(d_tree)cudaFree(d_tree);if(d_data)cudaFree(d_data);
 return status==cudaSuccess&&error==0?0:1;
}

extern "C" int xksv_cuda_decode_parallel(const void* input,std::size_t n,void* output){
 return launch_parallel(input,n,output,nullptr);
}
extern "C" int xksv_cuda_gemv_parallel(const void* input,std::size_t n,const float* x,float* output){
 if(!x)return 1;
 return launch_parallel(input,n,output,x);
}
