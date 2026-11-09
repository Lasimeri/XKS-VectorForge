import ctypes, os, random, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import xksv
if not torch.cuda.is_available():
    print('CUDA device unavailable: skipped')
    raise SystemExit(0)
native=xksv.Native()
lib=ctypes.CDLL(os.environ.get('XKSV_CUDA_LIBRARY',str(Path(__file__).resolve().parents[1]/'build'/'libxksv_cuda.so')))
lib.xksv_cuda_decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
lib.xksv_cuda_decode.restype=ctypes.c_int
rng=random.Random(51)
for rows,cols in [(1,1),(1,256),(7,19),(16,32),(17,33),(31,103)]:
    for mode in range(2):
        words=[rng.randrange(65536) if mode else ((127+(i%3))<<7)|rng.randrange(128) for i in range(rows*cols)]
        raw=b''.join(w.to_bytes(2,'little') for w in words)
        block=native.call('xksv_pack',raw,rows,cols)
        buf=ctypes.create_string_buffer(block)
        output=torch.empty(rows*cols*2,device='cuda',dtype=torch.uint8)
        assert lib.xksv_cuda_decode(buf,len(block),rows,cols,output.data_ptr())==0
        assert output.cpu().numpy().tobytes()==raw
print('CUDA random-access block decode matches CPU input')
lib.xksv_cuda_gemv.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
lib.xksv_cuda_gemv.restype=ctypes.c_int
for rows,cols in [(1,1),(7,19),(16,32),(17,33)]:
    weights=torch.randn(rows,cols,dtype=torch.float32).to(torch.bfloat16)
    raw=weights.view(torch.uint8).numpy().tobytes()
    block=native.call('xksv_pack',raw,rows,cols); buf=ctypes.create_string_buffer(block)
    x=torch.randn(cols,device='cuda'); out=torch.empty(rows,device='cuda')
    assert lib.xksv_cuda_gemv(buf,len(block),rows,cols,x.data_ptr(),out.data_ptr())==0
    reference=weights.float().to('cuda')@x
    assert torch.allclose(out,reference,atol=1e-5,rtol=1e-5), (out,reference)
print('experimental fused BF16 decode/GEMV agrees within FP32 tolerance')
lib.xksv_cuda_gemm.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
lib.xksv_cuda_gemm.restype=ctypes.c_int
for rows,cols,batch in [(3,7,2),(17,33,4)]:
    weights=torch.randn(rows,cols).to(torch.bfloat16)
    block=native.call('xksv_pack',weights.view(torch.uint8).numpy().tobytes(),rows,cols)
    buf=ctypes.create_string_buffer(block); x=torch.randn(cols,batch,device='cuda'); out=torch.empty(rows,batch,device='cuda')
    assert lib.xksv_cuda_gemm(buf,len(block),rows,cols,batch,x.data_ptr(),out.data_ptr())==0
    assert torch.allclose(out,weights.float().to('cuda')@x,atol=1e-5,rtol=1e-5)
print('experimental fused BF16 decode/GEMM agrees within FP32 tolerance')
native.lib.xksv_pack_parallel.argtypes=native.lib.xksv_pack.argtypes
native.lib.xksv_pack_parallel.restype=ctypes.c_int
native.lib.xksv_unpack_parallel.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_size_t)]
native.lib.xksv_unpack_parallel.restype=ctypes.c_int
lib.xksv_cuda_decode_parallel.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
lib.xksv_cuda_decode_parallel.restype=ctypes.c_int
for rows,cols in [(1,1),(1,256),(7,19),(16,32),(16,4096),(16,4097)]:
    words=[i%65536 for i in range(rows*cols)]
    raw=b''.join(w.to_bytes(2,'little') for w in words)
    block=native.call('xksv_pack_parallel',raw,rows,cols)
    buf=ctypes.create_string_buffer(block); out=ctypes.c_void_p(); size=ctypes.c_size_t()
    assert native.lib.xksv_unpack_parallel(buf,len(block),ctypes.byref(out),ctypes.byref(size))==0
    try: assert ctypes.string_at(out,size.value)==raw
    finally: native.lib.xksv_free(out)
    output=torch.empty(rows*cols*2,device='cuda',dtype=torch.uint8)
    assert lib.xksv_cuda_decode_parallel(buf,len(block),output.data_ptr())==0
    assert output.cpu().numpy().tobytes()==raw
print('parallel substream decoder matches all tested BF16 bits')
truncated=block[:-1]; buf=ctypes.create_string_buffer(truncated)
assert lib.xksv_cuda_decode_parallel(buf,len(truncated),output.data_ptr())!=0
out=ctypes.c_void_p(); size=ctypes.c_size_t()
assert native.lib.xksv_unpack_parallel(buf,len(truncated),ctypes.byref(out),ctypes.byref(size))!=0
lib.xksv_cuda_gemv_parallel.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
lib.xksv_cuda_gemv_parallel.restype=ctypes.c_int
for rows,cols in [(1,1),(7,19),(16,256),(16,4096)]:
    weights=torch.randn(rows,cols).bfloat16()
    blob=native.call('xksv_pack_parallel',weights.view(torch.uint8).numpy().tobytes(),rows,cols)
    buf=ctypes.create_string_buffer(blob); x=torch.randn(cols,device='cuda'); out=torch.empty(rows,device='cuda')
    assert lib.xksv_cuda_gemv_parallel(buf,len(blob),x.data_ptr(),out.data_ptr())==0
    assert torch.allclose(out,weights.float().to('cuda')@x,atol=1e-3,rtol=1e-3)
print('parallel fused decode/GEMV agrees within FP32 tolerance')
