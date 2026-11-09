"""Compare serial and independently coded GPU substreams, including API overhead."""
import ctypes,json,statistics,time
from pathlib import Path
import torch,xksv
native=xksv.Native(); native.lib.xksv_pack_parallel.argtypes=native.lib.xksv_pack.argtypes;native.lib.xksv_pack_parallel.restype=ctypes.c_int
lib=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'build'/'libxksv_cuda.so'))
lib.xksv_cuda_decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
lib.xksv_cuda_decode_parallel.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
lib.xksv_cuda_gemv.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
lib.xksv_cuda_gemv_parallel.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
torch.manual_seed(183);result=[]
for rows,cols in [(16,256),(16,4096)]:
    words=torch.randn(rows,cols).bfloat16().view(torch.uint8).numpy().tobytes()
    serial=native.call('xksv_pack',words,rows,cols); parallel=native.call('xksv_pack_parallel',words,rows,cols)
    sb,pb=ctypes.create_string_buffer(serial),ctypes.create_string_buffer(parallel)
    out=torch.empty(rows*cols*2,device='cuda',dtype=torch.uint8)
    def run(parallel_mode):
        if parallel_mode: error=lib.xksv_cuda_decode_parallel(pb,len(parallel),out.data_ptr())
        else:error=lib.xksv_cuda_decode(sb,len(serial),rows,cols,out.data_ptr())
        if error: raise RuntimeError('decode failed')
    stats=[]
    for mode in (False,True):
        for _ in range(2):run(mode)
        durations=[]
        for _ in range(10):
            torch.cuda.synchronize();start=time.perf_counter();run(mode);torch.cuda.synchronize();durations.append(time.perf_counter()-start)
        assert out.cpu().numpy().tobytes()==words
        stats.append(durations)
    x=torch.randn(cols,device='cuda'); y=torch.empty(rows,device='cuda'); fused_stats=[]
    for mode in (False,True):
        def fused():
            status=(lib.xksv_cuda_gemv_parallel(pb,len(parallel),x.data_ptr(),y.data_ptr()) if mode else lib.xksv_cuda_gemv(sb,len(serial),rows,cols,x.data_ptr(),y.data_ptr()))
            if status: raise RuntimeError('fused GEMV failed')
        for _ in range(2): fused()
        durations=[]
        for _ in range(10):
            torch.cuda.synchronize();start=time.perf_counter();fused();torch.cuda.synchronize();durations.append(time.perf_counter()-start)
        fused_stats.append(durations)
    result.append(dict(fused_serial_seconds=fused_stats[0],fused_parallel_seconds=fused_stats[1],fused_serial_median=statistics.median(fused_stats[0]),fused_parallel_median=statistics.median(fused_stats[1]),rows=rows,cols=cols,raw_bytes=len(words),serial_bytes=len(serial),parallel_bytes=len(parallel),serial_seconds=stats[0],parallel_seconds=stats[1],serial_median=statistics.median(stats[0]),parallel_median=statistics.median(stats[1])))
print(json.dumps(dict(device=torch.cuda.get_device_name(),warmup=2,repeats=10,measurement='wall clock including allocations, copies, launches and synchronization',results=result),indent=2))
