"""Measure experimental single-block CUDA kernels, including launch and allocation overhead."""
import ctypes
import json
import statistics
import time
from pathlib import Path
import torch
import xksv

def main():
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    native=xksv.Native(); lib=ctypes.CDLL(str(Path(__file__).resolve().parents[1]/'build'/'libxksv_cuda.so'))
    lib.xksv_cuda_decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
    lib.xksv_cuda_gemv.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
    torch.manual_seed(181); output=[]
    for rows,cols in [(16,256),(16,4096)]:
        weights=torch.randn(rows,cols).to(torch.bfloat16); raw=weights.view(torch.uint8).numpy().tobytes()
        block=native.call('xksv_pack',raw,rows,cols); buf=ctypes.create_string_buffer(block)
        x=torch.randn(cols,device='cuda'); decoded=torch.empty(rows,cols,device='cuda',dtype=torch.bfloat16); out=torch.empty(rows,device='cuda')
        def fused():
            if lib.xksv_cuda_gemv(buf,len(block),rows,cols,x.data_ptr(),out.data_ptr()): raise RuntimeError('fused GEMV failed')
            return out
        def separate():
            if lib.xksv_cuda_decode(buf,len(block),rows,cols,decoded.data_ptr()): raise RuntimeError('decode failed')
            return decoded.float() @ x
        def measure(fn):
            for _ in range(2): fn()
            torch.cuda.synchronize(); times=[]
            for _ in range(10):
                start=time.perf_counter(); value=fn(); torch.cuda.synchronize(); times.append(time.perf_counter()-start)
            return value,times
        y,fused_times=measure(fused); z,decode_times=measure(separate)
        output.append(dict(rows=rows,cols=cols,compressed_bytes=len(block),original_bytes=len(raw),
                           fused_seconds=fused_times,separate_seconds=decode_times,
                           fused_median_seconds=statistics.median(fused_times),separate_median_seconds=statistics.median(decode_times),
                           max_abs_difference=float((y-z).abs().max())))
    print(json.dumps(dict(device=torch.cuda.get_device_name(), warmup=2, repeats=10,
                          measurement='wall clock including allocations, copies, launches and synchronization',results=output),indent=2))
if __name__=='__main__':main()
