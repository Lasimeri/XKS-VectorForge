"""Conservative GPU block-decoder autotuning, not a model-throughput optimizer."""
import ctypes
import os
from pathlib import Path
import statistics
import time

class GpuTuner:
    def __init__(self, profile=None):
        import torch
        if not torch.cuda.is_available(): raise ValueError('autotuning requires CUDA-enabled PyTorch and an available GPU')
        self.torch=torch; self.device=torch.cuda.current_device(); props=torch.cuda.get_device_properties(self.device)
        identity={'name':props.name,'compute_capability':[props.major,props.minor]}
        if profile is not None and profile.get('device')!=identity: raise ValueError('resume GPU differs from the autotuning GPU')
        self.profile=profile if profile is not None else {'device':identity,'method':'median of three GPU block-decode wall times after warmup; parallel limited to 15% larger than serial','choices':{}}
        self.lib=ctypes.CDLL(os.environ.get('XKSV_CUDA_LIBRARY',str(Path(__file__).resolve().parents[1]/'build'/'libxksv_cuda.so')))
        self.lib.xksv_cuda_decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
        self.lib.xksv_cuda_decode_parallel.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
        self.lib.xksv_cuda_decode.restype=self.lib.xksv_cuda_decode_parallel.restype=ctypes.c_int
    def select(self,serial,parallel,raw,rows,cols):
        if len(serial)>=len(raw) and len(parallel)>=len(raw): return 'raw',raw
        if len(serial)>=len(raw): return 'bf16-huffman-parallel32',parallel
        if len(parallel)>=len(raw) or len(parallel)*100>len(serial)*115: return 'bf16-huffman-tile16',serial
        # Group similar distributions so measurements are amortized across slabs.
        key=f'{rows}:{cols}:{len(serial)*16//len(raw)}'
        choice=self.profile['choices'].get(key)
        if choice is None:
            sb,pb=ctypes.create_string_buffer(serial),ctypes.create_string_buffer(parallel)
            out=self.torch.empty(len(raw),device=f'cuda:{self.device}',dtype=self.torch.uint8)
            def run(par):
                if par: status=self.lib.xksv_cuda_decode_parallel(pb,len(parallel),out.data_ptr())
                else: status=self.lib.xksv_cuda_decode(sb,len(serial),rows,cols,out.data_ptr())
                if status: raise ValueError('GPU autotuning decode failed')
            times=[]
            for par in (False,True):
                run(par); self.torch.cuda.synchronize(self.device); samples=[]
                for _ in range(3):
                    begin=time.perf_counter();run(par);self.torch.cuda.synchronize(self.device);samples.append(time.perf_counter()-begin)
                times.append(statistics.median(samples))
            choice={'serial_seconds':times[0],'parallel_seconds':times[1],'prefer_parallel':times[1]<times[0]}
            self.profile['choices'][key]=choice
        return ('bf16-huffman-parallel32',parallel) if choice['prefer_parallel'] else ('bf16-huffman-tile16',serial)
