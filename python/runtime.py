"""Transformers fallback: load compiled tensors directly, decoding on the CPU."""
import ctypes
import gc
from collections.abc import Mapping
import json
import os
import re
import resource
import sys
import struct
from pathlib import Path
import tempfile
import time
import xksv

DTYPE_MAP = {'BOOL':'bool','U8':'uint8','I8':'int8','I16':'int16','F16':'float16','BF16':'bfloat16','I32':'int32','F32':'float32','I64':'int64','F64':'float64'}


def synchronize(torch, device):
    if device.startswith('cuda'): torch.cuda.synchronize()


def backend_class(config):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    architectures=getattr(config,'architectures',None) or []
    if any(name.endswith('ForConditionalGeneration') for name in architectures):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def load_archive(path, device, gpu_decode=False, match_backend_dtype=False):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    xksv.verify(path)
    native=xksv.Native()
    gpu = None
    if gpu_decode:
        if not device.startswith('cuda'): xksv.fail('GPU decode requires a CUDA device')
        torch.cuda.set_device(torch.device(device).index or 0)
        gpu=ctypes.CDLL(os.environ.get('XKSV_CUDA_LIBRARY',str(Path(__file__).resolve().parent.parent/'build'/'libxksv_cuda.so')))
        gpu.xksv_cuda_decode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
        gpu.xksv_cuda_decode.restype=ctypes.c_int
        gpu.xksv_cuda_decode_parallel.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
        gpu.xksv_cuda_decode_parallel.restype=ctypes.c_int
    assets=tempfile.TemporaryDirectory(prefix='xksv-assets-')
    try:
        with open(path,'rb') as f:
            index, _=xksv.load_index(f)
            for a in index.get('assets',[]):
                f.seek(a['offset']); (Path(assets.name)/a['name']).write_bytes(xksv.read_exact(f,a['size']))
            if not (Path(assets.name)/'config.json').is_file(): xksv.fail('archive lacks config.json')
            config=AutoConfig.from_pretrained(assets.name,local_files_only=True,trust_remote_code=False)
            source_bytes=sum(t['size'] for shard in index['shards'] for t in shard['tensors'])
            if device.startswith('cuda'):
                free,_=torch.cuda.mem_get_info(torch.device(device))
                if source_bytes > free:
                    xksv.fail(f'weights need at least {source_bytes} GPU bytes; only {free} are free and streaming is not implemented')
            # The backend initially constructs CPU weights, usually in FP32.
            available_pages=os.sysconf('SC_AVPHYS_PAGES'); page_size=os.sysconf('SC_PAGE_SIZE')
            if source_bytes*2 > available_pages*page_size:
                xksv.fail('insufficient available host RAM for the Transformers fallback initialization')
            configured_text_dtype=getattr(getattr(config,'text_config',None),'dtype',None)
            # CPU initialization preserves non-persistent buffers initialized by the backend.
            model=backend_class(config).from_config(config,trust_remote_code=False)
            model.eval()
            backend_dtype=None
            if match_backend_dtype:
                if getattr(config,'model_type',None)!='qwen3_5': xksv.fail('backend dtype matching is currently implemented only for Qwen3.5')
                if getattr(model,'_keep_in_fp32_modules',None) or getattr(model,'_keep_in_fp32_modules_strict',None): xksv.fail('backend has unsupported FP32 preservation rules')
                backend_dtype=configured_text_dtype
                if backend_dtype not in (torch.float16,torch.bfloat16,torch.float32): xksv.fail('unsupported backend dtype')
            expected=set(model.state_dict().keys())
            aliases={}
            for n,p in model.named_parameters(remove_duplicate=False): aliases.setdefault(id(p),set()).add(n)
            supplied=set(); unused_auxiliary=[]
            for shard in index['shards']:
                for t in shard['tensors']:
                    name=t['name']
                    if name not in expected:
                        if getattr(config,'model_type',None)=='qwen3_5' and name.startswith('mtp.'):
                            unused_auxiliary.append(name)
                            continue
                        xksv.fail(f'backend does not accept tensor {name}')
                    dtype=DTYPE_MAP.get(t['dtype'])
                    if dtype is None: xksv.fail(f'backend cannot load dtype {t["dtype"]}')
                    if gpu:
                        tensor=torch.empty(t['shape'],dtype=getattr(torch,dtype),device=device)
                        byte_offset=0
                        for b in t['blocks']:
                            f.seek(b['offset']); data=xksv.read_exact(f,b['size'])
                            if b['codec'] in ('bf16-huffman-tile16','bf16-huffman-parallel32'):
                                if b['codec']=='bf16-huffman-parallel32' and (len(data)<24 or struct.unpack_from('<QQ',data,8)!=(b['rows'],b['cols'])): xksv.fail('parallel block dimensions disagree with index')
                                buf=ctypes.create_string_buffer(data)
                                error=(gpu.xksv_cuda_decode_parallel(buf,len(data),tensor.data_ptr()+byte_offset) if b['codec']=='bf16-huffman-parallel32' else gpu.xksv_cuda_decode(buf,len(data),b['rows'],b['cols'],tensor.data_ptr()+byte_offset))
                                if error: xksv.fail(f"GPU block decode failed: {name}")
                            else:
                                view=tensor.view(torch.uint8).reshape(-1)
                                view[byte_offset:byte_offset+len(data)].copy_(torch.frombuffer(bytearray(data),dtype=torch.uint8).to(device))
                            byte_offset+=b['raw_size']
                    else:
                        raw=bytearray()
                        for b in t['blocks']:
                            f.seek(b['offset']); data=xksv.read_exact(f,b['size'])
                            if b['codec']=='bf16-huffman-tile16': data=native.call('xksv_unpack',data,b['rows'],b['cols'])
                            elif b['codec']=='bf16-huffman-parallel32': data=native.unpack_parallel(data)
                            raw.extend(data)
                        tensor=torch.frombuffer(raw,dtype=getattr(torch,dtype)).reshape(t['shape']).to(device=device,copy=True) if raw else torch.empty(t['shape'],dtype=getattr(torch,dtype),device=device)
                    if backend_dtype is not None and tensor.is_floating_point(): tensor=tensor.to(dtype=backend_dtype)
                    module_name, _, leaf=name.rpartition('.')
                    module=model.get_submodule(module_name) if module_name else model
                    if leaf in module._parameters:
                        old=module._parameters[leaf]
                        module._parameters[leaf]=torch.nn.Parameter(tensor,requires_grad=old.requires_grad)
                    elif leaf in module._buffers: module._buffers[leaf]=tensor
                    else: xksv.fail(f'backend tensor is not a parameter or buffer: {name}')
                    supplied.add(name)
            missing=expected-supplied
            for group in aliases.values():
                if group & supplied: missing-=group
            if missing: xksv.fail(f'archive is missing backend weights: {sorted(missing)[:10]}')
            model.tie_weights()
            # Move initialized non-persistent buffers and any tied tensors to device.
            model.to(device)
            tokenizer=AutoTokenizer.from_pretrained(assets.name,local_files_only=True,trust_remote_code=False)
            return model,tokenizer,assets
    except BaseException:
        assets.cleanup()
        raise


class ArchiveWeights(Mapping):
    def __init__(self,path,tensors,aliases,match_dtype):
        self.path=path; self.tensors=tensors; self.aliases=aliases; self.match_dtype=match_dtype
        self.native=xksv.Native()
    def __iter__(self): return iter(set(self.tensors)|set(self.aliases))
    def __len__(self): return len(set(self.tensors)|set(self.aliases))
    def __getitem__(self,name):
        import torch
        name=self.aliases.get(name,name)
        t=self.tensors[name]; raw=bytearray(t['size']); offset=0
        with open(self.path,'rb') as f:
            for b in t['blocks']:
                f.seek(b['offset']); data=xksv.read_exact(f,b['size'])
                if b['codec']=='bf16-huffman-tile16': data=self.native.call('xksv_unpack',data,b['rows'],b['cols'])
                elif b['codec']=='bf16-huffman-parallel32':
                    if len(data)<24 or struct.unpack_from('<QQ',data,8)!=(b['rows'],b['cols']): xksv.fail('parallel block dimensions disagree with index')
                    data=self.native.unpack_parallel(data)
                if len(data)!=b['raw_size'] or xksv.sha(data)!=b['sha256']: xksv.fail(f"streamed block checksum failed: {name}")
                raw[offset:offset+len(data)]=data; offset+=len(data)
        if offset!=len(raw) or xksv.sha(raw)!=t['sha256']: xksv.fail(f'streamed tensor checksum failed: {name}')
        dtype=DTYPE_MAP.get(t['dtype'])
        if dtype is None: xksv.fail(f'backend cannot load dtype {t["dtype"]}')
        value=torch.frombuffer(raw,dtype=getattr(torch,dtype)).reshape(t['shape']) if raw else torch.empty(t['shape'],dtype=getattr(torch,dtype))
        if self.match_dtype is not None and value.is_floating_point(): value=value.to(dtype=self.match_dtype)
        return value


class SourceWeights(Mapping):
    def __init__(self,tensors,aliases,match_dtype): self.tensors=tensors; self.aliases=aliases; self.match_dtype=match_dtype
    def __iter__(self): return iter(set(self.tensors)|set(self.aliases))
    def __len__(self): return len(set(self.tensors)|set(self.aliases))
    def __getitem__(self,name):
        import torch
        name=self.aliases.get(name,name); t=self.tensors[name]
        raw=bytearray(t['size'])
        with open(t['_source_path'],'rb') as f:
            f.seek(t['_source_base']+t['start'])
            view=memoryview(raw); offset=0
            while offset<len(view):
                count=f.readinto(view[offset:offset+xksv.RAW_CHUNK])
                if not count: xksv.fail(f'truncated source tensor: {name}')
                offset+=count
            view.release()
        dtype=DTYPE_MAP.get(t['dtype'])
        if dtype is None: xksv.fail(f'backend cannot load dtype {t["dtype"]}')
        value=torch.frombuffer(raw,dtype=getattr(torch,dtype)).reshape(t['shape']) if raw else torch.empty(t['shape'],dtype=getattr(torch,dtype))
        if self.match_dtype is not None and value.is_floating_point(): value=value.to(dtype=self.match_dtype)
        return value

def load_streaming(path,device,match_backend_dtype=False,fused_gemv=False,source_checkpoint=None):
    import torch
    from accelerate import init_empty_weights
    from accelerate.hooks import attach_align_device_hook
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoTokenizer
    if not device.startswith('cuda'): xksv.fail('streaming fallback currently requires CUDA')
    if source_checkpoint is None: xksv.verify(path,metadata_only=not fused_gemv)
    elif fused_gemv: xksv.fail('fused execution requires an archive')
    assets=tempfile.TemporaryDirectory(prefix='xksv-assets-')
    try:
        if source_checkpoint is None:
            with open(path,'rb') as f:
                index,_=xksv.load_index(f)
                for asset in index.get('assets',[]):
                    f.seek(asset['offset']); (Path(assets.name)/asset['name']).write_bytes(xksv.read_exact(f,asset['size']))
            model_dir=assets.name
        else:
            shards=xksv.checkpoint(source_checkpoint)
            index={'shards':[]}
            for shard in shards:
                entries=[dict(t,_source_path=shard['path'],_source_base=8+shard['header_size']) for t in shard['tensors']]
                index['shards'].append({'tensors':entries})
            source_path=Path(source_checkpoint)
            model_dir=str(source_path if source_path.is_dir() else source_path.parent)
        config=AutoConfig.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        if getattr(config,'model_type',None) not in ('llama','qwen3_5'): xksv.fail('streaming is only enabled for validated architectures')
        desired=getattr(getattr(config,'text_config',None),'dtype',None)
        if match_backend_dtype and (getattr(config,'model_type',None)!='qwen3_5' or desired not in (torch.float16,torch.bfloat16,torch.float32)):
            xksv.fail('unsupported streaming backend dtype matching')
        if not match_backend_dtype: desired=None
        with init_empty_weights(include_buffers=False):
            model=backend_class(config).from_config(config,trust_remote_code=False)
        model.tie_weights(); model.eval()
        if match_backend_dtype and (getattr(model,'_keep_in_fp32_modules',None) or getattr(model,'_keep_in_fp32_modules_strict',None)):
            xksv.fail('backend has unsupported FP32 preservation rules')
        expected=set(model.state_dict()); tensors={t['name']:t for shard in index['shards'] for t in shard['tensors']}
        extras=set(tensors)-expected
        invalid=[n for n in extras if not (getattr(config,'model_type',None)=='qwen3_5' and n.startswith('mtp.'))]
        if invalid: xksv.fail(f'backend does not accept tensors: {invalid[:5]}')
        aliases={}; missing=expected-set(tensors)
        groups={}
        for n,p in model.named_parameters(remove_duplicate=False): groups.setdefault(id(p),[]).append(n)
        for group in groups.values():
            present=[n for n in group if n in tensors]
            if present:
                for n in group:
                    if n in missing: aliases[n]=present[0]; missing.remove(n)
        if missing: xksv.fail(f'archive is missing backend weights: {sorted(missing)[:10]}')
        for name in expected:
            t=tensors[aliases.get(name,name)]; dtype=DTYPE_MAP.get(t['dtype'])
            if dtype is None: xksv.fail(f'backend cannot load dtype {t["dtype"]}')
            target=getattr(torch,dtype)
            if desired is not None and target in (torch.float16,torch.bfloat16,torch.float32,torch.float64): target=desired
            set_module_tensor_to_device(model,name,'meta',dtype=target)
        weights=(ArchiveWeights(path,tensors,aliases,desired) if source_checkpoint is None else SourceWeights(tensors,aliases,desired))
        if fused_gemv and getattr(config,'model_type',None)!='llama': xksv.fail('experimental fused GEMV is only implemented for Llama')
        attach_align_device_hook(model,execution_device=torch.device(device),offload=True,weights_map=weights,offload_buffers=True,preload_module_classes=['Qwen3_5GatedDeltaNet'] if getattr(config,'model_type',None)=='qwen3_5' else None)
        if fused_gemv:
            count=install_fused_gemv(model,path,tensors,weights)
            if not count: xksv.fail('no compatible linear layers for fused GEMV')
            print(f'experimental fused GEMV enabled for {count} linear layers; numerical results may differ',file=sys.stderr)
        tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        return model,tokenizer,assets
    except BaseException:
        assets.cleanup()
        raise

def install_fused_gemv(model,path,tensors,weights):
    import torch
    import torch.nn.functional as F
    lib=ctypes.CDLL(os.environ.get('XKSV_CUDA_LIBRARY',str(Path(__file__).resolve().parent.parent/'build'/'libxksv_cuda.so')))
    lib.xksv_cuda_gemv.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
    lib.xksv_cuda_gemv.restype=ctypes.c_int
    lib.xksv_cuda_gemv_parallel.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
    lib.xksv_cuda_gemv_parallel.restype=ctypes.c_int
    installed=0
    for name,module in model.named_modules():
        if not isinstance(module,torch.nn.Linear) or module.bias is not None or name=='lm_head': continue
        weight_name=name+'.weight'; t=tensors.get(weight_name)
        if not t or t['dtype']!='BF16' or len(t['shape'])!=2: continue
        rows,cols=t['shape']
        if not cols or not any(b['codec'] in ('bf16-huffman-tile16','bf16-huffman-parallel32') for b in t['blocks']): continue
        if any(b['raw_size']%(cols*2) or b['codec'] not in ('raw','bf16-huffman-tile16','bf16-huffman-parallel32') or (b['codec']!='raw' and b['cols']!=cols) for b in t['blocks']): continue
        def fused_forward(x,t=t,weight_name=weight_name,rows=rows,cols=cols):
            if x.numel()!=cols:
                return F.linear(x,weights[weight_name].to(x.device))
            vector=x.reshape(-1).float().contiguous()
            result=torch.empty(rows,device=x.device,dtype=torch.float32); offset=0
            with open(path,'rb') as f:
                for block in t['blocks']:
                    f.seek(block['offset']); data=xksv.read_exact(f,block['size'])
                    block_rows=block['raw_size']//(cols*2)
                    if block['codec'] in ('bf16-huffman-tile16','bf16-huffman-parallel32'):
                        buf=ctypes.create_string_buffer(data)
                        if block['codec']=='bf16-huffman-parallel32':
                            if len(data)<24 or struct.unpack_from('<QQ',data,8)!=(block_rows,cols): xksv.fail('parallel block dimensions disagree with index')
                            error=lib.xksv_cuda_gemv_parallel(buf,len(data),vector.data_ptr(),result.data_ptr()+offset*4)
                        else: error=lib.xksv_cuda_gemv(buf,len(data),block_rows,cols,vector.data_ptr(),result.data_ptr()+offset*4)
                        if error: xksv.fail(f'fused GEMV failed: {weight_name}')
                    else:
                        raw=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(block_rows,cols).to(x.device).float()
                        result[offset:offset+block_rows]=raw@vector
                    offset+=block_rows
            return result.to(x.dtype).reshape(*x.shape[:-1],rows)
        module.forward=fused_forward
        installed+=1
    return installed

def bit_equal(a,b):
    import torch
    return a.dtype==b.dtype and a.shape==b.shape and bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))


def max_difference(a,b):
    import torch
    return float((a.to(torch.float64)-b.to(torch.float64)).abs().max())

def layer_hooks(model, outputs):
    import torch
    handles=[]
    for name,module in model.named_modules():
        if not re.search(r'(?:^|\.)layers\.\d+$',name): continue
        def collect(_module,_args,result,name=name):
            value=result[0] if isinstance(result,(tuple,list)) else result
            if isinstance(value,torch.Tensor): outputs[name]=value.detach().cpu()
        handles.append(module.register_forward_hook(collect))
    return handles


def run(path,prompt,max_tokens,device,compare=None,gpu_decode=False,match_backend_dtype=False,compare_layers=False,stream=False,compare_step=False,fused_gemv=False,compare_source_streaming=False):
    import torch
    from transformers import AutoConfig
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available(): xksv.fail('CUDA unavailable')
    if device.startswith('cuda') and not stream:
        with open(path,'rb') as f:
            index,_=xksv.load_index(f)
        source_bytes=sum(t['size'] for shard in index['shards'] for t in shard['tensors'])
        free,_=torch.cuda.mem_get_info(torch.device(device))
        if source_bytes+max(1<<30,source_bytes//10)>free:
            if gpu_decode: xksv.fail('GPU decoder cannot be combined with automatic streaming')
            print('selecting module-level streaming because reconstructed weights leave insufficient GPU headroom',file=sys.stderr)
            stream=True
    begin=time.perf_counter()
    model,tokenizer,assets=(load_streaming(path,device,match_backend_dtype,fused_gemv) if stream else load_archive(path,device,gpu_decode,match_backend_dtype))
    synchronize(torch,device); load_seconds=time.perf_counter()-begin
    try:
        inputs=tokenizer(prompt,return_tensors='pt').to(device)
        compiled_layers={}
        handles=layer_hooks(model,compiled_layers) if compare_layers else []
        with torch.inference_mode():
            synchronize(torch,device); start=time.perf_counter()
            prefill_output=model(**inputs,use_cache=True)
            logits=prefill_output.logits[:,-1,:].detach().cpu()
            synchronize(torch,device); prefill=time.perf_counter()-start
            for handle in handles: handle.remove()
            step_logits=None
            if compare_step:
                token=prefill_output.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                step_logits=model(input_ids=token,past_key_values=prefill_output.past_key_values,use_cache=True).logits[:,-1,:].detach().cpu()
            synchronize(torch,device); start=time.perf_counter()
            result=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            synchronize(torch,device); duration=time.perf_counter()-start
        print(tokenizer.decode(result[0],skip_special_tokens=True))
        stats={'device':device,'streaming':stream,'experimental_fused_gemv':fused_gemv,'execution_weight_dtype':('transformers-auto' if match_backend_dtype else 'source'),'load_seconds':load_seconds,'prefill_seconds':prefill,'input_tokens':inputs['input_ids'].shape[-1], 'generation_seconds_including_prefill':duration,'generated_tokens':result.shape[-1]-inputs['input_ids'].shape[-1]}
        stats['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if device.startswith('cuda'): stats['peak_vram_bytes']=torch.cuda.max_memory_allocated()
        if compare:
            del model, prefill_output
            gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            source_assets=None
            reference_begin=time.perf_counter()
            if compare_source_streaming:
                baseline,_,source_assets=load_streaming(None,device,match_backend_dtype,source_checkpoint=compare)
            else:
                config=AutoConfig.from_pretrained(compare,local_files_only=True,trust_remote_code=False)
                baseline=backend_class(config).from_pretrained(compare,local_files_only=True,trust_remote_code=False,dtype='auto').to(device).eval()
            synchronize(torch,device)
            stats['reference_load_seconds']=time.perf_counter()-reference_begin
            reference_layers={}
            handles=layer_hooks(baseline,reference_layers) if compare_layers else []
            with torch.inference_mode():
                synchronize(torch,device); reference_begin=time.perf_counter()
                reference_output=baseline(**inputs,use_cache=True)
                ref=reference_output.logits[:,-1,:].detach().cpu()
                synchronize(torch,device); stats['reference_prefill_seconds']=time.perf_counter()-reference_begin
                for handle in handles: handle.remove()
                if compare_step:
                    token=reference_output.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                    ref_step=baseline(input_ids=token,past_key_values=reference_output.past_key_values,use_cache=True).logits[:,-1,:].detach().cpu()
                    stats['decode_logits_bit_identical']=bit_equal(step_logits,ref_step)
                    stats['decode_max_abs_logit_difference']=max_difference(step_logits,ref_step)
                synchronize(torch,device); reference_begin=time.perf_counter()
                baseline_ids=baseline.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
                synchronize(torch,device); stats['reference_generation_seconds_including_prefill']=time.perf_counter()-reference_begin
            if device.startswith('cuda'): stats['reference_peak_vram_bytes']=torch.cuda.max_memory_allocated()
            stats['process_peak_host_rss_after_reference_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            stats['logits_bit_identical']=bit_equal(logits,ref)
            stats['max_abs_logit_difference']=max_difference(logits,ref)
            stats['generated_ids_equal']=bool(torch.equal(result,baseline_ids))
            stats['reference_loading']='source-streaming' if compare_source_streaming else 'transformers-from-pretrained'
            if source_assets: source_assets.cleanup()
            if compare_layers:
                if set(compiled_layers)!=set(reference_layers): xksv.fail('layer comparison hooks differ between models')
                stats['layers']=[dict(name=n,bit_identical=bit_equal(compiled_layers[n],reference_layers[n]),max_abs_difference=max_difference(compiled_layers[n],reference_layers[n])) for n in sorted(compiled_layers)]
        print(json.dumps(stats,indent=2))
    finally: assets.cleanup()
