#!/usr/bin/env python3
"""Streaming Safetensors compiler prototype; see README for execution limitations."""
import argparse
import array
import base64
import ctypes
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

MAGIC = b'XKSV\x00\x01\r\n'
HEADER = struct.Struct('<8sQ')
DTYPES = {'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E5M2':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}
MAX_HEADER = 100_000_000
MAX_INDEX = 1_000_000_000
RAW_CHUNK = 1 << 20


def fail(msg):
    raise ValueError(msg)


def no_duplicates(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            fail(f'duplicate JSON key: {k}')
        result[k] = v
    return result


def parse_json(data):
    return json.loads(data, object_pairs_hook=no_duplicates, parse_constant=lambda x: fail('invalid JSON number'))


def tensor_role(name):
    if 'embed_tokens' in name or 'word_embeddings' in name: return 'token_embedding'
    if 'lm_head' in name or 'output_projection' in name: return 'language_model_head'
    if 'visual' in name or 'vision' in name: return 'vision_component'
    if 'linear_attn' in name:
        if 'conv' in name: return 'deltanet_convolution'
        if 'in_proj' in name: return 'deltanet_input_projection'
        if 'out_proj' in name: return 'deltanet_output_projection'
        return 'deltanet_state_parameter'
    if 'self_attn' in name or '.attention.' in name: return 'attention_parameter'
    if 'expert' in name: return 'moe_expert'
    if 'router' in name: return 'moe_router'
    if '.mlp.' in name or '.feed_forward.' in name: return 'feed_forward_parameter'
    if 'norm' in name: return 'normalization_parameter'
    return 'other'


def contiguous_strides(shape):
    result=[]; stride=1
    for dim in reversed(shape): result.append(stride); stride*=dim
    return list(reversed(result))


def tensor_statistics(shard,t):
    if t['dtype']!='BF16': return None
    hist=[0]*256; bits=[0]*16
    with open(shard['path'],'rb') as f:
        f.seek(8+shard['header_size']+t['start']); remaining=t['size']
        while remaining:
            data=read_exact(f,min(remaining,RAW_CHUNK)); remaining-=len(data)
            words=array.array('H'); words.frombytes(data)
            if sys.byteorder!='little': words.byteswap()
            for word in words:
                hist[(word>>7)&255]+=1
                for bit in range(16): bits[bit]+=(word>>bit)&1
    count=t['size']//2
    entropy=-sum((v/count)*math.log2(v/count) for v in hist if v) if count else 0.0
    return dict(exponent_histogram=hist,bit_ones=bits,exponent_entropy_bits=entropy,ideal_payload_bytes=count*(8+entropy)/8)


def checkpoint(path):
    path = Path(path)
    index_path = path / 'model.safetensors.index.json' if path.is_dir() else None
    mapping = None
    if index_path and index_path.is_file():
        mapping = parse_json(index_path.read_bytes()).get('weight_map')
        if not isinstance(mapping, dict) or any(not isinstance(v,str) or Path(v).name != v or not v.endswith('.safetensors') for v in mapping.values()): fail('invalid shard index')
        files = [path / name for name in sorted(set(mapping.values()))]
    else:
        files = sorted(path.glob('*.safetensors')) if path.is_dir() else [path]
    if not files:
        fail('no Safetensors shards found')
    shards, names = [], set()
    for file in files:
        if not file.is_file() or file.suffix != '.safetensors':
            fail(f'not a Safetensors file: {file}')
        size = file.stat().st_size
        with file.open('rb') as f:
            prefix = f.read(8)
            if len(prefix) != 8: fail(f'truncated header: {file}')
            hlen = struct.unpack('<Q', prefix)[0]
            if hlen > MAX_HEADER or hlen > size-8: fail(f'invalid header length: {file}')
            header = f.read(hlen)
        j = parse_json(header)
        if not isinstance(j, dict): fail('invalid Safetensors header')
        tensors, ranges = [], []
        for name, desc in j.items():
            if name == '__metadata__': continue
            if name in names: fail(f'duplicate tensor: {name}')
            names.add(name)
            if not isinstance(desc, dict): fail(f'invalid descriptor: {name}')
            dtype, shape, offsets = desc.get('dtype'), desc.get('shape'), desc.get('data_offsets')
            if dtype not in DTYPES: fail(f'unsupported dtype {dtype} for {name}')
            if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape): fail(f'invalid shape: {name}')
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets): fail(f'invalid offsets: {name}')
            start, end = offsets
            num = 1
            for dim in shape: num *= dim
            if start < 0 or end < start or end > size-8-hlen or end-start != num*DTYPES[dtype]: fail(f'invalid tensor extent: {name}')
            tensors.append(dict(name=name, dtype=dtype, shape=shape, strides=contiguous_strides(shape), role=tensor_role(name), start=start, size=end-start))
            if end > start: ranges.append((start,end))
        ranges.sort()
        cursor = 0
        for start,end in ranges:
            if start != cursor: fail(f'overlapping or noncontiguous tensors in {file}')
            cursor = end
        if cursor != size-8-hlen: fail(f'unreferenced data in {file}')
        tensors.sort(key=lambda x:x['start'])
        shards.append(dict(name=file.name, path=str(file), header=base64.b64encode(header).decode('ascii'), header_size=hlen, tensors=tensors))
    if mapping is not None:
        actual = {t['name']:s['name'] for s in shards for t in s['tensors']}
        if mapping != actual: fail('shard index does not match discovered tensor names and files')
    return shards


class Native:
    def __init__(self):
        path = os.environ.get('XKSV_LIBRARY', str(Path(__file__).resolve().parent.parent / 'build' / 'libxksv_native.so'))
        self.lib = ctypes.CDLL(path)
        for name in ('xksv_pack','xksv_unpack'):
            f = getattr(self.lib, name)
            f.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_size_t)]
            f.restype = ctypes.c_int
        self.lib.xksv_pack_parallel.argtypes=self.lib.xksv_pack.argtypes
        self.lib.xksv_pack_parallel.restype=ctypes.c_int
        self.lib.xksv_unpack_parallel.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_size_t)]
        self.lib.xksv_unpack_parallel.restype=ctypes.c_int
        self.lib.xksv_free.argtypes = [ctypes.c_void_p]
    def unpack_parallel(self,data):
        buf=ctypes.create_string_buffer(data); out,size=ctypes.c_void_p(),ctypes.c_size_t()
        if self.lib.xksv_unpack_parallel(buf,len(data),ctypes.byref(out),ctypes.byref(size)): fail('parallel block decode failed')
        try: return ctypes.string_at(out,size.value)
        finally: self.lib.xksv_free(out)
    def call(self, op, data, rows, cols):
        buf = ctypes.create_string_buffer(data)
        out, size = ctypes.c_void_p(), ctypes.c_size_t()
        if getattr(self.lib, op)(buf, len(data), rows, cols, ctypes.byref(out), ctypes.byref(size)):
            fail(f'native {op} failed')
        try: return ctypes.string_at(out, size.value)
        finally: self.lib.xksv_free(out)


def read_exact(f,n):
    data = f.read(n)
    if len(data) != n: fail('unexpected end of file')
    return data


def sha(b): return hashlib.sha256(b).hexdigest()


def sha_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            data=f.read(RAW_CHUNK)
            if not data: return digest.hexdigest()
            digest.update(data)


def _save_journal(path, state):
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            f.write(json.dumps(state,separators=(',',':'),ensure_ascii=False))
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def compile_model(source, output, resume=False, workers=4, codec_preference='dfloat11'):
    if workers<1 or workers>32: fail('workers must be between 1 and 32')
    shards=checkpoint(source); output=Path(output).absolute()
    for s in shards:
        if Path(s['path']).resolve()==output.resolve(): fail('output would overwrite source')
    if output.exists(): fail('output already exists')
    partial=output.with_name(output.name+'.partial')
    journal=output.with_name(output.name+'.journal')
    native=Native()
    tuner=None
    if codec_preference=='autotune':
        import tune
        tuner=tune.GpuTuner(parse_json(journal.read_bytes())['index'].get('tuning') if resume and journal.is_file() else None)
    if resume:
        if not partial.is_file() or not journal.is_file(): fail('resume requires both partial file and journal')
        state=parse_json(journal.read_bytes()); index=state['index']; position=state['position']
        if tuner: index['tuning']=tuner.profile
        if index.get('codec_preference','dfloat11')!=codec_preference: fail('resume codec differs from journal')
        if type(position) is not int or position<HEADER.size or partial.stat().st_size<position: fail('invalid resume position')
        if len(index['shards'])!=len(shards): fail('source shard set changed')
        for saved,src in zip(index['shards'],shards):
            if saved['name']!=src['name'] or saved['header']!=src['header'] or saved.get('source_sha256')!=sha_file(src['path']): fail('source shard changed')
            if len(saved['tensors'])>len(src['tensors']): fail('invalid resume tensor count')
            with open(src['path'],'rb') as f:
                for old,t in zip(saved['tensors'],src['tensors']):
                    if any(old[k]!=t[k] for k in ('name','dtype','shape','start','size')): fail('source tensor descriptor changed')
                    f.seek(8+src['header_size']+t['start']); digest=hashlib.sha256(); remain=t['size']
                    while remain:
                        data=read_exact(f,min(remain,RAW_CHUNK)); digest.update(data); remain-=len(data)
                    if digest.hexdigest()!=old['sha256']: fail(f"source tensor changed: {t['name']}")
        for a in index.get('assets',[]):
            src=Path(source)/a['name']
            if not src.is_file() or sha(src.read_bytes())!=a['sha256']: fail('source assets changed')
        # Verify every committed block before trusting the journal. The temporary
        # index is removed even on a failed verification; a later resume repeats it.
        with open(partial,'r+b') as f:
            f.truncate(position); f.seek(position)
            partial_index=dict(index,partial=True)
            f.write(json.dumps(partial_index,separators=(',',':'),ensure_ascii=False).encode())
            f.seek(0); f.write(HEADER.pack(MAGIC,position)); f.flush(); os.fsync(f.fileno())
        try: verify(partial,allow_partial=True)
        finally:
            with open(partial,'r+b') as f:
                f.truncate(position); f.seek(0); f.write(HEADER.pack(MAGIC,0)); f.flush(); os.fsync(f.fileno())
    else:
        if partial.exists() or journal.exists(): fail('partial compilation exists; use --resume or choose another output')
        source_hashes=[sha_file(s['path']) for s in shards]
        fd=os.open(partial,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        with os.fdopen(fd,'w+b') as dst:
            dst.write(HEADER.pack(MAGIC,0))
            index=dict(version=1,format='safetensors',codec_preference=codec_preference,shards=[dict(name=s['name'],header=s['header'],source_sha256=digest,tensors=[]) for s,digest in zip(shards,source_hashes)],assets=[])
            if tuner: index['tuning']=tuner.profile
            if Path(source).is_dir():
                for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json','chat_template.jinja','tokenizer.model','vocab.txt','preprocessor_config.json','processor_config.json','video_preprocessor_config.json','LICENSE','NOTICE','README.md'):
                    path=Path(source)/name
                    if path.is_file():
                        if path.stat().st_size>100_000_000: fail(f'asset too large: {name}')
                        data=path.read_bytes()
                        index['assets'].append(dict(name=name,offset=dst.tell(),size=len(data),sha256=sha(data))); dst.write(data)
            dst.flush(); os.fsync(dst.fileno()); position=dst.tell()
        _save_journal(journal,dict(index=index,position=position))
    with ThreadPoolExecutor(max_workers=workers) as pool, open(partial,'r+b') as dst:
        dst.seek(position)
        for shard,sout in zip(shards,index['shards']):
            with open(shard['path'],'rb') as src:
                for tensor in shard['tensors'][len(sout['tensors']):]:
                    t=dict(tensor); t['blocks']=[]; digest=hashlib.sha256()
                    src.seek(8+shard['header_size']+tensor['start']); remaining=tensor['size']; shape=tensor['shape']
                    cols=(shape[1] if len(shape)==2 and shape[1] and shape[1]*2<=RAW_CHUNK else min(4096,remaining//2)) if tensor['dtype']=='BF16' else 0
                    pending=deque()
                    def write_block(item):
                        future,raw_size,rows,cols,checksum=item
                        codec,data=future.result()
                        if codec=='autotune': codec,data=tuner.select(data[0],data[1],data[2],rows,cols)
                        t['blocks'].append(dict(offset=dst.tell(),size=len(data),raw_size=raw_size,codec=codec,rows=rows if codec!='raw' else 0,cols=cols if codec!='raw' else 0,sha256=checksum))
                        dst.write(data)
                    def compress(raw,rows,cols):
                        if codec_preference=='autotune' and rows:
                            return 'autotune',(native.call('xksv_pack',raw,rows,cols),native.call('xksv_pack_parallel',raw,rows,cols),raw)
                        operation='xksv_pack_parallel' if codec_preference=='dfloat11-parallel' else 'xksv_pack'
                        encoded=native.call(operation,raw,rows,cols) if rows else raw
                        name='bf16-huffman-parallel32' if codec_preference=='dfloat11-parallel' else 'bf16-huffman-tile16'
                        return (name,encoded) if rows and len(encoded)<len(raw) else ('raw',raw)
                    while remaining:
                        rows=min(16,RAW_CHUNK//(cols*2),remaining//(cols*2)) if cols else 0
                        if cols and not rows: cols=min(4096,remaining//2); rows=1
                        length=rows*cols*2 if cols else min(remaining,RAW_CHUNK)
                        raw=read_exact(src,length); digest.update(raw); remaining-=length
                        pending.append((pool.submit(compress,raw,rows,cols),len(raw),rows,cols,sha(raw)))
                        if len(pending)>=workers*2: write_block(pending.popleft())
                    while pending: write_block(pending.popleft())
                    t['sha256']=digest.hexdigest(); sout['tensors'].append(t)
                    dst.flush(); os.fsync(dst.fileno()); position=dst.tell()
                    _save_journal(journal,dict(index=index,position=position))
                    print(f"compiled {tensor['name']} ({tensor['size']} bytes)",file=sys.stderr,flush=True)
        offset=dst.tell(); payload=json.dumps(index,separators=(',',':'),ensure_ascii=False).encode()
        dst.write(payload); dst.seek(0); dst.write(HEADER.pack(MAGIC,offset)); dst.flush(); os.fsync(dst.fileno())
    verify(partial,source)
    os.link(partial,output); os.unlink(partial); os.unlink(journal)
    dfd=os.open(output.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(dfd)
    finally: os.close(dfd)
    print(f'created {output}: artifact {output.stat().st_size} bytes')


def load_index(f):
    size = os.fstat(f.fileno()).st_size
    magic, offset = HEADER.unpack(read_exact(f,HEADER.size))
    if magic != MAGIC or offset < HEADER.size or offset > size or size-offset>MAX_INDEX: fail('invalid archive header')
    f.seek(offset); index = parse_json(read_exact(f,size-offset))
    if index.get('version') != 1 or index.get('format') != 'safetensors': fail('unsupported archive')
    return index, offset


def verify(archive, source=None, allow_partial=False, metadata_only=False):
    if metadata_only and source is not None: fail('metadata-only verification cannot compare source bytes')
    native = Native(); sources = checkpoint(source) if source else None
    source_map = {s['name']:s for s in sources} if sources is not None else {}
    tensor_count = byte_count = 0; seen_shards=set(); seen_names=set(); spans=[]
    with open(archive,'rb') as f:
        index, limit = load_index(f)
        asset_names=set()
        for asset in index.get('assets',[]):
            name,off,n=asset['name'],asset['offset'],asset['size']
            if not isinstance(name,str) or Path(name).name!=name or name in asset_names or type(off) is not int or type(n) is not int or off<HEADER.size or n<0 or n>100_000_000 or off>limit or n>limit-off: fail('invalid asset')
            asset_names.add(name); spans.append((off,off+n)); f.seek(off)
            if sha(read_exact(f,n))!=asset['sha256']: fail('asset checksum failed')
            if sources is not None:
                p=Path(source)/name
                if not p.is_file() or sha(p.read_bytes())!=asset['sha256']: fail(f'source asset differs: {name}')
        for shard in index['shards']:
            name=shard['name']
            if name in seen_shards or Path(name).name != name: fail('invalid archive shard name')
            seen_shards.add(name)
            header=parse_json(base64.b64decode(shard['header'],validate=True))
            if not isinstance(header,dict): fail('invalid stored shard header')
            expected={k:v for k,v in header.items() if k!='__metadata__'}
            stored_names=[t['name'] for t in shard['tensors']]
            if len(stored_names)!=len(set(stored_names)) or set(stored_names)!=set(expected):
                # Partial artifacts are used only during resume verification.
                if not allow_partial or not set(stored_names)<=set(expected): fail('archive tensor set differs from header')
            original=source_map.get(name)
            if sources is not None and not original: fail(f'missing source shard: {name}')
            if original and shard['header'] != original['header']: fail(f'header differs: {name}')
            if original and shard.get('source_sha256') and sha_file(original['path'])!=shard['source_sha256']: fail(f'source shard hash differs: {name}')
            src = open(original['path'],'rb') if original else None
            source_tensors = {t['name']:t for t in original['tensors']} if original else {}
            try:
                for t in shard['tensors']:
                    if t['name'] in seen_names: fail('duplicate archive tensor')
                    seen_names.add(t['name'])
                    desc=expected[t['name']]; shape=t['shape']; dtype=t['dtype']
                    if dtype not in DTYPES or not isinstance(shape,list) or any(type(x) is not int or x<0 for x in shape): fail('invalid tensor descriptor')
                    count=1
                    for dim in shape: count*=dim
                    if count*DTYPES[dtype]!=t['size'] or desc.get('dtype')!=dtype or desc.get('shape')!=shape or desc.get('data_offsets')!=[t['start'],t['start']+t['size']]: fail('tensor descriptor conflicts with stored header')
                    st=source_tensors.get(t['name'])
                    if original and (not st or any(t[k]!=st[k] for k in ('dtype','shape','start','size'))): fail(f'tensor descriptor differs: {t["name"]}')
                    if src: src.seek(8+original['header_size']+t['start'])
                    digest=hashlib.sha256(); count=0
                    for b in t['blocks']:
                        off,n=b['offset'],b['size']
                        if type(off) is not int or type(n) is not int or off<HEADER.size or n<0 or off>limit or n>limit-off or n>RAW_CHUNK or type(b['raw_size']) is not int or b['raw_size']<0 or b['raw_size']>RAW_CHUNK: fail('invalid block extent')
                        spans.append((off,off+n))
                        if b['codec']=='raw':
                            if n!=b['raw_size']: fail('invalid raw block length')
                        elif b['codec'] in ('bf16-huffman-tile16','bf16-huffman-parallel32'):
                            if type(b['rows']) is not int or type(b['cols']) is not int or b['rows']<=0 or b['cols']<=0 or b['rows']*b['cols']*2 != b['raw_size']: fail('invalid compressed block dimensions')
                        else: fail('unknown codec')
                        if metadata_only:
                            count+=b['raw_size']; continue
                        f.seek(off); encoded=read_exact(f,n)
                        if b['codec']=='bf16-huffman-parallel32' and (n<24 or struct.unpack_from('<QQ',encoded,8)!=(b['rows'],b['cols'])): fail('parallel block dimensions disagree with index')
                        data=encoded if b['codec']=='raw' else (native.unpack_parallel(encoded) if b['codec']=='bf16-huffman-parallel32' else native.call('xksv_unpack',encoded,b['rows'],b['cols']))
                        if len(data)!=b['raw_size'] or sha(data)!=b['sha256']: fail(f'block checksum failed: {t["name"]}')
                        if src and data!=read_exact(src,len(data)): fail(f'source bytes differ: {t["name"]}')
                        digest.update(data); count+=len(data)
                    if count!=t['size'] or (not metadata_only and digest.hexdigest()!=t['sha256']): fail(f'tensor verification failed: {t["name"]}')
                    tensor_count+=1; byte_count+=count
                if original and set(source_tensors)!={t['name'] for t in shard['tensors']}: fail('source tensor set differs')
            finally:
                if src: src.close()
        if sources is not None and seen_shards!=set(source_map): fail('source shard set differs')
        spans.sort(); cursor=HEADER.size
        for start,end in spans:
            if start!=cursor: fail('overlapping or unreferenced archive data')
            cursor=end
        if cursor!=limit: fail('unreferenced archive data')
    print((f'validated metadata for {tensor_count} tensors; data checks deferred until use' if metadata_only else f'verified {tensor_count} tensors, {byte_count} bytes' + (' against source' if source else ' against stored checksums')))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    i=sub.add_parser('inspect'); i.add_argument('source'); i.add_argument('--stats',action='store_true')
    c=sub.add_parser('compile'); c.add_argument('source'); c.add_argument('--output',required=True); c.add_argument('--codec',choices=['dfloat11','dfloat11-parallel'],default='dfloat11'); c.add_argument('--resume',action='store_true'); c.add_argument('--workers',type=int,default=4); c.add_argument('--autotune',action='store_true')
    v=sub.add_parser('verify'); v.add_argument('archive'); v.add_argument('--source')
    r=sub.add_parser('run'); r.add_argument('archive'); r.add_argument('--prompt',default='Once upon a time'); r.add_argument('--max-new-tokens',type=int,default=16); r.add_argument('--device',default='auto'); r.add_argument('--compare-original'); r.add_argument('--gpu-decode',action='store_true'); r.add_argument('--backend',choices=['auto','transformers'],default='auto'); r.add_argument('--match-backend-dtype',action='store_true'); r.add_argument('--compare-layers',action='store_true'); r.add_argument('--stream',action='store_true'); r.add_argument('--compare-step',action='store_true'); r.add_argument('--experimental-fused-gemv',action='store_true'); r.add_argument('--compare-source-streaming',action='store_true')
    b=sub.add_parser('bench'); b.add_argument('archive'); b.add_argument('--compare-original',required=True); b.add_argument('--device',default='auto'); b.add_argument('--prefill',default='512,2048,8192'); b.add_argument('--batches',default='1,4,8'); b.add_argument('--generate',default='1,128,512'); b.add_argument('--repeats',type=int,default=3); b.add_argument('--output'); b.add_argument('--match-backend-dtype',action='store_true'); b.add_argument('--stream',action='store_true'); b.add_argument('--source-streaming',action='store_true')
    a=p.parse_args()
    if a.cmd=='inspect':
        for s in checkpoint(a.source):
            for t in s['tensors']:
                result=dict(shard=s['name'],**t)
                if a.stats: result['statistics']=tensor_statistics(s,t)
                print(json.dumps(result,ensure_ascii=False))
    elif a.cmd=='compile':
        if a.autotune and a.codec!='dfloat11': fail('--autotune selects between codecs; use the default --codec')
        compile_model(a.source,a.output,a.resume,a.workers,'autotune' if a.autotune else a.codec)
    elif a.cmd=='verify': verify(a.archive,a.source)
    elif a.cmd=='bench':
        import benchmark
        parse=lambda s:[int(x) for x in s.split(',')]
        lengths,batches,generations=parse(a.prefill),parse(a.batches),parse(a.generate)
        if a.repeats<1 or any(x<1 for x in lengths+batches+generations): fail('benchmark dimensions must be positive')
        benchmark.run(a.archive,a.compare_original,a.device,lengths,batches,generations,a.repeats,a.output,a.match_backend_dtype,a.stream,a.source_streaming)
    else:
        if a.max_new_tokens < 1: fail('max-new-tokens must be positive')
        if (a.compare_layers or a.compare_step) and not a.compare_original: fail('layer and decode comparisons require --compare-original')
        if a.stream and a.gpu_decode: fail('--gpu-decode is not supported with --stream')
        if a.experimental_fused_gemv and not a.stream: fail('experimental fused GEMV requires --stream')
        if a.compare_source_streaming and not a.compare_original: fail('source streaming comparison requires --compare-original')
        import runtime
        runtime.run(a.archive,a.prompt,a.max_new_tokens,a.device,a.compare_original,a.gpu_decode,a.match_backend_dtype,a.compare_layers,a.stream,a.compare_step,a.experimental_fused_gemv,a.compare_source_streaming)

if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError,TypeError,OverflowError,struct.error) as e:
        print(f'xksv: {e}',file=sys.stderr); sys.exit(1)
