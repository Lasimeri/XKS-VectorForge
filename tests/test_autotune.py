import json,os,random,struct,subprocess,sys,tempfile
from pathlib import Path
import torch
if not torch.cuda.is_available():
    print('CUDA unavailable: skipped'); raise SystemExit(0)
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); source=tmp/'model.safetensors'; archive=tmp/'model.xksv'
    rng=random.Random(123); words=[((127+i%3)<<7)|rng.randrange(128) for i in range(16*4096)]
    data=struct.pack('<%dH'%len(words),*words)
    header=json.dumps({'w':{'dtype':'BF16','shape':[16,4096],'data_offsets':[0,len(data)]}},separators=(',',':')).encode();header+=b' '*((-len(header))%8)
    source.write_bytes(struct.pack('<Q',len(header))+header+data)
    subprocess.run([sys.executable,str(root/'python'/'xksv.py'),'compile',str(source),'--output',str(archive),'--autotune'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=os.environ.copy())
    subprocess.run([sys.executable,str(root/'python'/'xksv.py'),'verify',str(archive),'--source',str(source)],check=True,stdout=subprocess.DEVNULL,env=os.environ.copy())
    with archive.open('rb') as f:
        _,offset=struct.unpack('<8sQ',f.read(16));f.seek(offset);index=json.load(f)
    assert index['codec_preference']=='autotune' and index['tuning']['choices']
print('GPU autotuning recorded a measured choice and verified source bytes')
