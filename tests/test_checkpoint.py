import json, os, random, struct, subprocess, sys, tempfile
from pathlib import Path

script = Path(__file__).resolve().parents[1] / 'python' / 'xksv.py'
def run(*args, ok=True):
    p = subprocess.run([sys.executable,str(script),*map(str,args)],capture_output=True,text=True)
    if (p.returncode == 0) != ok: raise AssertionError((args,p.returncode,p.stdout,p.stderr))
    return p
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); model=tmp/'model'; model.mkdir()
    rng=random.Random(42)
    words=[((127+(i%3))<<7)|rng.randrange(128) for i in range(33*4096)]
    bf=struct.pack('<%dH'%len(words),*words)
    odd=bytes(range(32))
    header={'weights':{'dtype':'BF16','shape':[33,4096],'data_offsets':[0,len(bf)]},'other':{'dtype':'U8','shape':[len(odd)],'data_offsets':[len(bf),len(bf)+len(odd)]}}
    h=json.dumps(header,separators=(',',':')).encode(); h+=b' '*((-len(h))%8)
    source=model/'model.safetensors'; source.write_bytes(struct.pack('<Q',len(h))+h+bf+odd)
    artifact=tmp/'model.xksv'
    assert 'weights' in run('inspect',model).stdout
    run('compile',model,'--output',artifact)
    assert '2 tensors' in run('verify',artifact,'--source',model).stdout
    serial=tmp/'serial.xksv'; run('compile',model,'--output',serial,'--workers',1)
    assert serial.read_bytes()==artifact.read_bytes()
    parallel=tmp/'parallel.xksv'; run('compile',model,'--output',parallel,'--codec','dfloat11-parallel')
    run('verify',parallel,'--source',model)
    with parallel.open('rb') as f:
        _,offset=struct.unpack('<8sQ',f.read(16)); f.seek(offset); meta=json.load(f)
    assert any(block['codec']=='bf16-huffman-parallel32' for shard in meta['shards'] for t in shard['tensors'] for block in t['blocks'])
    resume_artifact=tmp/'resume.xksv'
    injection="""import sys
sys.path.insert(0,sys.argv[1])
import xksv
save=xksv._save_journal
count=0
def interrupted(path,state):
 global count
 save(path,state)
 count+=1
 if count==2: raise RuntimeError('simulated interruption')
xksv._save_journal=interrupted
xksv.compile_model(sys.argv[2],sys.argv[3])
"""
    stopped=subprocess.run([sys.executable,'-c',injection,str(script.parent),str(model),str(resume_artifact)],capture_output=True)
    assert stopped.returncode != 0
    assert Path(str(resume_artifact)+'.partial').exists() and Path(str(resume_artifact)+'.journal').exists()
    unchanged=source.read_bytes()
    modified=bytearray(unchanged); modified[-1]^=1; source.write_bytes(modified)
    run('compile',model,'--output',resume_artifact,'--resume',ok=False)
    source.write_bytes(unchanged)
    # Simulate uncommitted bytes from a crash between journal checkpoints.
    with open(str(resume_artifact)+'.partial','ab') as f: f.write(b'uncommitted')
    run('compile',model,'--output',resume_artifact,'--resume')
    run('verify',resume_artifact,'--source',model)
    assert not Path(str(resume_artifact)+'.journal').exists()
    data=bytearray(artifact.read_bytes()); data[20]^=1; bad=tmp/'bad.xksv'; bad.write_bytes(data)
    run('verify',bad,'--source',model,ok=False)
    data=bytearray(source.read_bytes()); data[-1]^=1; source.write_bytes(data)
    run('verify',artifact,'--source',model,ok=False)
    run('compile',model,'--output',artifact,ok=False)
print('checkpoint integration tests passed')
