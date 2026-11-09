import json, os, subprocess, sys, tempfile
from pathlib import Path
import torch
if not torch.cuda.is_available():
    print('CUDA unavailable: skipped')
    raise SystemExit(0)
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, AutoModelForCausalLM
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'python'))
import runtime
import xksv
assert not runtime.bit_equal(torch.tensor([0.0]),torch.tensor([-0.0]))
assert not runtime.bit_equal(torch.tensor([1.0],dtype=torch.float16),torch.tensor([1.0],dtype=torch.float32))
with tempfile.TemporaryDirectory() as tmp:
    source=Path(tmp)/'source'; archive=Path(tmp)/'model.xksv'; source.mkdir()
    torch.manual_seed(712)
    c=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
    LlamaForCausalLM(c).to(torch.bfloat16).save_pretrained(source,safe_serialization=True)
    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'hello':1,'world':2},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]').save_pretrained(source)
    subprocess.run([sys.executable,str(root/'python'/'xksv.py'),'compile',str(source),'--output',str(archive)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=os.environ.copy())
    with archive.open('rb') as f:
        index,_=xksv.load_index(f)
    item=next(t for shard in index['shards'] for t in shard['tensors'] if t['name'].endswith('embed_tokens.weight'))
    damaged=bytearray(archive.read_bytes()); damaged[item['blocks'][0]['offset']]^=1
    corrupt=Path(tmp)/'corrupt.xksv'; corrupt.write_bytes(damaged)
    xksv.verify(corrupt,metadata_only=True)
    try:
        runtime.ArchiveWeights(corrupt,{item['name']:item},{},None)[item['name']]
    except ValueError: pass
    else: raise AssertionError('corrupt streamed tensor was accepted')
    streamed,_,assets=runtime.load_streaming(archive,'cuda')
    baseline=AutoModelForCausalLM.from_pretrained(source,dtype='auto',local_files_only=True).to('cuda').eval()
    inputs=torch.tensor([[1,2,1]],device='cuda')
    with torch.inference_mode():
        a=streamed(input_ids=inputs,use_cache=True); b=baseline(input_ids=inputs,use_cache=True)
        assert runtime.bit_equal(a.logits,b.logits)
        next_id=a.logits[:,-1,:].argmax(-1,keepdim=True)
        a=streamed(input_ids=next_id,past_key_values=a.past_key_values,use_cache=True)
        b=baseline(input_ids=next_id,past_key_values=b.past_key_values,use_cache=True)
        assert runtime.bit_equal(a.logits,b.logits)
    assets.cleanup()
    source_stream,_,source_assets=runtime.load_streaming(None,'cuda',source_checkpoint=source)
    with torch.inference_mode():
        a=source_stream(input_ids=inputs,use_cache=True); b=baseline(input_ids=inputs,use_cache=True)
        assert runtime.bit_equal(a.logits,b.logits)
        next_id=a.logits[:,-1,:].argmax(-1,keepdim=True)
        a=source_stream(input_ids=next_id,past_key_values=a.past_key_values,use_cache=True)
        b=baseline(input_ids=next_id,past_key_values=b.past_key_values,use_cache=True)
        assert runtime.bit_equal(a.logits,b.logits)
    source_assets.cleanup()
    fused,_,fused_assets=runtime.load_streaming(archive,'cuda',fused_gemv=True)
    with torch.inference_mode():
        prefill=fused(input_ids=inputs,use_cache=True)
        next_id=prefill.logits[:,-1,:].argmax(-1,keepdim=True)
        next_fused=fused(input_ids=next_id,past_key_values=prefill.past_key_values,use_cache=True)
        expected=baseline(input_ids=inputs,use_cache=True)
        next_reference=baseline(input_ids=next_id,past_key_values=expected.past_key_values,use_cache=True)
        assert torch.allclose(next_fused.logits.float(),next_reference.logits.float(),atol=0.03,rtol=0.03)
    fused_assets.cleanup()
    report=Path(tmp)/'bench.json'
    subprocess.run([sys.executable,str(root/'python'/'xksv.py'),'bench',str(archive),'--compare-original',str(source),'--device','cuda','--stream','--source-streaming','--prefill','3','--batches','1','--generate','1','--repeats','1','--output',str(report)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=os.environ.copy())
    results=json.loads(report.read_text())['results']
    assert results[-1]['last_prefill_logits_bit_identical'] and results[-1]['generated_ids_equal']
    run=subprocess.run([sys.executable,str(root/'python'/'xksv.py'),'run',str(archive),'--device','cuda','--stream','--prompt','hello world','--max-new-tokens','1','--compare-original',str(source),'--compare-source-streaming','--compare-step'],check=True,capture_output=True,text=True,env=os.environ.copy())
    report=json.loads(run.stdout[run.stdout.index('{\n'):])
    assert report['logits_bit_identical'] and report['decode_logits_bit_identical'] and report['reference_loading']=='source-streaming'
    assert report['reference_prefill_seconds']>0 and report['reference_generation_seconds_including_prefill']>0
print('streamed model, source benchmark, and experimental fused decode tests passed')
