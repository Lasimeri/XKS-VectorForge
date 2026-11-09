"""Synchronized, cold-prompt baseline/compiled inference comparisons."""
import gc
import inspect
import json
import resource
import statistics
import time
from pathlib import Path


def run(archive, original, device, lengths, batches, generations, repeats, output, match_backend_dtype=False, stream=False, source_streaming=False):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    import runtime
    import xksv
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    def sync():
        if device.startswith('cuda'): torch.cuda.synchronize()
    def timed(fn):
        sync(); start=time.perf_counter(); value=fn(); sync(); return value,time.perf_counter()-start
    results=[]; references={}
    for mode in ('original','compiled'):
        if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        if mode=='original':
            if source_streaming:
                model,tokenizer,assets=runtime.load_streaming(None,device,match_backend_dtype,source_checkpoint=original)
            else:
                tokenizer=AutoTokenizer.from_pretrained(original,local_files_only=True,trust_remote_code=False)
                config=AutoConfig.from_pretrained(original,local_files_only=True,trust_remote_code=False)
                model=runtime.backend_class(config).from_pretrained(original,local_files_only=True,trust_remote_code=False,dtype='auto').to(device).eval()
                assets=None
        else: model,tokenizer,assets=(runtime.load_streaming(archive,device,match_backend_dtype) if stream else runtime.load_archive(archive,device,match_backend_dtype=match_backend_dtype))
        sync(); load=time.perf_counter()-start
        seed=tokenizer('The capital of France is',add_special_tokens=False)['input_ids']
        if not seed: raise ValueError('tokenizer returned no input ids')
        forward_opts={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
        with torch.inference_mode():
            # One uncached warmup, excluding all timed results.
            _,_=timed(lambda:model(input_ids=torch.tensor([seed],device=device),use_cache=True,**forward_opts))
            for batch in batches:
                for length in lengths:
                    ids=torch.tensor([(seed*((length+len(seed)-1)//len(seed)))[:length]]*batch,device=device)
                    warm,_=timed(lambda:model(input_ids=ids,use_cache=True,**forward_opts))
                    del warm
                    for generated in generations:
                        prefills=[]; decodes=[]; outputs=[]; final_logits=None
                        for _ in range(repeats):
                            result,prefill=timed(lambda:model(input_ids=ids,use_cache=True,**forward_opts))
                            final_logits=result.logits[:,-1,:].detach().cpu()
                            token=result.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                            cache=result.past_key_values; token_list=[token]
                            sync(); start=time.perf_counter()
                            for step in range(generated-1):
                                result=model(input_ids=token,past_key_values=cache,use_cache=True,**forward_opts)
                                cache=result.past_key_values; token=result.logits[:,-1,:].argmax(dim=-1,keepdim=True); token_list.append(token)
                            sync(); decode=time.perf_counter()-start
                            prefills.append(prefill); decodes.append(decode); outputs.append(torch.cat(token_list,dim=1).cpu()); del result,cache,token_list,token
                        key=(batch,length,generated)
                        entry={'mode':mode,'batch':batch,'prompt_tokens':length,'generated_tokens':generated,'repeats':repeats,
                               'prefill_seconds':prefills,'decode_seconds':decodes,
                               'prefill_tokens_per_second_median':batch*length/statistics.median(prefills),
                               'decode_tokens_per_second_median':(batch*(generated-1)/statistics.median(decodes) if generated>1 else None),
                               'model_load_seconds':load,
                               'process_peak_host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                               'peak_vram_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None}
                        if mode=='original': references[key]=(final_logits,outputs[-1])
                        else:
                            ref,seq=references[key]
                            entry['last_prefill_logits_bit_identical']=runtime.bit_equal(ref,final_logits)
                            entry['max_abs_logit_difference']=runtime.max_difference(ref,final_logits)
                            entry['generated_ids_equal']=bool(torch.equal(seq,outputs[-1]))
                        results.append(entry); print(json.dumps(entry),flush=True)
        del model; gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()
        if assets: assets.cleanup()
    report={'device':device,'streaming':stream,'source_streaming':source_streaming,'execution_weight_dtype':('transformers-auto' if match_backend_dtype else 'source'),'artifact_bytes':Path(archive).stat().st_size,'original_safetensors_bytes':sum(Path(shard['path']).stat().st_size for shard in xksv.checkpoint(original)),'warmup_runs':'one initial and one per prompt shape','synchronization':'cuda synchronize before and after timed region','results':results}
    if output: Path(output).write_text(json.dumps(report,indent=2)+'\n')
