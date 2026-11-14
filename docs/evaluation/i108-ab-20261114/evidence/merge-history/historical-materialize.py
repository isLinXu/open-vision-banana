"""Average effective full Transformer parameters in FP32; cast once to BF16."""
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parent
SHARED=Path('/data/yrl/ovb-benchmark-e2b-20261028')
NYU=Path('/data/yrl/ovb-nyu-all-20261105')
OLD=Path('/data/yrl/ovb-i97-external-20261104')
PAIRS=[('Fusion-CK128-M0','Fusion-C1-KQ0r128','M0'),('Fusion-CK128-M1','Fusion-C1-KQ0r128','M1'),('Fusion-C1-KQ0r128-60-40','C1_retest','K-Q0-r128'),('Fusion-C1-KQ0r128-75-25','C1_retest','K-Q0-r128'),('Fusion-C1-KQ0r128-90-10','C1_retest','K-Q0-r128')]
WEIGHTS={PAIRS[0][0]:0.5,PAIRS[1][0]:0.5,PAIRS[2][0]:0.6,PAIRS[3][0]:0.75,PAIRS[4][0]:0.9}
torch.set_num_threads(16)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    write(ROOT/'fusion.status.json',dict(status='validating_inputs',updated_at=now()))
    parents={n:json.loads((NYU/'models'/n/'plan.json').read_text()) for n in ['Q0','K-Q0-r128','M0','M1']}
    c=next(r for r in json.loads((OLD/'checkpoints/checkpoint-integrity.json').read_text()) if r['model']=='C1_retest')
    parents['C1_retest']=dict(model='C1_retest',repo=c['repo'],revision=c['revision'],step=10000,base=str(SHARED/'base_model'),lora=str(OLD/'checkpoints/C1_retest/ckpt/checkpoint-10000_final'),files=[dict(remote_path=f['path'],size=f['size'],sha256=f['sha256']) for f in c['files']])
    original_models=json.loads(Path('/data/yrl/ovb-fusion-20261106/run-manifest.json').read_text())['models']
    parents['Fusion-C1-KQ0r128']=next(m for m in original_models if m['model']=='Fusion-C1-KQ0r128')
    evidence={};verified=set()
    sources=json.loads((OLD/'provenance/base-integrity.json').read_text())
    sources+=[dict(path=f['remote_path'],size=f['size'],sha256=f['sha256']) for f in json.loads((NYU/'models/E-merged-step0/plan.json').read_text())['files']]
    for p in parents.values():sources+=[dict(path=f['remote_path'],size=f['size'],sha256=f['sha256']) for f in p['files']]
    for item in sources:
        path=Path(item['path']).resolve()
        if path in verified:continue
        assert path.stat().st_size==item['size'] and digest(path)==item['sha256'],str(path)
        verified.add(path);evidence[str(path)]=item['sha256']
    write(ROOT/'provenance/input-integrity.json',evidence)
    indexes={};configs={}
    for name,p in parents.items():
        base=Path(p['base']);indexes[name]=json.loads((base/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
        cfg=json.loads((base/'transformer/config.json').read_text());cfg={k:v for k,v in cfg.items() if not k.startswith('_')}
        configs[name]=cfg
    keys=sorted(indexes['Q0'])
    assert all(sorted(i)==keys for i in indexes.values())
    assert all(c==configs['Q0'] for c in configs.values())
    recipes={};buffers={name:{} for name,a,b in PAIRS};sizes={name:0 for name,a,b in PAIRS};shards={name:[] for name,a,b in PAIRS};maps={name:{} for name,a,b in PAIRS};total_bytes=0
    for name,a,b in PAIRS:
        recipes[name]=dict(operation='FP32 weighted sum of full effective Transformer weights; one final BF16 cast; saved fusion parent loaded exactly',coefficients=[WEIGHTS[name],1-WEIGHTS[name]],parents=[parents[a],parents[b]],additional_optimizer_steps=0,frozen_components=str(SHARED/'base_model'),input_hashes=evidence)
    def flush(name):
        if not buffers[name]:return
        folder=ROOT/'models'/name/'transformer';folder.mkdir(parents=True,exist_ok=True)
        filename=f'diffusion_pytorch_model-{len(shards[name])+1:05d}.safetensors';p=folder/filename
        save_file(buffers[name],str(p)+'.tmp',metadata={'format':'pt'});Path(str(p)+'.tmp').replace(p)
        for key in buffers[name]:maps[name][key]=filename
        shards[name].append(p);buffers[name]={};sizes[name]=0
    with contextlib.ExitStack() as stack:
        handles={};adapters={};targets={};scales={};used={n:set() for n in parents}
        for name,p in parents.items():
            targets[name]={}
            if p.get('lora') is None:continue
            folder=Path(p['lora'])/'lora_adapter';cfg=json.loads((folder/'adapter_config.json').read_text())
            assert not cfg.get('use_dora') and not cfg.get('fan_in_fan_out') and not cfg.get('rank_pattern') and not cfg.get('alpha_pattern') and cfg.get('bias')=='none' and not cfg.get('modules_to_save')
            scales[name]=cfg['lora_alpha']/(math.sqrt(cfg['r']) if cfg.get('use_rslora') else cfg['r'])
            adapters[name]=stack.enter_context(safe_open(folder/'adapter_model.safetensors',framework='pt',device='cpu'))
            targets[name]={}
            for key in adapters[name].keys():
                if key.endswith('.lora_A.weight'):
                    target=key.removeprefix('base_model.model.').removesuffix('.lora_A.weight')+'.weight'
                    assert target in keys;targets[name][target]=(key,key.replace('.lora_A.weight','.lora_B.weight'))
            assert len(list(adapters[name].keys()))==2*len(targets[name])==448
            # Check the closed-form delta against PEFT's own implementation on one real layer.
            from peft.tuners.lora.layer import Linear
            from peft import LoraConfig
            ak,bk=next(iter(targets[name].values()));a=adapters[name].get_tensor(ak).float();b=adapters[name].get_tensor(bk).float()
            layer=Linear(torch.nn.Linear(a.shape[1],b.shape[0],bias=False),adapter_name='default',config=LoraConfig(r=cfg['r'],lora_alpha=cfg['lora_alpha'],use_rslora=cfg.get('use_rslora',False)),r=cfg['r'],lora_alpha=cfg['lora_alpha'])
            with torch.no_grad():layer.lora_A['default'].weight.copy_(a);layer.lora_B['default'].weight.copy_(b)
            assert torch.equal(layer.get_delta_weight('default'),(b@a)*scales[name]),name
            del layer,a,b
        write(ROOT/'provenance/delta-verification.json',dict(status='passed',parents=list(parents),method='Exact FP32 equality with PEFT get_delta_weight on a real layer for each parent'))
        for i,key in enumerate(keys):
            effective={}
            for name,p in parents.items():
                path=Path(p['base'])/'transformer'/indexes[name][key];canonical=str(path.resolve())
                if canonical not in handles:handles[canonical]=stack.enter_context(safe_open(path,framework='pt',device='cpu'))
                raw=handles[canonical].get_tensor(key);assert raw.is_floating_point(),key
                tensor=raw.float()
                if key in targets[name]:
                    ak,bk=targets[name][key];a=adapters[name].get_tensor(ak).float();b=adapters[name].get_tensor(bk).float()
                    tensor=tensor+(b@a)*scales[name];used[name].add(key)
                assert torch.isfinite(tensor).all(),(name,key)
                effective[name]=tensor
            for name,a,b in PAIRS:
                assert effective[a].shape==effective[b].shape
                result=(effective[a]*WEIGHTS[name]+effective[b]*(1-WEIGHTS[name])).to(torch.bfloat16).contiguous();n=result.numel()*result.element_size()
                if sizes[name]+n>4_500_000_000:flush(name)
                buffers[name][key]=result;sizes[name]+=n
            total_bytes+=n
            if i%50==0:write(ROOT/'fusion.status.json',dict(status='materializing',tensors_done=i+1,tensors_total=len(keys),updated_at=now()))
        assert all(used[n]==set(targets[n]) for n in parents)
    model_plans=[]
    for name,a,b in PAIRS:
        flush(name);folder=ROOT/'models'/name
        shutil.copy2(SHARED/'base_model/transformer/config.json',folder/'transformer/config.json')
        write(folder/'transformer/diffusion_pytorch_model.safetensors.index.json',dict(metadata={'total_size':total_bytes},weight_map=maps[name]))
        for source in (SHARED/'base_model').iterdir():
            if source.name in ['transformer','.cache','.git']:continue
            target=folder/source.name
            if not target.exists():target.symlink_to(source,target_is_directory=source.is_dir())
        signature=hashlib.sha256(json.dumps(recipes[name],sort_keys=True).encode()).hexdigest();write(folder/'fusion-recipe.json',recipes[name])
        files=[]
        for p in [*shards[name],folder/'transformer/config.json',folder/'transformer/diffusion_pytorch_model.safetensors.index.json',folder/'fusion-recipe.json']:
            files.append(dict(path=str(p.relative_to(folder)),relative=str(p.relative_to(folder)),remote_path=str(p),size=p.stat().st_size,sha256=digest(p)))
        # Roundtrip read validates every saved shard and complete tensor coverage.
        actual=set()
        for p in shards[name]:
            with safe_open(p,framework='pt',device='cpu') as f:actual.update(f.keys())
        assert actual==set(keys)
        model_plans.append(dict(model=name,repo='local-weight-fusion/'+name,revision=signature,step=0,weight_type='fp32_full_weight_weighted_average',expected_adapter_sha256=None,files=files,base=str(folder),lora=None,initialization='fusion',fusion_recipe=recipes[name]))
    plan=json.loads((ROOT/'manifest-template.json').read_text());plan['models']=model_plans+original_models;write(ROOT/'run-manifest.json',plan)
    write(ROOT/'model-paths.json',{m['model']:{k:m[k] for k in ['base','lora','step']} for m in plan['models']})
    write(ROOT/'fusion.status.json',dict(status='completed',models=[m['model'] for m in model_plans],coefficients={n:[WEIGHTS[n],1-WEIGHTS[n]] for n,a,b in PAIRS},dtype='bfloat16',arithmetic='float32',updated_at=now()))
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(ROOT/'fusion.status.json',dict(status='failed',error=traceback.format_exc(),updated_at=now()));raise
