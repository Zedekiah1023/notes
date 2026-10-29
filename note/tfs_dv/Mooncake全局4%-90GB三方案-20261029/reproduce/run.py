"""在统一K=1、90GB/s原生进程栈中回放全局4%两种负载。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
W=next(p for p in HERE.parents if (p/'workspace.json').is_file())
P=W/'data/dynamo_vllm_simulation/dynamo_vllm_cpu_sim'
BASE=P/'results/mooncake360_prefill_v2_four_20261028'
EXP=P/'results/mooncake360_global4_90g_20261029'
ALLOC=W/'temp/tfs_opt_global4_20261029'
OVERLAY=W/'temp/weighted-block16-mac-20261027/python'
ARMS={'direct':1.,'tfs-a01':.1,'tfs-a1':1.}
SEED=20261029
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def host():
    import psutil
    return dict(epoch_s=time.time(),available_bytes=psutil.virtual_memory().available,
        swap_bytes=psutil.swap_memory().used,
        power=subprocess.check_output(['pmset','-g','batt'],text=True).strip())


def prepare():
    assert not EXP.exists(),'Never overwrite an experiment'
    EXP.mkdir()
    original=read(BASE/'input_check.json')
    assert sha(OVERLAY/'dynamo/_core.abi3.so')==original['native_sha256']
    for dataset in ('ABC','ABCD'):
        A=EXP/dataset;root=A/'package';root.mkdir(parents=True)
        for name in ('sim','scripts','configs','patched'):
            shutil.copytree(P/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('versions.json','source-fingerprints.json'):
            shutil.copy2(P/name,root/name)
        for name in ('.venv','upstream'):
            (root/name).symlink_to(P/name,target_is_directory=True)
        adapter=root/'results/concurrency_original_step5_20261024'
        adapter.mkdir(parents=True)
        for name in ('gateway.py','storage_transport.py'):
            shutil.copy2(BASE/'package/results/concurrency_original_step5_20261024'/name,adapter/name)
        for name in ('controlled_gc_gateway_20261027.py','tfs_warmup_gateway_20261027.py'):
            shutil.copy2(BASE/name,A/name)
        runner=(BASE/'base_runner.py').read_text()
        before="main = [json.loads(l) for l in (ROOT/'results/input360/trace.jsonl').read_text().splitlines()]"
        assert runner.count(before)==1
        runner=runner.replace(before,"from sim.trace_io import load_trace\n    main = load_trace(ROOT/'results/input360/trace.jsonl')")
        (A/'base_runner.py').write_text(runner)
        shutil.copy2(__file__,A/'run.py')
        inp=root/'results/input360';inp.mkdir()
        cfg=read(BASE/'package/results/input360/config.json')
        cfg['storage'].update(disk_Bps=90e9,shared_link_Bps=90e9)
        cfg['admission']['reference_bandwidth_Bps']=90e9
        cfg.update(wall_time_scale=1.,policy='priority',request_ordering='edf')
        cfg['experiment']=dict(batch_feedback=True,coalesced_feedback=False)
        sys.path.insert(0,str(root))
        from sim.request_profile import normalize_config,build_profile
        from sim.mooncake import validate_hit_row
        cfg=normalize_config(cfg)
        with (ALLOC/dataset/'allocation.csv').open() as stream:
            allocation=list(csv.DictReader(stream))
        motif=[1332,995,2581,640,1366,23748];meta=[]
        with (inp/'trace.jsonl').open('w') as stream:
            for r in allocation:
                n,h,u=int(r['n']),int(r['h']),int(r['u'])
                row=dict(id=f'conversation:L{r["source_line"]}',arrival_s=float(r['arrival_s']),
                    prompt_token_ids=(motif*((n+5)//6))[:n],output_tokens=1,
                    remote_prefix_tokens=h,hit_tokens=h,u_tokens=u,
                    full_hit_adjusted=u==1,cache_model='first_seen',class_id=r['class_id'],
                    construction=dict(global_compute_budget=204699,
                        original_input_length=n,source_line=int(r['source_line']),
                        original_output_used=False,hash_ids_used=False))
                validate_hit_row(row,16)
                profile=build_profile(row,cfg)
                meta.append(dict(id=row['id'],class_id=row['class_id'],prompt_tokens=n,h=h,u=u,
                    arrival_s=row['arrival_s'],read_bytes=profile.estimated_read_bytes,
                    compute_s=profile.standalone_compute_s,T0_s=profile.reference_service_s,
                    deadline_width_s=profile.deadline_s-profile.arrival_s,
                    pressure_Bps=profile.pressure_Bps))
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
        assert len(meta)==360 and sum(r['prompt_tokens'] for r in meta)==5117755
        assert sum(r['u'] for r in meta)==204699 and sum(r['read_bytes'] for r in meta)==539571462144
        write(inp/'config.json',cfg);write(A/'profiles.json',meta)
        frozen={str(f.relative_to(root)):sha(f) for name in ('sim','scripts','configs','patched')
                for f in (root/name).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
        frozen['results/concurrency_original_step5_20261024/gateway.py']=sha(adapter/'gateway.py')
        frozen['results/concurrency_original_step5_20261024/storage_transport.py']=sha(adapter/'storage_transport.py')
        write(A/'input_check.json',dict(dataset=dataset,input_sha256=sha(inp/'trace.jsonl'),
            config_sha256=sha(inp/'config.json'),allocation_sha256=sha(ALLOC/dataset/'allocation.csv'),
            native_sha256=sha(OVERLAY/'dynamo/_core.abi3.so'),runtime_sha256=frozen,
            source_head=subprocess.check_output(['git','-C',str(P.parent),'rev-parse','HEAD'],text=True).strip(),
            working_tree_dirty=True,bandwidth_Bps=90e9,K=1,
            input_tokens=5117755,new_tokens=204699,read_bytes=539571462144,
            reference_compute_s=sum(r['compute_s'] for r in meta)))
        (A/'source_worktree.diff').write_text(subprocess.check_output(['git','-C',str(P.parent),'diff'],text=True))
    write(EXP/'plan.json',dict(seed=SEED,datasets=['ABC','ABCD'],arms=ARMS,
        formal_requests=360,warmup_requests=72,K=1,shared_storage_Bps=90e9,
        sorting='EDF for all three arms',release='first_token',forced_admission=False,
        common_runtime='Current working tree snapshot; synchronous native-batch layer feedback, compact trace, active permit index; coalesced feedback off.',
        repeats=1,limits='Single run per arm. New synthetic hit allocation, not an original96% optimization claim.'))
    print('prepared two frozen inputs and common runtime snapshots',flush=True)


def check_frozen(dataset):
    A=EXP/dataset;root=A/'package';check=read(A/'input_check.json')
    assert check['K']==1 and check['bandwidth_Bps']==90e9
    assert sha(root/'results/input360/trace.jsonl')==check['input_sha256']
    assert sha(root/'results/input360/config.json')==check['config_sha256']
    assert sha(OVERLAY/'dynamo/_core.abi3.so')==check['native_sha256']
    assert all(sha(root/name)==value for name,value in check['runtime_sha256'].items())
    return A,root,check


def audit(dataset,name,smoke):
    A,root,expected=check_frozen(dataset);folder=root/'results'/name/'run'
    cfg,state=read(folder/'config.json'),read(folder/'admission.final.json')
    rows={k:v for k,v in state['records'].items() if k.startswith('conversation:')}
    intended={r['id']:r for r in read(A/'profiles.json')}
    assert not state['fatal'] and not state['invalid_reasons'] and not state['pending_engine_events']
    assert not state['snapshot']['heavy_held'] and not state['snapshot']['pending_ids']
    assert cfg['wall_time_scale']==1 and cfg['storage']['disk_Bps']==cfg['storage']['shared_link_Bps']==90e9
    assert cfg['admission']['reference_bandwidth_Bps']==90e9
    for rid,rec in rows.items():
        a,b=rec['profile'],intended[rid]
        assert a['prompt_tokens']==b['prompt_tokens']
        assert a['estimated_read_bytes']==b['read_bytes']==rec['submitted_read_bytes']
        for field,target in [('standalone_compute_s','compute_s'),('reference_service_s','T0_s'),('pressure_Bps','pressure_Bps')]:
            assert a[field]==b[target],(field,a[field],b[target])
        assert abs((a['deadline_s']-a['arrival_s'])-b['deadline_width_s'])<1e-9
        assert rec['state']=='FINISHED' and not rec['lease_open']
    assert len(rows)==(6 if smoke and dataset=='ABC' else 8 if smoke else 360)
    if not smoke:
        assert sum(intended[rid]['u'] for rid in rows)==204699
        assert sum(rec['submitted_read_bytes'] for rec in rows.values())==539571462144
    ttft=sorted((r['first_token_s']-r['profile']['arrival_s'])*1000 for r in rows.values())
    result=dict(status='passed',dataset=dataset,name=name,formal=len(rows),K=1,
        mean_ttft_ms=sum(ttft)/len(ttft),p95_ttft_ms=ttft[max(0,math_ceil(.95*len(ttft))-1)],
        completion_ms=(max(r['first_token_s'] for r in rows.values())-min(r['profile']['arrival_s'] for r in rows.values()))*1000,
        slo_pct=sum(r['first_token_s']<=r['profile']['deadline_s'] for r in rows.values())/len(rows)*100)
    write(folder/'global4.check.json',result);print(json.dumps(result),flush=True)


def math_ceil(value):
    import math
    return math.ceil(value)


def run(dataset,arm,smoke=False):
    A,root,expected=check_frozen(dataset)
    alpha=ARMS[arm];name=('smoke-' if smoke else '')+arm
    assert not (root/'results'/name).exists(),'Never overwrite an attempt'
    source=(A/'base_runner.py').read_text()
    changes={
        "case = ROOT/'results'/('smoke' if smoke else arm)":"case = ROOT/'results'/CASE_NAME",
        "cfg.update(policy='priority' if new else 'fcfs',request_ordering='budget_priority' if new else 'fcfs')":"cfg.update(policy='priority',request_ordering='edf')",
        "cfg['routing_accounting'].update(enabled=new,waiting_factor=.1)":"cfg['routing_accounting'].update(enabled=new,waiting_factor=ALPHA)",
        "main = [dict(r,arrival_s=0.) for r in sorted(main,key=lambda r:len(r['prompt_token_ids']))[::60]]":"main = [dict(r,arrival_s=0.) for label in sorted({r['class_id'] for r in main}) for r in sorted((v for v in main if v['class_id']==label),key=lambda v:len(v['prompt_token_ids']))[::max(1,len([v for v in main if v['class_id']==label])-1)]]",
        "history=control('warmup/finish');assert history['status']=='pass'":"history=control('warmup/finish');assert history['status']=='pass'\n                    storage=StorageClient(storage_path[0]);storage.gc_end(collect=False)\n                    boundary=storage.gc_begin();storage.close()\n                    write(out/'storage.gc.formal.begin.json',boundary)",
        'from check_budget_priority import check as budget_check':'from check_edf_priority import check as budget_check',
        "case/'run/budget_priority.check.json'":"case/'run/edf_priority.check.json'",
        'score_check(case/\'run\',require_discount=True)':'score_check(case/\'run\',require_discount=ALPHA<1)',
    }
    for before,after in changes.items():
        assert source.count(before)==1,before
        source=source.replace(before,after)
    executed=A/f'executed-{name}.py';executed.write_text(source)
    sys.path[:0]=[str(root),str(root/'scripts')]
    scope=dict(__name__='global4_runtime',__file__=str(executed))
    exec(compile(source,str(executed),'exec'),scope)
    scope.update(W=W,P=P,A=A,ROOT=root,OVERLAY=OVERLAY,CASE_NAME=name,ALPHA=alpha)
    write(A/f'launch-{name}.json',dict(dataset=dataset,arm=arm,alpha=alpha,smoke=smoke,
        host=host(),runner_sha256=sha(Path(__file__)),executed_sha256=sha(executed),
        input_sha256=expected['input_sha256']))
    scope['run']('immediate' if arm=='direct' else 'new-tfs',smoke=smoke)
    audit(dataset,name,smoke)


def matrix():
    jobs=[dict(dataset=d,arm=a) for d in ('ABC','ABCD') for a in ARMS]
    random.Random(SEED).shuffle(jobs)
    assert not (EXP/'matrix.plan.json').exists(),'Do not overwrite a matrix'
    frozen={str(HERE/'run.py'):sha(HERE/'run.py')}
    for dataset in ('ABC','ABCD'):
        A,root,_=check_frozen(dataset)
        for name in ('base_runner.py','controlled_gc_gateway_20261027.py','tfs_warmup_gateway_20261027.py','input_check.json'):
            frozen[str(A/name)]=sha(A/name)
    write(EXP/'matrix.plan.json',dict(seed=SEED,jobs=jobs,files=frozen,repeats=1,host=host()))
    completed=[]
    for job in jobs:
        assert all(sha(Path(p))==h for p,h in frozen.items()),'Frozen driver changed'
        log=EXP/f'{job["dataset"]}-{job["arm"]}.log'
        print(json.dumps(dict(event='starting',**job)),flush=True)
        with log.open('w') as stream:
            child=subprocess.Popen([sys.executable,str(HERE/'run.py'),'run',job['dataset'],job['arm']],cwd=W,stdout=stream,stderr=subprocess.STDOUT)
            try:code=child.wait()
            except BaseException:
                import signal
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10)
                raise
        entry=dict(**job,returncode=code,host=host())
        if not code:entry['result']=read(EXP/job['dataset']/f'package/results/{job["arm"]}/run/global4.check.json')
        completed.append(entry)
        write(EXP/'matrix.progress.json',dict(status='failed' if code else 'completed' if len(completed)==6 else 'running',completed=completed,total=6))
        print(json.dumps(dict(event='completed',**entry)),flush=True)
        if code:raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','matrix'])
    parser.add_argument('dataset',nargs='?',choices=['ABC','ABCD'])
    parser.add_argument('arm',nargs='?',choices=list(ARMS))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='matrix':matrix()
    else:
        assert args.dataset and args.arm
        run(args.dataset,args.arm,args.smoke)
