"""Freeze Mooncake96/block16 input and run four prewarmed K1 native arms.

Experiment orchestration only. Simulator, native picker and timing models are
copied without edits; warmup and GC controls are separately archived.
"""
import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import psutil

W = Path('/Users/zed/PycharmProjects/kv_aware_net')
P = W/'data/dynamo_vllm_simulation/dynamo_vllm_cpu_sim'
A = P/'results/mooncake360_block16_four_20261028'
ROOT = A/'package'
OLD = W/'note/tfs_dv/Mooncake96命中四秒单多进程-20261026'
OVERLAY = W/'temp/weighted-block16-mac-20261027/python'
ARMS = ('immediate', 'old-tfs', 'new-tfs', 'idle-bypass')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def environment():
    os.environ.update(PYTHONHASHSEED='0', VLLM_PLUGINS='', HF_HUB_OFFLINE='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      VLLM_ALLOW_LONG_MAX_MODEL_LEN='1', VLLM_LOGGING_LEVEL='ERROR',
                      DYN_TCP_RPC_HOST='127.0.0.1')
    for key in ('SIM_FINE_TRACE_DIR', 'DYN_WAITING_SCORE_FACTOR', 'DYN_WAITING_SCORE_READY'):
        os.environ.pop(key, None)
    sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    for name in ('sim', 'scripts', 'configs', 'patched'):
        shutil.copytree(P/name, ROOT/name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('versions.json', 'source-fingerprints.json'):
        shutil.copy2(P/name, ROOT/name)
    for name in ('.venv', 'upstream'):
        (ROOT/name).symlink_to(P/name, target_is_directory=True)
    adapters = ROOT/'results/concurrency_original_step5_20261024'
    adapters.mkdir(parents=True)
    for name in ('gateway.py', 'storage_transport.py'):
        shutil.copy2(P/'results/concurrency_original_step5_20261024'/name, adapters/name)
    for name in ('controlled_gc_gateway_20261027.py', 'tfs_warmup_gateway_20261027.py'):
        shutil.copy2(W/'temp'/name, A/name)
    shutil.copy2(__file__, A/'runner.py')
    source = W/'data/status_aware_network/mooncake_trace/conversation_trace.jsonl'
    assert sha(source) == read(OLD/'report_data/input_check.json')['raw_trace_sha256']
    raw = [json.loads(s) for s in source.read_text().splitlines()][:360]
    cfg = read(OLD/'report_data/native-immediate/config.json')
    cfg.update(block_size=16, num_blocks=math.ceil(482537/16), wall_time_scale=1.)
    cfg['admission'].update(rpc_transport='unix', idle_bypass=False)
    cfg['routing_accounting'] = dict(enabled=False, waiting_factor=.1, python_path=str(OVERLAY))
    environment()
    from sim.request_profile import normalize_config, build_profile
    from sim.mooncake import validate_hit_row
    cfg = normalize_config(cfg)
    motif = [1332, 995, 2581, 640, 1366, 23748]
    rows = []
    for i, r in enumerate(raw):
        n = r['input_length']; old_h = n*96//100; h = old_h//16*16
        row = dict(id=f'conversation:L{i+1}',
                   arrival_s=(r['timestamp']-raw[0]['timestamp'])/(raw[-1]['timestamp']-raw[0]['timestamp'])*4,
                   prompt_token_ids=(motif*((n+5)//6))[:n], output_tokens=1,
                   remote_prefix_tokens=h, hit_tokens=h, u_tokens=n-h,
                   full_hit_adjusted=h==n-1, cache_model='first_seen', class_id='unclassified',
                   mooncake_source=dict(file=source.name, line=i+1, timestamp=r['timestamp'],
                                        original_output_length=r['output_length']),
                   hit_assumption='floor(floor(0.96*input_length)/16)*16; artificial remote hit',
                   block1_hit_tokens=old_h)
        validate_hit_row(row, 16); rows.append(row)
    pressures = sorted(build_profile(r, cfg).pressure_Bps for r in rows)
    x = (len(pressures)-1)*.75; j = int(x)
    threshold = pressures[j]+(pressures[j+1]-pressures[j])*(x-j)
    for r in rows:
        r['class_id'] = 'A' if build_profile(r,cfg).pressure_Bps >= threshold else 'B'
    inp = ROOT/'results/input360'; inp.mkdir()
    (inp/'trace.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    write(inp/'config.json', cfg)
    assert len(rows)==360 and rows[-1]['arrival_s']==4 and sum(len(r['prompt_token_ids']) for r in rows)==5117755
    profiles = [build_profile(r,cfg) for r in rows]
    check = dict(requests=360, classes={k:sum(r['class_id']==k for r in rows) for k in 'ABC'},
                 planned_arrival_span_s=4., block_size=16, workers=12, bandwidth_Bps=120e9,
                 input_tokens=sum(len(r['prompt_token_ids']) for r in rows),
                 hit_tokens=sum(r['hit_tokens'] for r in rows),
                 new_tokens=sum(r['u_tokens'] for r in rows),
                 block1_hit_tokens=sum(r['block1_hit_tokens'] for r in rows),
                 total_read_bytes=sum(p.estimated_read_bytes for p in profiles),
                 reference_compute_instance_s=sum(p.standalone_compute_s for p in profiles),
                 raw_trace_sha256=sha(source), effective_trace_sha256=sha(inp/'trace.jsonl'),
                 plot_threshold_Bps=threshold,
                 hit_model='counterfactual uniform96%, floored to complete 16-token blocks',
                 runtime_sources={str(f.relative_to(ROOT)):sha(f) for name in ('sim','scripts','patched')
                                  for f in (ROOT/name).rglob('*') if f.is_file() and '__pycache__' not in f.parts},
                 source_commit=subprocess.check_output(['git','-C',str(P.parent),'rev-parse','HEAD'],text=True).strip(),
                 native_sha256=sha(OVERLAY/'dynamo/_core.abi3.so'))
    check.update(token_hit_ratio=check['hit_tokens']/check['input_tokens'],
                 offered_storage_load=check['total_read_bytes']/120e9/4,
                 offered_reference_compute_load=check['reference_compute_instance_s']/12/4)
    write(A/'input_check.json',check)
    write(A/'plan.json',dict(input_check=check,arms=ARMS,wall_time_scale=1.,warmup=72,
          changes='block1 -> block16, h floored to whole blocks, logical block pool capacity preserved; current runtime/K1/prewarm for all arms',
          release='first_token',repeats=1))
    print(json.dumps({k:v for k,v in check.items() if k!='runtime_sources'},ensure_ascii=False),flush=True)


def run(arm, smoke=False):
    environment(); os.chdir(ROOT)
    import httpx
    import replay_http
    import run_tfs_stack
    from check_stack_routing import metrics_drained
    from sim.request_profile import normalize_config
    # Read-only post-run admission snapshots may include tens of thousands of
    # layer/permit facts. Extend only diagnostic GET timeouts; runtime permit,
    # storage and routing RPC timeouts and measured request paths are unchanged.
    original_request_json=run_tfs_stack.request_json
    def diagnostic_json(url):
        if '/sim/admission' not in url:return original_request_json(url)
        with urllib.request.urlopen(url,timeout=30) as response:return json.load(response)
    run_tfs_stack.request_json=diagnostic_json
    case = ROOT/'results'/('smoke' if smoke else arm)
    case.mkdir(exist_ok=False)
    from sim.trace_io import load_trace
    main = load_trace(ROOT/'results/input360/trace.jsonl')
    cfg = read(ROOT/'results/input360/config.json')
    new = arm in ('new-tfs','idle-bypass')
    cfg['admission'].update(mode='immediate' if arm=='immediate' else 'tfs',
                            placement='scheduler' if new else 'gateway',
                            classification='arrival' if new else 'dynamic',
                            idle_bypass=arm=='idle-bypass',aging_enabled=False)
    cfg.update(policy='priority' if new else 'fcfs',request_ordering='budget_priority' if new else 'fcfs')
    cfg['routing_accounting'].update(enabled=new,waiting_factor=.1)
    if smoke:
        main = [dict(r,arrival_s=0.) for r in sorted(main,key=lambda r:len(r['prompt_token_ids']))[::60]]
        cfg['workers'] = 2
    cfg = normalize_config(cfg)
    warm=[]; phases=[]
    if not smoke:
        ordered=sorted(main,key=lambda r:len(r['prompt_token_ids']))
        for name, q in [('small',.1),('median',.5),('large',.9)]:
            template=ordered[int((len(ordered)-1)*q)]
            ids=[]
            for i in range(24):
                row=dict(template,id=f'warmup-{name}-{i:02d}',arrival_s=0.)
                warm.append(row);ids.append(row)
            phase=case/f'warmup-{name}.jsonl';phase.write_text(''.join(json.dumps(r)+'\n' for r in ids));phases.append((name,phase))
    measured=case/'measured_trace.jsonl';measured.write_text(''.join(json.dumps(r)+'\n' for r in main))
    combined=case/'trace.jsonl';combined.write_text(''.join(json.dumps(r)+'\n' for r in warm+main))
    write(case/'effective_config.json',cfg)
    for name in ('gateway.py','storage_transport.py'):
        shutil.copy2(ROOT/'results/concurrency_original_step5_20261024'/name,case/name)
    for port in (28765,28766):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);probe.bind(('127.0.0.1',port))
    original_popen=subprocess.Popen;original_client=httpx.AsyncClient;original_replay=replay_http.replay
    original_verify=run_tfs_stack.verify;original_killpg=os.killpg
    trace_dirs=set();storage_path=[None];phase_state=['startup']
    def safe_killpg(pid,sig):
        try:return original_killpg(pid,sig)
        except PermissionError:
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
    os.killpg=safe_killpg
    def launch(argv,*args,**kw):
        if argv[1:3]==['-m','sim.stack.admission_service']:
            argv=[argv[0],str(case/'gateway.py'),*argv[3:]]
        elif argv[1:3]==['-m','sim.stack.storage_service']:
            argv=[argv[0],str(case/'storage_transport.py'),*argv[3:]]
        env=kw.get('env',{})
        if env.get('SIM_STACK_CONFIG'):
            folder=Path(env['SIM_STACK_CONFIG']).parent/'fine_trace';folder.mkdir(exist_ok=True);trace_dirs.add(folder)
            kw['env']={**env,'SIM_FINE_TRACE_DIR':str(folder)}
            if env.get('SIM_STACK_STORAGE_SOCKET'):storage_path[0]=env['SIM_STACK_STORAGE_SOCKET']
            if argv[1].endswith('gateway.py'):
                argv=[argv[0],str(A/'controlled_gc_gateway_20261027.py'),
                      *([] if smoke else [str(A/'tfs_warmup_gateway_20261027.py')]),*argv[1:]]
            elif argv[1].endswith('storage_transport.py'):
                argv=[argv[0],'-m','sim.stack.fine_trace','storage',*argv[1:]]
        return original_popen(argv,*args,**kw)
    class Client(original_client):
        def __init__(self,*args,**kw):
            kw.update(timeout=600,limits=httpx.Limits(max_connections=512,max_keepalive_connections=64));super().__init__(*args,**kw)
    def control(action):
        url='http://127.0.0.1:28766/diagnostic/'+('warmup/finish' if action=='warmup/finish' else 'gc/'+action)
        req=urllib.request.Request(url,data=b'',method='POST')
        with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
    async def replay(trace,url,model,out,lossy,api,**kwargs):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'configs/model'));cache={}
        for r in warm+main:
            ids=r['prompt_token_ids'];key=(len(ids),ids[0])
            if key not in cache:
                text=tokenizer.decode(ids);assert tokenizer.encode(text)==ids;cache[key]=(ids,text)
            else:assert cache[key][0]==ids
        class PreparedTokenizer:
            def decode(self,ids,*a,**k):
                known,text=cache[len(ids),ids[0]];assert known==list(ids);return text
            def encode(self,text,*a,**k):
                for ids,known in cache.values():
                    if text==known:return ids
                raise ValueError('unexpected prompt')
            def __getattr__(self,key):return getattr(tokenizer,key)
        loader=AutoTokenizer.from_pretrained;AutoTokenizer.from_pretrained=lambda *a,**k:PreparedTokenizer()
        client_type=httpx.AsyncClient;reports=[]
        class BorrowedClient:
            def __init__(self,*a,**k):pass
            async def __aenter__(self):return shared
            async def __aexit__(self,*a):return False
        try:
            async with client_type(timeout=600,trust_env=False) as shared:
                httpx.AsyncClient=BorrowedClient
                for name,path in phases:
                    phase_state[0]='warmup-'+name;dest=out/('warmup-'+name);dest.mkdir()
                    assert await original_replay(path,url,model,dest,lossy,api,**kwargs)==0
                    report=read(dest/'report.json');assert report['n_ok']==24;reports.append(report)
                if phases:
                    limit=time.monotonic()+30
                    while True:
                        state=run_tfs_stack.request_json('http://127.0.0.1:28766/sim/admission')
                        records=state['records']
                        if (len(records)==72 and all(r['state']=='FINISHED' and not r['lease_open'] for r in records.values())
                                and not state['snapshot']['pending_ids'] and metrics_drained(run_tfs_stack.router_metrics())):break
                        if time.monotonic()>limit:raise TimeoutError('warmup drain')
                        await asyncio.sleep(.02)
                    coverage=sorted({r['worker_id'] for r in records.values()});assert coverage==list(range(12)),coverage
                    from sim.stack.storage_client import StorageClient
                    storage=StorageClient(storage_path[0]);snap=storage.stats();storage.close();assert snap['pending_reads']==0
                    history=control('warmup/finish');assert history['status']=='pass'
                    write(out/'warmup.check.json',dict(status='pass',requests=72,coverage=coverage,pending_reads=0,
                          threshold_reset=True,shared_http_client=True,prefix_cache=False))
                phase_state[0]='measured';dest=out/'measured';dest.mkdir();control('disable')
                print(f'{arm}: warmup drained; measured {len(main)} starts',flush=True)
                try:assert await original_replay(measured,url,model,dest,lossy,api,**kwargs)==0
                finally:control('restore')
                main_report=read(dest/'report.json');assert main_report['n_ok']==len(main);reports.append(main_report)
                phase_state[0]='verification'
        finally:httpx.AsyncClient=client_type;AutoTokenizer.from_pretrained=loader
        requests={rid:r for report in reports for rid,r in report['requests'].items()};assert len(requests)==len(warm+main)
        entry=next(iter(reports[0]['requests'].values()));origin=entry['sent_perf_s']-entry['actual_arrival_s']
        for report in reports:
            first=next(iter(report['requests'].values()));phase_origin=first['sent_perf_s']-first['actual_arrival_s']
            for r in report['requests'].values():
                r['actual_arrival_s']=r['sent_perf_s']-origin;r['planned_arrival_s']+=phase_origin-origin
        all_report=dict(main_report,n_total=len(requests),n_ok=len(requests),n_skipped=0,requests=requests,
                        wall_clock_s=time.perf_counter()-origin,model_elapsed_s=time.perf_counter()-origin,
                        warmup_requests=len(warm),measured_requests=len(main),
                        ttft_mean_s=sum(r['ttft_s'] for r in requests.values())/len(requests),
                        scope='Audit including warmup; measured/report.json is formal only')
        write(out/'report.json',all_report);return 0
    def verify(*args,**kwargs):
        for folder in trace_dirs:
            people=[read(f) for f in folder.glob('ready-*.json')]
            for person in people:os.kill(person['pid'],signal.SIGUSR1)
            limit=time.monotonic()+20
            while any(not (folder/f"flushed-{person['pid']}.json").exists() for person in people):
                if time.monotonic()>limit:raise TimeoutError('fine trace flush')
                time.sleep(.05)
            write(folder/'flush_check.json',dict(status='pass',processes=people))
        return original_verify(*args,**kwargs)
    parent=psutil.Process();stopped=threading.Event();samples=[];captured={};guards=[];started=time.monotonic()
    def monitor():
        low=None
        with (case/'resources.jsonl').open('w',buffering=1) as stream:
            while not stopped.is_set():
                rss=0
                for process in parent.children(recursive=True):
                    try:captured[process.pid]=process.create_time();rss+=process.memory_info().rss
                    except psutil.Error:pass
                sample=dict(elapsed_s=time.monotonic()-started,phase=phase_state[0],rss=rss,
                            available=psutil.virtual_memory().available,swap=psutil.swap_memory().used)
                samples.append(sample);stream.write(json.dumps(sample)+'\n')
                low=(low or time.monotonic()) if sample['available']<768*2**20 else None
                if not guards and ((low and time.monotonic()-low>10) or sample['elapsed_s']>900):
                    guards.append(sample);os.kill(os.getpid(),signal.SIGINT)
                stopped.wait(1)
    result=dict(status='failed',arm=arm,smoke=smoke);thread=threading.Thread(target=monitor,daemon=True)
    subprocess.Popen=launch;httpx.AsyncClient=Client;replay_http.replay=replay;run_tfs_stack.verify=verify;thread.start()
    try:
        with (case/'driver.log').open('w') as stream,contextlib.redirect_stdout(stream):
            check=run_tfs_stack.run_one(cfg,combined,case/'run',Path(sys.executable),cfg['admission']['mode'],'first_token',cfg['workers'])
        assert check['requests']==len(warm+main)
        assert check['workers']==list(range(cfg['workers'])),check['workers']
        assert check['actual_submitted_read_bytes']==sum(r['hit_tokens']*cfg['storage']['kv_bytes_per_token'] for r in warm+main)
        if new:
            from check_budget_priority import check as budget_check
            from check_routing_accounting import check as score_check
            write(case/'run/budget_priority.check.json',budget_check(case/'run'))
            write(case/'run/routing.score.check.json',score_check(case/'run',require_discount=True))
        result.update(status='passed',check=check)
    except BaseException:
        result['error']=traceback.format_exc();raise
    finally:
        stopped.set();thread.join(5)
        subprocess.Popen=original_popen;httpx.AsyncClient=original_client;replay_http.replay=original_replay
        run_tfs_stack.verify=original_verify;os.killpg=original_killpg
        run_tfs_stack.request_json=original_request_json
        alive=[]
        for pid,created in captured.items():
            try:
                proc=psutil.Process(pid)
                if proc.create_time()==created and proc.status()!=psutil.STATUS_ZOMBIE:alive.append(pid)
            except psutil.Error:pass
        result.update(wall_s=time.monotonic()-started,guards=guards,alive=alive,
                      min_available_gib=min(s['available'] for s in samples)/2**30,
                      peak_children_rss_gib=max(s['rss'] for s in samples)/2**30,
                      max_swap_gib=max(s['swap'] for s in samples)/2**30)
        if guards or alive:result['status']='failed'
        write(case/'driver.json',result)
    assert result['status']=='passed'
    print(json.dumps(dict(phase='completed',arm=arm,smoke=smoke,wall_s=result['wall_s'],requests=len(main))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','smoke',*ARMS]);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:run('idle-bypass' if args.action=='smoke' else args.action,smoke=args.action=='smoke')
