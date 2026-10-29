"""保留原360条输入，仅分配整批4%的未命中计算额度。"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

W=next(p for p in Path(__file__).resolve().parents if (p/'workspace.json').is_file())
P=W/'data/dynamo_vllm_simulation/dynamo_vllm_cpu_sim'
SOURCE=W/'data/status_aware_network/mooncake_trace/conversation_trace.jsonl'
CONFIG=P/'results/mooncake360_prefill_v2_four_20261028/package/results/input360/config.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def csvout(p,values):
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]))
        writer.writeheader();writer.writerows(values)


def allocate(source,target,with_d):
    rows=[dict(r,class_id='A',u=r['n']%16 or 16)
          for r in sorted(source,key=lambda r:(r['n'],r['source_line']))]
    for r in rows[:30]:
        r.update(class_id='C',u=r['n'])
    for r in rows[30:120]:
        r['class_id']='B'
    if with_d:
        for r in rows[-20:]:
            r.update(class_id='D',u=r['u']+16*((2048-r['u'])//16))
    remaining=target-sum(r['u'] for r in rows)
    assert remaining>=0 and remaining%16==0
    b=[r for r in rows if r['class_id']=='B']
    capacities=[(r['n']-16-r['u'])//16 for r in b]
    units,total=remaining//16,sum(capacities)
    assert units<=total
    grants=[units*c//total for c in capacities]
    spare=units-sum(grants)
    order=sorted(range(len(b)),key=lambda i:(-(units*capacities[i]%total),b[i]['source_line']))
    for i in order[:spare]:
        grants[i]+=1
    for r,grant,cap in zip(b,grants,capacities):
        assert 0<=grant<=cap
        r['u']+=16*grant
    for r in rows:
        r['h']=r['n']-r['u']
        assert 1<=r['u']<=r['n'] and r['h']%16==0
        assert (r['h']==0)==(r['class_id']=='C')
    assert sum(r['u'] for r in rows)==target
    return sorted(rows,key=lambda r:r['source_line'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=W/'temp/tfs_opt_global4_20261029')
    args=parser.parse_args();out=args.output.resolve()
    assert not out.exists(),'Refuse to overwrite an existing analysis'
    out.mkdir(parents=True)
    with SOURCE.open() as f:
        raw=[json.loads(next(f)) for _ in range(360)]
    source=[dict(source_line=i+1,n=r['input_length'],timestamp=r['timestamp'])
            for i,r in enumerate(raw)]
    del raw
    N=sum(r['n'] for r in source);ideal=Fraction(N,25)
    floor_budget=N//25
    target=N-16*((N-floor_budget+15)//16)
    assert target<=ideal and target+16>ideal
    assert target%16==N%16
    sys.path.insert(0,str(P))
    from sim.request_profile import normalize_config,build_profile
    from sim.mooncake import validate_hit_row
    from sim.compute_time import estimate_step
    cfg=normalize_config(json.loads(CONFIG.read_text()))
    assert cfg['wall_time_scale']==1 and cfg['block_size']==16
    assert cfg['workers']==12 and cfg['max_num_seqs']==4 and cfg['max_num_batched_tokens']==8192
    assert cfg['storage']['disk_Bps']==120e9
    start,end=source[0]['timestamp'],source[-1]['timestamp']
    summaries=[];allstats=[]
    for name,with_d in [('ABC',False),('ABCD',True)]:
        folder=out/name;folder.mkdir()
        rows=allocate(source,target,with_d)
        assert rows==allocate(source,target,with_d),'Allocation must be deterministic'
        for r in rows:
            row=dict(id=f'{name}:L{r["source_line"]}',arrival_s=(r['timestamp']-start)/(end-start)*4,
                prompt_token_ids=range(r['n']),hit_tokens=r['h'],remote_prefix_tokens=r['h'],
                u_tokens=r['u'],cache_model='first_seen',full_hit_adjusted=r['u']==1,
                class_id=r['class_id'])
            validate_hit_row(row,16)
            p=build_profile(row,cfg)
            assert p.estimated_remote_tokens==r['h'] and p.prompt_tokens==r['n']
            assert p.estimated_read_bytes==r['h']*109824
            r.update(arrival_s=row['arrival_s'],read_bytes=p.estimated_read_bytes,
                read_GB=p.estimated_read_bytes/1e9,compute_ms=p.standalone_compute_s*1000,
                pressure_GBps=p.pressure_Bps/1e9,
                T0_ms=p.reference_service_s*1000,deadline_width_ms=(p.deadline_s-p.arrival_s)*1000)
        assert [r['n'] for r in rows]==[r['n'] for r in source]
        assert sum(r['h'] for r in rows)+sum(r['u'] for r in rows)==N
        csvout(folder/'allocation.csv',rows)
        classstats=[];batch_model_s=0.;batch_count=0
        for label in 'ABCD':
            xs=[r for r in rows if r['class_id']==label]
            if not xs:continue
            pending=[];tokens=0;class_model_s=0.;class_batches=0
            for r in xs+[None]:
                if pending and (r is None or len(pending)==4 or tokens+r['u']>8192):
                    timing=estimate_step(cfg['compute'],[(p['h'],p['u']) for p in pending])
                    assert timing['real_prefill_tokens']==tokens<=8192
                    class_model_s+=timing['duration_s'];class_batches+=1
                    pending=[];tokens=0
                if r is not None:
                    assert r['u']<=8192
                    pending.append(r);tokens+=r['u']
            batch_model_s+=class_model_s;batch_count+=class_batches
            stats=dict(case=name,class_id=label,requests=len(xs),
                input_min=min(r['n'] for r in xs),input_max=max(r['n'] for r in xs),
                new_min=min(r['u'] for r in xs),new_max=max(r['u'] for r in xs),
                total_new_tokens=sum(r['u'] for r in xs),
                read_GB_sum=sum(r['read_bytes'] for r in xs)/1e9,
                read_GB_mean=st.mean(r['read_GB'] for r in xs),
                read_GB_min=min(r['read_GB'] for r in xs),read_GB_max=max(r['read_GB'] for r in xs),
                compute_ms_mean=st.mean(r['compute_ms'] for r in xs),
                compute_ms_min=min(r['compute_ms'] for r in xs),compute_ms_max=max(r['compute_ms'] for r in xs),
                hypothetical_homogeneous_batch_model_s=class_model_s,
                hypothetical_batches=class_batches)
            classstats.append(stats);allstats.append(stats)
        A=next(s for s in classstats if s['class_id']=='A')
        for label in 'BC':
            s=next(s for s in classstats if s['class_id']==label)
            assert A['read_GB_min']>s['read_GB_max']
            assert A['compute_ms_max']<s['compute_ms_min']
        if with_d:
            D=next(s for s in classstats if s['class_id']=='D')
            assert D['read_GB_mean']>A['read_GB_mean']*3
            assert D['compute_ms_min']>A['compute_ms_max']*3
        read_bytes=sum(r['read_bytes'] for r in rows)
        assert read_bytes==(N-target)*109824
        summary=dict(case=name,requests=360,input_tokens=N,
            ideal_global_new_tokens=float(ideal),actual_global_new_tokens=target,
            global_new_pct=target/N*100,unused_budget_tokens=float(ideal-target),
            total_read_bytes=read_bytes,total_read_GB=read_bytes/1e9,
            K=1,arrival_window_s=4.,
            storage_input_load_pct=read_bytes/(120e9*4)*100,
            singleton_compute_instance_s=sum(r['compute_ms'] for r in rows)/1000,
            hypothetical_homogeneous_batch_model_s=batch_model_s,
            hypothetical_compute_input_load_pct=batch_model_s/(12*4)*100,
            hypothetical_batch_count=batch_count,classes=classstats)
        summaries.append(summary);write(folder/'summary.json',summary)
    assert summaries[0]['total_read_bytes']==summaries[1]['total_read_bytes']
    write(out/'summary.json',summaries);csvout(out/'class_summary.csv',allstats)
    source_hashes={str(p.relative_to(W)):sha(p) for p in [SOURCE,CONFIG,
        P/'sim/request_profile.py',P/'sim/reference_time.py',P/'sim/compute_time.py',
        P/'sim/timing_models/prefill_time_batch.py',P/'sim/timing_models/glm5_d950pr_reference.py']}
    write(out/'validation.json',dict(status='passed',requests_checked=720,
        formal_simulations_run=0,original_input_lengths_unchanged=True,
        output_length_used=False,hash_ids_used=False,global_budget_not_exceeded=True,
        source_sha256=source_hashes,
        artifact_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
