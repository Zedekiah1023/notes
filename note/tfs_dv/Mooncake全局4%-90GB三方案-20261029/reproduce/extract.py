"""逐运行提取全局4%实验，保留零读取请求并核对字节积分。"""
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st

W=next(p for p in Path(__file__).resolve().parents if (p/'workspace.json').is_file())
P=W/'data/dynamo_vllm_simulation/dynamo_vllm_cpu_sim'
EXP=P/'results/mooncake360_global4_90g_20261029'
OUT=W/'note/tfs_dv/Mooncake全局4%-90GB三方案-20261029'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def csvout(p,values):
    with p.open('w',newline='') as f:
        fields=list(dict.fromkeys(k for row in values for k in row))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(values)


def merge(xs):
    result=[]
    for a,b in sorted(xs):
        if b<=a:continue
        if result and a<=result[-1][1]+1e-12:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result


def length(xs):
    return math.fsum(b-a for a,b in merge(xs))


def subtract(xs,ys):
    result=[];cuts=merge(ys)
    for a,b in merge(xs):
        for c,d in cuts:
            if d<=a:continue
            if c>=b:break
            if c>a:result.append([a,min(c,b)])
            a=max(a,d)
            if a>=b:break
        if a<b:result.append([a,b])
    return result


def percentile(xs,q):
    return sorted(xs)[max(0,math.ceil(len(xs)*q)-1)]


def extract(dataset,arm):
    A=EXP/dataset;case=A/f'package/results/{arm}';run=case/'run'
    cfg=read(run/'config.json');state=read(run/'admission.final.json');storage=read(run/'storage.final.json')
    client=read(run/'measured/report.json');driver=read(case/'driver.json')
    intended={r['id']:r for r in read(A/'profiles.json')}
    assert len(intended)==360 and client['n_total']==client['n_ok']==360 and not client['n_skipped']
    assert driver['status']=='passed' and not driver['guards'] and not driver['alive']
    assert read(run/'global4.check.json')['status']=='passed'
    assert read(run/'warmup.check.json')['coverage']==list(range(12))
    assert not state['fatal'] and not state['invalid_reasons'] and not state['pending_engine_events']
    assert not state['snapshot']['pending_ids'] and not state['snapshot']['heavy_held']
    assert storage['pending_reads']==0 and cfg['wall_time_scale']==1
    assert cfg['storage']['disk_Bps']==cfg['storage']['shared_link_Bps']==90e9
    origin=None
    for ready in (run/'fine_trace').glob('ready-*.json'):
        if read(ready)['role']!='gateway':continue
        for line in ready.with_name(ready.name.replace('ready-','trace-').replace('.json','.jsonl')).open():
            e=json.loads(line)
            if e['event']=='gateway_origin':origin=e['origin_s'];break
    assert origin is not None
    formal={rid:state['records'][rid] for rid in intended}
    first=origin+min(r['profile']['arrival_s'] for r in formal.values())
    end=origin+max(r['first_token_s'] for r in formal.values());span=end-first
    requests=[]
    for rid,r in formal.items():
        p=r['profile'];m=intended[rid];cp=client['requests'][rid]
        assert r['state']=='FINISHED' and not r['lease_open']
        assert r['submitted_read_bytes']==m['read_bytes']
        ttft=(r['first_token_s']-p['arrival_s'])*1000
        wait=None if r['load_submit_s'] is None else (r['load_submit_s']-p['arrival_s'])*1000
        requests.append(dict(dataset=dataset,arm=arm,id=rid,class_id=m['class_id'],worker=r['worker_id'],
            input_tokens=m['prompt_tokens'],remote_tokens=m['h'],new_tokens=m['u'],read_bytes=m['read_bytes'],
            ttft_ms=ttft,slo=int(r['first_token_s']<=p['deadline_s']),
            deadline_ms=(p['deadline_s']-p['arrival_s'])*1000,
            client_ttft_ms=cp['ttft_s']*1000,
            lateness_ms=(cp['actual_arrival_s']-cp['planned_arrival_s'])*1000,
            read_before_wait_ms=wait))
    bindings=storage['hit_bindings'];sid=lambda s:bindings[s.replace('-','')[:32]]
    comp=[];io=defaultdict(list);model=Counter();tokens=Counter();hist=Counter();members=0
    for path in sorted(run.glob('worker*.steps.jsonl')):
        worker=int(path.name.split('.')[0][6:]);opened={}
        for line in path.open():
            e=json.loads(line)
            if e.get('num_scheduled_tokens'):
                ids=[sid(s) for s in e['num_scheduled_tokens']]
                if not set(ids)&set(intended):continue
                assert set(ids)<=set(intended)
                hist[len(ids)]+=1;members+=len(ids)
                model[worker]+=e['timing']['duration_s']
                for engine,u in e['num_scheduled_tokens'].items():tokens[sid(engine)]+=u
            if e.get('event')=='layer_compute_start':opened[e['step'],e['layer']]=e
            elif e.get('event')=='layer_compute_complete':
                start=opened.pop((e['step'],e['layer']));ids=[sid(s) for s in start['req_ids']]
                if not set(ids)&set(intended):continue
                assert set(ids)<=set(intended)
                t=start['occurred_mono']-first;duration=start['duration_model_s']
                assert -1e-6<=t<=t+duration<=span+1e-5
                labels={intended[rid]['class_id'] for rid in ids}
                comp.append([worker,t,t+duration,next(iter(labels)) if len(labels)==1 else 'mixed'])
    assert sum(tokens.values())==204699 and all(tokens[rid]==r['u'] for rid,r in intended.items())
    assert len(comp)==sum(hist.values())*78
    assert abs(sum(c[2]-c[1] for c in comp)-sum(model.values()))<1e-7
    changes=defaultdict(list);expected=539571462144;read_bytes=0;loaded=set()
    demands={rid:r['demand']*sum(r['counts'].values()) for rid,r in storage['layer_loads'].items()}
    for i,r in enumerate(storage['layer_read_history']):
        rid=sid(r['rid'])
        if rid not in intended:continue
        a=storage['t0_monotonic_s']+r['submitted_s']-first
        b=storage['t0_monotonic_s']+r['completed_s']-first
        assert -1e-6<=a<b<=span+1e-5
        io[r['worker']].append([a,b]);read_bytes+=r['bytes'];loaded.add(rid)
        changes[r['submitted_s']].append((i,True,r['rid']))
        changes[r['completed_s']].append((i,False,r['rid']))
    assert loaded=={rid for rid,r in intended.items() if r['h']>0}
    assert len(loaded)==330 and read_bytes==expected
    active={};previous=None;band=[];integral=0.;over=0.;saturated=0.
    # 先相减大时钟，再平移小的相对时刻，避免两次大数相减损失区间精度。
    storage_offset=storage['t0_monotonic_s']-first
    for t,events in sorted(changes.items()):
        if previous is not None and t>previous:
            a=previous+storage_offset
            b=t+storage_offset
            actual=90e9 if active else 0.;demand=sum(demands[rid] for rid in active.values())
            dt=t-previous;integral+=dt*actual;over+=dt*(demand>90e9+.09);saturated+=dt*bool(active)
            band.append([a,b,demand/1e9,actual/1e9])
        for key,add,rid in events:
            if add:active[key]=rid
            else:del active[key]
        previous=t
    assert not active and abs(integral-expected)<expected*1e-11,(dataset,arm,integral-expected)
    devices=[];red=[]
    for worker in range(12):
        cc=[[c[1],c[2]] for c in comp if c[0]==worker]
        assert abs(length(cc)-model[worker])<1e-8
        rr=subtract(io[worker],cc);red.extend([[worker,a,b,'io'] for a,b in rr])
        red_s=length(rr);other=span-model[worker]-red_s
        assert other>=-1e-7
        devices.append(dict(dataset=dataset,arm=arm,worker=worker,model_s=model[worker],
            io_without_model_s=red_s,other_s=max(0,other),span_s=span,model_pct=model[worker]/span*100))
    ts=[r['ttft_ms'] for r in requests]
    result=dict(dataset=dataset,arm=arm,requests=360,completion_ms=span*1000,
        mean_ttft_ms=st.mean(ts),p50_ms=percentile(ts,.5),p95_ms=percentile(ts,.95),
        slo_pct=st.mean(r['slo'] for r in requests)*100,steps=sum(hist.values()),
        mean_batch=members/sum(hist.values()),model_s=sum(model.values()),
        model_pct=sum(model.values())/(12*span)*100,
        worker_model_cv=st.pstdev([model[w] for w in range(12)])/st.mean([model[w] for w in range(12)]),
        bandwidth_mean_GBps=expected/span/1e9,over90_pct=over/span*100,saturated_pct=saturated/span*100,
        no_io_s=span-saturated,band_integral_error_B=integral-expected,
        batch1=hist[1],batch2=hist[2],batch3=hist[3],batch4=hist[4],
        chunked_requests=sum(tokens[rid]!=intended[rid]['u'] for rid in intended),
        lateness_mean_ms=st.mean(r['lateness_ms'] for r in requests),
        lateness_p95_ms=percentile([r['lateness_ms'] for r in requests],.95),
        host_min_available_gib=driver['min_available_gib'],host_max_swap_gib=driver['max_swap_gib'])
    # Membership multiplicity, rather than token equality, detects split requests.
    memberships=Counter()
    for path in run.glob('worker*.steps.jsonl'):
        for line in path.open():
            e=json.loads(line)
            for raw in e.get('num_scheduled_tokens',{}):
                rid=sid(raw)
                if rid in intended:memberships[rid]+=1
    result['chunked_requests']=sum(n>1 for n in memberships.values())
    for label in sorted({r['class_id'] for r in requests}):
        group=[r for r in requests if r['class_id']==label]
        result[label+'_ttft_ms']=st.mean(r['ttft_ms'] for r in group)
        result[label+'_slo_pct']=st.mean(r['slo'] for r in group)*100
    payload=dict(dataset=dataset,arm=arm,span_s=span,comp=comp,red=red,band=band,
        first_model_by_worker={str(w):min([c[1] for c in comp if c[0]==w],default=span) for w in range(12)})
    checks=dict(status='passed',run=str(run),formal=360,warmup=72,new_tokens=204699,
        read_bytes=expected,loaded_requests=330,zero_read_requests=30,
        band_integral_error_B=integral-expected,clock_scale=1,
        raw_sha256={name:sha(run/name) for name in ['config.json','check.json','global4.check.json','formula_check.json','layer_check.json','hit_check.json','admission.final.json','storage.final.json']})
    return result,requests,devices,payload,checks


def main():
    progress=read(EXP/'matrix.progress.json')
    assert progress['status']=='completed' and len(progress['completed'])==6
    data=OUT/'report_data';data.mkdir(parents=True,exist_ok=True)
    results=[];requests=[];devices=[];trajectories=[];checks=[]
    for dataset in ('ABC','ABCD'):
        for arm in ('direct','tfs-a01','tfs-a1'):
            s,rr,dd,tr,ch=extract(dataset,arm)
            results.append(s);requests+=rr;devices+=dd;trajectories.append(tr);checks.append(ch)
            print(dataset,arm,s['mean_ttft_ms'],s['completion_ms'],flush=True)
    for name,rows in [('run_summary.csv',results),('requests.csv',requests),('device_utilization.csv',devices)]:
        csvout(data/name,rows)
    write(data/'summary.json',results);write(data/'extraction_checks.json',checks)
    with gzip.open(data/'trajectories.json.gz','wt') as stream:
        json.dump(trajectories,stream,separators=(',',':'))
    write(data/'matrix.plan.json',read(EXP/'matrix.plan.json'))
    write(data/'matrix.progress.json',progress)
    write(data/'input_profiles.json',{d:read(EXP/d/'profiles.json') for d in ('ABC','ABCD')})
    write(data/'provenance.json',{d:read(EXP/d/'input_check.json') for d in ('ABC','ABCD')})
    print('extraction passed',flush=True)


if __name__=='__main__':
    main()
