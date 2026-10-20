"""从 24 次到达率实验原始记录复算指标并生成报告；不运行仿真。"""
import csv
import hashlib
import json
import math
import re
import shutil
import statistics as st
import zipfile
from pathlib import Path

import report_style as style
style.configure()
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RAW = ROOT/'temp/rate_sweep_20261020'
DATA, FIG = OUT/'report_data', OUT/'figure'
FACTORS = [1, 1.5, 2]
MODES = ['immediate', 'tfs']
ORDER = [(f,m) for f in FACTORS for m in MODES]
LABEL = {'immediate':'直接提交', 'tfs':'TFS'}
SEEDS = {'A':[416,417,418], 'B':[424]}
CHECKS = {'runs':0, 'requests':0, 'worker_time_checks':0, 'bandwidth_integrals':0, 'cdf_checks':0}


def label(f,m):
    return f'{f:g}倍 · {LABEL[m]}'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (DATA/name).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def quantile(xs, q):
    return sorted(xs)[max(0,math.ceil(len(xs)*q)-1)]


def collect():
    manifest=json.loads((RAW/'manifest.json').read_text())
    assert len(manifest)==24
    records=[];selected={};requests=[];workers=[]
    for job in manifest:
        folder=Path(job['folder']); report=json.loads((folder/'report.json').read_text())
        events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
        assert sha(folder/'report.json')==job['report_sha256']
        assert sha(folder/'events.jsonl')==job['events_sha256']
        assert report['main_experiment_valid'] and report['requests_completed']==360
        qs=list(report['requests'].values()); end=report['makespan_s']
        steps=[e for e in events if e['event']=='step_started']
        step_ends={(e['worker'],e['step_id']):e['t'] for e in events if e['event']=='step_complete'}
        comp=[e for e in events if e['event']=='layer_compute_started']
        intervals=[e for e in events if e['event']=='storage_interval']
        byte_integral=math.fsum((e['t']-e['from'])*e['actual_Bps'] for e in intervals)
        assert abs(byte_integral-report['bytes_read'])/report['bytes_read']<1e-10
        assert abs(report['bytes_read']-205581189120)<1
        CHECKS['bandwidth_integrals']+=1
        row={k:job[k] for k in ['variant','seed','factor','rate','mode']}
        row.update(slo=st.mean(q['first_token_s']<=q['deadline_s'] for q in qs),
                   ttft_ms=st.mean(q['ttft_s'] for q in qs)*1000,
                   p95_ms=quantile([q['ttft_s'] for q in qs],.95)*1000,
                   central_wait_ms=st.mean(q['central_wait_s'] for q in qs)*1000,
                   engine_ttft_ms=st.mean(q['engine_ttft_s'] for q in qs)*1000,
                   makespan_ms=end*1000,last_arrival_ms=max(q['arrival_s'] for q in qs)*1000,
                   steps=len(steps),batch_mean=st.mean(e['num_requests'] for e in steps),
                   batch_max=max(e['num_requests'] for e in steps),
                   multi_fraction=st.mean(e['num_requests']>1 for e in steps),
                   tokens_mean=st.mean(sum(e['num_scheduled_tokens'].values()) for e in steps),
                   bytes_read=report['bytes_read'],average_GBps=report['bytes_read']/end/1e9,
                   saturated_fraction=sum(e['t']-e['from'] for e in intervals if e['actual_Bps']>=108e9)/end,
                   oversubscribed_fraction=sum(e['t']-e['from'] for e in intervals if e['demand_Bps']>120e9)/end)
        assert abs(row['ttft_ms']/1000-report['mean_ttft_s'])<1e-12
        assert row['slo']==report['profile_deadline_attainment']
        for cls in 'ABC':
            sub=[q for q in qs if q['class_id']==cls]
            assert len(sub)==120
            row[f'{cls}_slo']=st.mean(q['first_token_s']<=q['deadline_s'] for q in sub)
            row[f'{cls}_last_finish_ms']=max(q['first_token_s'] for q in sub)*1000
            for key in ['ttft_s','central_wait_s','engine_ttft_s']:
                row[f'{cls}_{key[:-2]}_ms']=st.mean(q[key] for q in sub)*1000
        for kind in ['compute','stall','idle']:
            row[f'{kind}_fraction']=sum(w[f'{kind}_s'] for w in report['workers'])/(32*end)
        assert abs(sum(row[f'{k}_fraction'] for k in ['compute','stall','idle'])-1)<1e-10
        for w in report['workers']:
            active=sum(step_ends[e['worker'],e['step_id']]-e['t'] for e in steps if e['worker']==w['id'])
            compute=sum(e['end_s']-e['t'] for e in comp if e['worker']==w['id'])
            assert abs(compute-w['compute_s'])<1e-9
            assert abs(active-compute-w['stall_s'])<1e-9
            assert abs(sum(w[k] for k in ['compute_s','stall_s','idle_s'])-end)<1e-9
            CHECKS['worker_time_checks']+=1
            workers.append({**{k:row[k] for k in ['variant','seed','factor','rate','mode']},
                            'worker':w['id'],'utilization':w['compute_s']/end,
                            'compute_s':w['compute_s'],'stall_s':w['stall_s'],'idle_s':w['idle_s']})
        for q in qs:
            assert abs(q['ttft_s']-q['central_wait_s']-q['engine_ttft_s'])<1e-10
            requests.append({**{k:row[k] for k in ['variant','seed','factor','rate','mode']},
                             **{k:q[k] for k in ['request_id','class_id','arrival_s','submitted_s',
                                                 'first_token_s','deadline_s','ttft_s','central_wait_s','engine_ttft_s']}})
        records.append(row);CHECKS['runs']+=1;CHECKS['requests']+=len(qs)
        if job['seed']==SEEDS[job['variant']][0]:
            selected[job['variant'],job['factor'],job['mode']]=(report,events,steps,step_ends,comp)
    write_csv('runs_24.csv',records);write_csv('requests_8640.csv',requests);write_csv('workers_768.csv',workers)
    shutil.copy2(RAW/'manifest.json',DATA/'manifest.json')
    summaries=[]
    for variant in SEEDS:
        for f,m in ORDER:
            rs=[r for r in records if (r['variant'],r['factor'],r['mode'])==(variant,f,m)]
            summary={'variant':variant,'factor':f,'mode':m,'seed_count':len(rs)}
            for key in records[0]:
                if key not in ['variant','seed','factor','rate','mode']:
                    summary[key]=st.median(r[key] for r in rs)
            summary['slo_min']=min(r['slo'] for r in rs);summary['slo_max']=max(r['slo'] for r in rs)
            summaries.append(summary)
    write_csv('summary.csv',summaries)
    return records,selected,summaries


def summary_plots(summary):
    index={(r['variant'],r['factor'],r['mode']):r for r in summary}
    fig,ax=plt.subplots(figsize=(13,6))
    xs=np.arange(6)
    for variant,shift,color in [('A',-.19,'#6A9FDA'),('B',.19,'#2C5F9E')]:
        rs=[index[variant,f,m] for f,m in ORDER]
        ys=np.array([r['slo']*100 for r in rs])
        ax.bar(xs+shift,ys,.36,label=f'C_{variant}',color=color)
        if variant=='A':
            ax.errorbar(xs+shift,ys,yerr=[ys-np.array([r['slo_min']*100 for r in rs]),
                         np.array([r['slo_max']*100 for r in rs])-ys],fmt='none',color='#333',capsize=3)
        for x,y in zip(xs+shift,ys):ax.text(x,y+2,f'{y:.1f}%',ha='center',fontsize=9)
    ax.set(xticks=xs,xticklabels=[label(f,m).replace(' · ','\n') for f,m in ORDER],ylim=(0,105),ylabel='请求自身期限达标率（%）')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend()
    ax.set_title('图1｜不同到达率与准入策略的 SLO；C_A 三种子中位数，C_B 单种子')
    fig.text(.5,.02,'误差线为 C_A 输入种子最小/最大值，不是置信区间；每个格点只运行一次原生路由。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));style.save(fig,FIG/'slo.png')
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for col,variant in enumerate(SEEDS):
        for mode in MODES:
            rs=[index[variant,f,mode] for f in FACTORS]
            axes[0,col].plot(FACTORS,[r['batch_mean'] for r in rs],marker='o',color=style.POLICY_COLOR[mode],label=LABEL[mode])
            axes[1,col].plot(FACTORS,[r['idle_fraction']*100 for r in rs],marker='o',color=style.POLICY_COLOR[mode],label=LABEL[mode])
        axes[0,col].set_title(f'C_{variant}：实际 batch 请求数');axes[0,col].set_ylabel('每 step 平均请求数')
        axes[1,col].set_title(f'C_{variant}：实例空闲占比');axes[1,col].set_ylabel('空闲 / 32实例总时长（%）')
        axes[1,col].set_xlabel('每类请求到达率倍率');axes[1,col].set_xticks(FACTORS)
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('组批与空闲：C_A 三种子中位数，C_B seed424；分母为各次自身 makespan')
    fig.tight_layout();style.save(fig,FIG/'batch_idle.png')


def trajectory_plots(selected,variant):
    horizon=max(selected[variant,f,m][0]['makespan_s'] for f,m in ORDER)
    fig,axes=plt.subplots(6,1,figsize=(16.5,18),sharex=True)
    for ax,(f,m) in zip(axes,ORDER):
        report,events,steps,ends,comp=selected[variant,f,m]
        first={w:min((e['t'] for e in steps if e['worker']==w),default=math.inf) for w in range(32)}
        order=sorted(range(32),key=lambda w:(first[w],w));pos={w:31-i for i,w in enumerate(order)}
        polys={k:[] for k in style.STATE_COLOR}
        def rect(kind,a,b,w):
            y=pos[w];polys[kind].append([(a*1000,y-.45),(b*1000,y-.45),(b*1000,y+.45),(a*1000,y+.45)])
        for w in range(32):rect('idle',0,horizon,w)
        starts={(e['worker'],e['step_id']):e for e in steps}
        for e in steps:rect('io',e['t'],ends[e['worker'],e['step_id']],e['worker'])
        for e in comp:
            cl={report['requests'][r]['class_id'] for r in starts[e['worker'],e['step_id']]['num_scheduled_tokens']}
            kind=next(iter(cl)) if len(cl)==1 else 'mixed'
            rect(kind,e['t'],e['end_s'],e['worker'])
        for kind in ['idle','io','A','B','C','mixed']:
            ax.add_collection(PolyCollection(polys[kind],facecolors=style.STATE_COLOR[kind],edgecolors='none',antialiased=False))
        end=report['makespan_s']*1000
        ax.axvline(end,color='#333',ls='--',lw=.8)
        ax.set(xlim=(0,horizon*1000),ylim=(-.6,31.6),ylabel=label(f,m),yticks=[pos[w] for w in order],yticklabels=order)
        ax.tick_params(axis='y',labelsize=6);ax.grid(axis='x',alpha=.14)
        ax.set_title(f'完成 {end:.1f} ms；虚线后为统一横轴补灰',loc='right',fontsize=9)
    fig.legend(handles=[Patch(color=style.STATE_COLOR[k],label=t) for k,t in
                       [('A','A计算'),('B','B计算'),('C','C计算'),('mixed','混合计算'),('io','IO等待'),('idle','实例无请求')]],
               loc='upper center',ncol=6,bbox_to_anchor=(.5,.97))
    fig.suptitle(f'图2｜C_{variant} 完整执行时间线 · seed{SEEDS[variant][0]} · 单次原生路由',y=.99)
    axes[-1].set_xlabel('仿真时间（ms）；各行独立按首次 step 启动时间排序，左侧保留实例编号')
    fig.tight_layout(rect=(0,0,1,.95));style.save(fig,FIG/f'C{variant}_timeline.png')
    values=[[w['compute_s']/selected[variant,f,m][0]['makespan_s']*100 for w in selected[variant,f,m][0]['workers']] for f,m in ORDER]
    fig,(left,right)=plt.subplots(1,2,figsize=(16,6.3),gridspec_kw={'width_ratios':[1,1.7]})
    bp=left.boxplot(values,tick_labels=[label(f,m).replace(' · ','\n') for f,m in ORDER],patch_artist=True,medianprops={'color':'black'})
    for patch in bp['boxes']:patch.set_facecolor(style.UTIL_BOX_COLOR)
    left.scatter(range(1,7),[st.mean(v) for v in values],color=style.UTIL_MEAN_COLOR,marker='D',label='均值')
    left.set(ylim=(0,100),ylabel='纯计算 / 自身 makespan（%）');left.grid(axis='y',alpha=.2);left.legend()
    im=right.imshow(values,aspect='auto',cmap='viridis',vmin=0,vmax=100)
    right.set(yticks=range(6),yticklabels=[label(f,m) for f,m in ORDER],xticks=[0,7,15,23,31],xlabel='原实例编号')
    fig.colorbar(im,ax=right,label='计算占比（%）')
    fig.suptitle(f'图3｜C_{variant} 计算利用率 · seed{SEEDS[variant][0]} · IO等待不计入计算')
    fig.tight_layout();style.save(fig,FIG/f'C{variant}_utilization.png')
    fig,axes=plt.subplots(6,1,figsize=(14.6,16.2),sharex=True,sharey=True)
    rows=[];maximum=120.;curves=[];bin_s=.01;n=math.ceil(horizon/bin_s)
    for ax,(f,m) in zip(axes,ORDER):
        report,events,*_=selected[variant,f,m];actual=np.zeros(n);demand=np.zeros(n)
        for e in events:
            if e['event']!='storage_interval':continue
            for j in range(int(e['from']/bin_s),min(n,math.ceil(e['t']/bin_s))):
                dt=max(0,min(e['t'],(j+1)*bin_s)-max(e['from'],j*bin_s))
                actual[j]+=dt*e['actual_Bps'];demand[j]+=dt*e['demand_Bps']
        assert abs(actual.sum()-report['bytes_read'])/report['bytes_read']<1e-10
        actual/=bin_s*1e9;demand/=bin_s*1e9;xs=(np.arange(n)+.5)*bin_s*1000
        maximum=max(maximum,float(demand.max()))
        curves.append((ax,xs,actual,demand,f,m))
        for t,a,d in zip(xs,actual,demand):rows.append(dict(factor=f,mode=m,time_ms=t,actual_GBps=a,demand_GBps=d))
    for ax,xs,actual,demand,f,m in curves:
        ax.plot(xs,np.maximum(demand,.5),color='#DD8452',lw=.8,label='申请需求')
        ax.plot(xs,np.maximum(actual,.5),color='#4C72B0',lw=1,label='实际速率')
        ax.axhline(120,color='black',ls='--',lw=.9)
        ax.set(yscale='log',ylim=(.5,maximum*1.3),xlim=(0,horizon*1000),ylabel=label(f,m)+'\nGB/s')
        ax.grid(alpha=.18,which='both')
    axes[0].legend(loc='lower right');axes[-1].set_xlabel('仿真时间（ms）；10ms区间积分；零速率仅在对数图显示为0.5，CSV保留0')
    fig.suptitle(f'图4｜C_{variant} 存储带宽 · seed{SEEDS[variant][0]} · 容量120 GB/s')
    fig.tight_layout(rect=(0,0,1,.975));style.save(fig,FIG/f'C{variant}_bandwidth.png')
    write_csv(f'C{variant}_bandwidth_10ms.csv',rows)


def cdf_plots(selected,variant):
    styles={1:'-',1.5:'--',2:':'};rows=[]
    for by_class in [False,True]:
        fig,axes=plt.subplots(1,3 if by_class else 1,figsize=(16,5.6) if by_class else (11,6.5),squeeze=False,sharey=True)
        for ax,cls in zip(axes.flat,'ABC' if by_class else ['all']):
            for f,m in ORDER:
                report=selected[variant,f,m][0]
                qs=[q for q in report['requests'].values() if cls=='all' or q['class_id']==cls]
                xs=sorted(q['ttft_s']*1000 for q in qs);ys=np.arange(1,len(xs)+1)/len(xs)
                assert all(a<=b for a,b in zip(xs,xs[1:])) and ys[-1]==1
                CHECKS['cdf_checks']+=1
                title=label(f,m) if by_class else label(f,m)+f' p50={quantile(xs,.5):.0f}/p95={quantile(xs,.95):.0f}ms'
                ax.step([0]+xs,[0]+list(ys),where='post',color=style.POLICY_COLOR[m],ls=styles[f],label=title)
                for rank,(x,y) in enumerate(zip(xs,ys),1):rows.append(dict(factor=f,mode=m,class_id=cls,rank=rank,ttft_ms=x,cdf=y))
            if by_class:
                widths={(q['deadline_s']-q['arrival_s'])*1000 for q in qs}
                width=st.mean(widths);assert max(widths)-min(widths)<1e-8
                ax.axvline(width,color='#444',ls=':',lw=1)
                ax.set_title(f'{cls}类；期限宽度 {width:.1f} ms')
            ax.set(xlim=(0,None),ylim=(0,1.02),xlabel='TTFT（ms，包含中央等待）');ax.grid(alpha=.2)
        axes[0,0].set_ylabel('经验CDF')
        if by_class:
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.93))
        else:axes[0,0].legend(loc='lower right',fontsize=8)
        fig.suptitle(f'图5｜C_{variant} '+('分类' if by_class else '总体')+f' TTFT分布 · seed{SEEDS[variant][0]}')
        fig.tight_layout(rect=(0,0,1,.80 if by_class else .95))
        style.save(fig,FIG/(f'C{variant}_cdf'+('_class.png' if by_class else '.png')))
    write_csv(f'C{variant}_cdf.csv',rows)


def report_text(records,summary):
    ix={(r['variant'],r['factor'],r['mode']):r for r in summary}
    text='''# 多级速率 TFS 对照（2026-10-20）

| 项目 | 约定 |
|---|---|
| 原始数据 | 本次新运行24次，不复用旧报告数值 |
| 负载 | 每份A/B/C各120条，共360条；C_A种子416/417/418，C_B种子424 |
| 到达率 | 每类98、147、196条/秒；A仍每簇16条，B/C逐条泊松到达；总名义到达率294、441、588条/秒 |
| 受控变量 | 同种子只改lam_each；到达时间按倍率压缩，形状/T0/期限宽度不变；deadline=新arrival+4T0 |
| 资源 | 32独立vLLM调度实例/模拟计算槽位；8层；120GB/s；8192 token/step；max_num_seqs=8 |
| 策略 | immediate直接提交、tfs请求级准入；真实Dynamo原生选路，逐层读算仿真 |
| 重复与汇总 | 各输入×各到达率×各策略只跑一次；C_A取三个输入种子中位数，C_B为单次；指标独立取中位数 |
| 固定轨迹 | C_A固定seed416，C_B固定seed424，展示全部三档与两种策略，不按成绩选图 |
| TTFT与SLO | TTFT从arrival到首token，包含中央等待；逐请求以TTFT≤4T0判断达标 |
| 配额 | 原P75与120GB/s参考带宽；到首token释放；aging关闭、反馈无延迟 |
| 来源 | SA提交3cd826961bb53b05d09eb7f6e6949e4222230605；运行实现SHA见manifest.json |

## 0. 总判决

以下是本轮实际结果。提高到达率同时改变队列、路由和组批，不能把所有变化归因于batch。

'''
    for v in SEEDS:
        a,b=ix[v,1,'tfs'],ix[v,2,'tfs']
        text+=f"- **C_{v}，TFS 1→2倍**：平均batch {a['batch_mean']:.3f}→{b['batch_mean']:.3f}；空闲 {a['idle_fraction']:.1%}→{b['idle_fraction']:.1%}；平均TTFT {a['ttft_ms']:.1f}→{b['ttft_ms']:.1f}ms；总SLO {a['slo']:.1%}→{b['slo']:.1%}。\n"
    for v in SEEDS:
        for f in FACTORS:
            a,b=ix[v,f,'immediate'],ix[v,f,'tfs']
            if b['A_ttft_ms']>a['A_ttft_ms']:
                text+=f"- **分类取舍**：C_{v}、{f:g}倍到达率，A平均TTFT直接提交 {a['A_ttft_ms']:.1f}ms、TFS {b['A_ttft_ms']:.1f}ms；不能将总体时延改善表述为每类都改善。\n"
    pairs={(r['variant'],r['seed'],r['factor'],r['mode']):r for r in records}
    for v,seeds in SEEDS.items():
        for seed in seeds:
            for f in FACTORS:
                a,b=pairs[v,seed,f,'immediate'],pairs[v,seed,f,'tfs']
                if b['slo']<a['slo']:
                    text+=f"- **保留反例**：C_{v} seed{seed}、{f:g}倍到达率，直接提交SLO {a['slo']:.1%}，TFS {b['slo']:.1%}；总体中位数优势不代表每个seed都提高达标率。\n"
    text+='\n## 1. 汇总表与SLO柱状图\n\n'
    for v in SEEDS:
        text+=f'### C_{v}\n\n| 倍率 | 策略 | 总SLO | A/B/C SLO | 平均TTFT(ms) | 中央等待(ms) | makespan(ms) |\n|---|---|---:|---:|---:|---:|---:|\n'
        for f,m in ORDER:
            r=ix[v,f,m];cl='/'.join(f'{r[c+"_slo"]:.1%}' for c in 'ABC')
            text+=f"| {f:g} | {LABEL[m]} | {r['slo']:.1%} | {cl} | {r['ttft_ms']:.1f} | {r['central_wait_ms']:.1f} | {r['makespan_ms']:.1f} |\n"
        text+='\n'
    text+='![SLO汇总](figure/slo.png)\n\n**读法**：C_A为三个输入种子中位数，误差线为种子范围；C_B只有一个输入种子。分类中位数不能加权还原总中位数。原生路由平局未固定，同一倍率两臂的差异包含路由变化。\n\n'
    text+='## 2. 完整执行时间线与真实组批\n\n![组批与空闲](figure/batch_idle.png)\n\n**读法**：请求数增加与token数增加不等价。每档仍为360请求，倍率改变的是输入速度。空闲分母为自身makespan，不含为统一横轴补出的末尾灰色。\n\n'
    text+='| 工况 | 倍率 | 策略 | 平均batch | 多请求step | 平均token/step |\n|---|---:|---|---:|---:|---:|\n'
    for r in summary:text+=f"| C_{r['variant']} | {r['factor']:g} | {LABEL[r['mode']]} | {r['batch_mean']:.3f} | {r['multi_fraction']:.1%} | {r['tokens_mean']:.1f} |\n"
    for v in SEEDS:text+=f'\n![C_{v}完整时间线](figure/C{v}_timeline.png)\n\n**读法**：固定seed{SEEDS[v][0]}；每个面板独立按实例首次启动排序。蓝/橙/绿是A/B/C计算，紫色是混批，红色是当前step等待本层读取，灰色为实例无请求。竖虚线后全部请求已完成，为共享横轴补灰。预取与计算重叠时显示计算色。\n'
    text+='\n## 3. 模拟计算利用率与空闲\n\n| 工况 | 倍率 | 策略 | 计算 | IO等待 | 空闲 |\n|---|---:|---|---:|---:|---:|\n'
    for r in summary:text+=f"| C_{r['variant']} | {r['factor']:g} | {LABEL[r['mode']]} | {r['compute_fraction']:.1%} | {r['stall_fraction']:.1%} | {r['idle_fraction']:.1%} |\n"
    for v in SEEDS:text+=f'\n![C_{v}利用率](figure/C{v}_utilization.png)\n\n**读法**：固定seed{SEEDS[v][0]}，纯计算时间除以该次makespan；箱线与热图来自同一组32实例数据。并非硬件实测利用率。\n'
    text+='\n## 4. 存储带宽时序\n\n| 工况 | 倍率 | 策略 | 平均实际GB/s | 实际≥108GB/s占比 | 需求超订占比 |\n|---|---:|---|---:|---:|---:|\n'
    for r in summary:text+=f"| C_{r['variant']} | {r['factor']:g} | {LABEL[r['mode']]} | {r['average_GBps']:.1f} | {r['saturated_fraction']:.1%} | {r['oversubscribed_fraction']:.1%} |\n"
    for v in SEEDS:text+=f'\n![C_{v}带宽](figure/C{v}_bandwidth.png)\n\n**读法**：固定seed{SEEDS[v][0]}；10ms桶按真实区间求交积分，橙色申请、蓝色实际、虚线120GB/s；对数图把0显示在0.5底线，CSV保留0。所有格读取总量相同；各策略全程平均带宽分母不同。申请率不是物理限速，也不是TFS配额计数。\n'
    text+='\n## 5. TTFT累积分布\n'
    for v in SEEDS:
        text+=f'\n![C_{v}总体CDF](figure/C{v}_cdf.png)\n\n![C_{v}分类CDF](figure/C{v}_cdf_class.png)\n\n**读法**：固定seed{SEEDS[v][0]}，每个总体360请求、每类120请求。实线/虚线/点线分别是1/1.5/2倍。经验CDF为rank/n，分位数采用nearest-rank；TTFT包含中央等待。分类竖线为各请求类自身期限宽度，总SLO按请求一一比较，不从总体边际CDF推导。\n'
    text+='\n### TFS 的 A 类等待与输入结束后的收尾\n\n| 工况 | 倍率 | A中央等待(ms) | A下发后耗时(ms) | A总TTFT(ms) | 最后到达(ms) | 全部完成(ms) |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for v in SEEDS:
        for f in FACTORS:
            r=ix[v,f,'tfs']
            text+=f"| C_{v} | {f:g} | {r['A_central_wait_ms']:.1f} | {r['A_engine_ttft_ms']:.1f} | {r['A_ttft_ms']:.1f} | {r['last_arrival_ms']:.1f} | {r['makespan_ms']:.1f} |\n"
    text+='\n各指标分别取种子中位数，A中央等待中位数与下发后耗时中位数之和未必等于A总TTFT中位数。输入结束更早不代表全部计算和读取更早完成。\n'
    text+='''
## 6. 机制解释与前序实验衔接

- 本轮提高每类到达率，保持每类请求数120不变。输入更密集，输入持续时间更短，并非增加总请求数。
- 更密集的输入可以增加实例内组批候选，同时也增加中央等待、实例内排队和混批共同等待；批次变大不保证TTFT下降。
- 原生Dynamo每格只运行一次，同代价路由存在随机性。不要将本轮1倍数字与旧三次路由中位数直接相减作为算法收益。
- 本轮比较的是当前TFS与直接提交，没有加入前一轮“空泡额外放行A”的例外，也没有修改路由或组批规则。

## 7. 计算模型与限制

沿用SA synthetic-v1，8层读取与计算重叠。512-token效率参数未经过硬件校准；每请求只生成一个合成token。真实代码边界为Dynamo SelectionService、vLLM Scheduler/KV管理，未执行真实ModelRunner与设备kernel。C_A只有三输入种子，C_B只有一输入种子，每格一次路由；用于观察方向与反例，不作显著性或真实部署结论。

## 8. 逐seed完整附表

| 工况/seed | 倍率 | 策略 | 总SLO | 平均TTFT(ms) | 平均batch | 空闲 | IO等待 |
|---|---:|---|---:|---:|---:|---:|---:|
'''
    for r in records:text+=f"| C_{r['variant']}/{r['seed']} | {r['factor']:g} | {LABEL[r['mode']]} | {r['slo']:.1%} | {r['ttft_ms']:.1f} | {r['batch_mean']:.3f} | {r['idle_fraction']:.1%} | {r['stall_fraction']:.1%} |\n"
    text+='''
## 9. 验证、数据与复现

本次实际运行24次，全部通过原始`check_sa_results.py`验收。另复算8640请求、768实例时间账本、24条实际带宽字节积分、真实step与计算/IO区间、CDF单调性和终点概率。具体检查及交付文件SHA见[validation.json](report_data/validation.json)。

- [24次运行指标](report_data/runs_24.csv)、[汇总中位数](report_data/summary.csv)、[8640条请求](report_data/requests_8640.csv)、[768实例时间数据](report_data/workers_768.csv)。
- [输入与运行清单](report_data/manifest.json)、[实验设计](plan.json)、[实验执行脚本](run_sweep.py)、[报告生成脚本](build_report.py)。
- 完整原始资产在工作区`temp/rate_sweep_20261020/`，包括12份输入、24份report.json/events.jsonl及运行与验收日志；分享包不包含大型原始事件文件。manifest保留原始路径和SHA。
- 重新生成报告：使用DV的`.venv/bin/python`执行本目录`build_report.py`。需要保留上述原始资产。
- 重新运行：将`run_sweep.py`和`plan.json`放入工作区`temp/rate_sweep_20261020/`，使用DV的`.venv/bin/python`执行。已有运行目录会报错，防止覆盖记录。需要本地固定SA/DV源码与环境。

## 10. 简要解读

提高到达率后，先看batch与灰色是否变化，再检查TTFT、分类达标率与红色IO等待。灰色减少可能意味着计算更连续，也可能意味着等待或积压更多。应使用本轮同倍率直接提交对照评估TFS，并保留各seed结果。
'''
    (OUT/'多级速率tfs对照-20261020.md').write_text(text)


def main():
    DATA.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    records,selected,summary=collect();summary_plots(summary)
    for variant in SEEDS:trajectory_plots(selected,variant);cdf_plots(selected,variant)
    shutil.copy2(RAW/'run_sweep.py',OUT/'run_sweep.py')
    report_text(records,summary)
    report=OUT/'多级速率tfs对照-20261020.md'
    checks={**CHECKS,'source_runtime_sha256':json.loads((RAW/'manifest.json').read_text())[0]['implementation_sha256'],
            'simulation_rerun':True,'original_checker_runs':24,'arrival_scaling_and_deadline_widths':'passed',
            'aggregation':'median across input seeds; one native routing run per cell',
            'files':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='validation.json' and '__pycache__' not in p.parts}}
    (DATA/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):assert (OUT/target).exists(),target
    package=OUT/'到达率加速实验报告-20261020.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts:z.write(p,p.relative_to(OUT))
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for name in z.namelist():assert z.read(name)==(OUT/name).read_bytes()
    print(json.dumps(CHECKS),flush=True)


if __name__=='__main__':main()
