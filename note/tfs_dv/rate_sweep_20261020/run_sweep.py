"""按原 SA 生成器改变到达率，调用未修改的 DV 执行器与完整验收。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PKG = ROOT / 'data/dynamo_vllm_simulation/dynamo_vllm_cpu_sim'
SA = ROOT / 'data/status_aware_network'
OLD = PKG / 'results/sa_alignment_20261019'
PLAN = json.loads((HERE/'plan.json').read_text())
INPUTS = [('A', s) for s in PLAN['inputs']['C_A']] + [('B', s) for s in PLAN['inputs']['C_B']]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    sys.path.insert(0, str(SA))
    from sim.experiments import e26f_third as e26f
    source = json.loads((OLD/'source/source.json').read_text())
    for name, digest in source['source_sha256'].items():
        assert sha(SA/name) == digest, name
    traces = HERE/'traces'
    traces.mkdir(exist_ok=True)
    for variant, seed in INPUTS:
        frozen = json.loads((OLD/f'source/D1f{variant}-s{seed}.json').read_text())
        for factor, rate in zip(PLAN['multipliers'], PLAN['arrival_rates_each_per_s']):
            specs, centers = e26f.d1f_specs(16, seed, e26f.C_CLASSES[variant], n=120, lam_each=rate)
            rows = [dict(id=f'r{s.rid:05d}', sa_rid=s.rid, class_id=s.class_id,
                         arrival_s=float(s.arrival_s), h_tokens=s.h_tokens, u_tokens=s.u_tokens,
                         T0_s=float(s.T0_s), deadline_s=float(s.deadline_s)) for s in specs]
            assert len(rows) == len(frozen) == 360
            for a,b in zip(rows, frozen):
                for key in ['id','sa_rid','class_id','h_tokens','u_tokens','T0_s']:
                    assert a[key] == b[key], (variant, seed, factor, key)
                assert abs(a['arrival_s'] - b['arrival_s']/factor) < 1e-12
                assert abs(a['deadline_s'] - a['arrival_s'] - 4*a['T0_s']) < 1e-12
            if factor == 1:
                assert rows == frozen
            (traces/f'C{variant}-s{seed}-rate{rate}.json').write_text(json.dumps(rows)+ '\n')
    print('12 traces validated: paired shapes, scaled arrivals, unchanged deadline widths.', flush=True)


def run():
    with (HERE/'generate.log').open('w') as log:
        subprocess.run([sys.executable, str(__file__), 'generate'], stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = []
    for variant, seed in INPUTS:
        base = json.loads((OLD/f'runs/D1f{variant}-s{seed}/cap8-tfs-r0/config.json').read_text())
        for factor, rate in zip(PLAN['multipliers'], PLAN['arrival_rates_each_per_s']):
            trace = HERE/f'traces/C{variant}-s{seed}-rate{rate}.json'
            for mode in PLAN['admission_modes']:
                name = f'C{variant}-s{seed}-rate{rate}-{mode}'
                folder = HERE/'runs'/name
                folder.mkdir(parents=True, exist_ok=False)
                cfg = copy.deepcopy(base)
                cfg['admission']['mode'] = mode
                cfg['source_trace'] = str(trace)
                config = folder/'config.json'
                config.write_text(json.dumps(cfg, indent=2)+'\n')
                with (folder/'run.log').open('w') as log:
                    subprocess.run([sys.executable, '-m', 'sim.sa_aligned', '--config', str(config),
                                    '--trace', str(trace), '--output', str(folder)], cwd=PKG,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                with (folder/'check.log').open('w') as log:
                    subprocess.run([sys.executable, str(PKG/'scripts/check_sa_results.py'), str(folder)],
                                   cwd=PKG, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads((folder/'report.json').read_text())
                assert report['requests_completed'] == 360 and abs(report['bytes_read']-205581189120)<1
                record = dict(name=name, variant=variant, seed=seed, factor=factor, rate=rate, mode=mode,
                              trace=str(trace), folder=str(folder), config_sha256=sha(config),
                              trace_sha256=sha(trace), report_sha256=sha(folder/'report.json'),
                              events_sha256=sha(folder/'events.jsonl'),
                              implementation_sha256=report['implementation_sha256'])
                manifest.append(record)
                (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                print(f"{len(manifest)}/24 {name}: SLO={report['profile_deadline_attainment']:.4f}, "
                      f"TTFT={report['mean_ttft_s']*1000:.2f}ms, "
                      f"batch={report['actual_batch']['requests']['mean']:.3f}", flush=True)
    assert len(manifest) == 24
    assert len({m['implementation_sha256'] for m in manifest}) == 1
    print('24 runs and original check_sa_results validations completed.', flush=True)


if __name__ == '__main__':
    generate() if sys.argv[1:] == ['generate'] else run()
