"""从冻结的 96 次运行生成 E26f 风格报告；不执行仿真。"""

import csv
import hashlib
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / "data/dynamo_vllm_simulation"
PKG = REPO / "dynamo_vllm_cpu_sim"
RAW = PKG / "results/sa_alignment_20261019"
ARCHIVE = PKG / "results/sa_alignment_summary_20261019"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figure"
DATA = OUT / "report_data"
MODES = ("immediate", "slack_only", "quota_fcfs", "tfs")
LABEL = {"immediate": "直接提交", "slack_only": "仅等待预算排序",
         "quota_fcfs": "配额+先到先服务", "tfs": "TFS"}
POLICY_COLOR = dict(zip(MODES, ("#888888", "#C44E52", "#DD8452", "#55A868")))
STATE_COLOR = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868",
               "mixed": "#9974A5", "io": "#C44E52", "idle": "#C7C7C7"}
SEEDS = {"A": (416, 417, 418), "B": (424,)}
FIGURES = []
CHECKS = {"verified_raw_files": 0, "reports": 0, "requests": 0,
          "worker_time_checks": 0, "selected_event_runs": 0,
          "timeline_compute_checks": 0, "bandwidth_integral_checks": 0}


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (DATA / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows, variant, cap, mode, field):
    # 先对同一输入种子的 3 次路由重复取中位数，再对输入种子取中位数。
    values = seed_values(rows, variant, cap, mode, field)
    return stats.median(values)


def seed_values(rows, variant, cap, mode, field):
    return [stats.median(r[field] for r in rows if r["variant"] == variant
                         and r["seed"] == seed and r["cap"] == cap
                         and r["mode"] == mode) for seed in SEEDS[variant]]


def collect():
    for row in csv_rows(ARCHIVE / "raw_hashes.csv"):
        assert sha(RAW / row["cell"] / row["file"]) == row["sha256"], row
        CHECKS["verified_raw_files"] += 1
    manifest = json.loads((RAW / "manifest.json").read_text())
    records, reports = [], {}
    for job in manifest:
        path = RAW / "runs" / job["cell"] / f"cap{job['cap']}-{job['mode']}-r{job['repeat']}"
        report = json.loads((path / "report.json").read_text())
        identity = {k: job[k] for k in ("variant", "seed", "cap", "mode", "repeat")}
        key = tuple(identity.values())
        reports[key] = report
        assert report["status"] == "completed" and report["main_experiment_valid"]
        assert report["implementation_sha256"] == job["implementation_sha256"]
        assert len(report["requests"]) == report["requests_completed"] == 360
        reqs = list(report["requests"].values())
        for req in reqs:
            assert abs(req["ttft_s"] - req["first_token_s"] + req["arrival_s"]) < 1e-10
            assert abs(req["ttft_s"] - req["central_wait_s"] - req["engine_ttft_s"]) < 1e-10
            assert req["submitted_s"] >= req["arrival_s"] - 1e-12
        slo = stats.mean(q["first_token_s"] <= q["deadline_s"] for q in reqs)
        ttft = stats.mean(q["ttft_s"] for q in reqs) * 1000
        assert abs(slo - report["profile_deadline_attainment"]) < 1e-12
        assert abs(ttft / 1000 - report["mean_ttft_s"]) < 1e-12
        workers, end = report["workers"], report["makespan_s"]
        assert [w["id"] for w in workers] == list(range(32))
        for w in workers:
            assert abs(w["compute_s"] + w["stall_s"] + w["idle_s"] - end) < 1e-9
            CHECKS["worker_time_checks"] += 1
        util = np.array([w["compute_s"] / end * 100 for w in workers])
        histogram = report["actual_batch"]["request_count_histogram"]
        steps = sum(histogram.values())
        multi = sum(n for k, n in histogram.items() if int(k) > 1)
        row = {**identity, "slo": slo, "ttft_ms": ttft,
               "central_wait_ms": stats.mean(q["central_wait_s"] for q in reqs) * 1000,
               "makespan_ms": end * 1000, "steps": steps, "multi_steps": multi,
               "multi_fraction": multi / steps, "batch_mean": report["actual_batch"]["requests"]["mean"],
               "batch_max": max(map(int, histogram)), "bytes_read": report["bytes_read"],
               "util_min": float(util.min()), "util_p50": float(np.median(util)),
               "util_p90": float(np.quantile(util, .9)), "util_max": float(util.max()),
               "util_mean": float(util.mean()), "util_cv": float(util.std() / util.mean()),
               "busy_mean": stats.mean((w["compute_s"] + w["stall_s"]) / end * 100 for w in workers),
               "io_fraction": sum(w["stall_s"] for w in workers) / (32 * end) * 100}
        for cl in "ABC":
            subset = [q for q in reqs if q["class_id"] == cl]
            assert len(subset) == 120
            row[f"{cl}_slo"] = stats.mean(q["first_token_s"] <= q["deadline_s"] for q in subset)
            row[f"{cl}_ttft_ms"] = stats.mean(q["ttft_s"] for q in subset) * 1000
        records.append(row)
        CHECKS["reports"] += 1
        CHECKS["requests"] += len(reqs)
    assert len(records) == 96
    by_run = {(r["variant"], int(r["seed"]), int(r["cap"]), r["mode"], int(r["repeat"])): r
              for r in csv_rows(ARCHIVE / "runs.csv")}
    by_class = {(r["variant"], int(r["seed"]), int(r["cap"]), r["mode"], int(r["repeat"]), r["class"]): r
                for r in csv_rows(ARCHIVE / "classes.csv")}
    for row in records:
        key = tuple(row[k] for k in ("variant", "seed", "cap", "mode", "repeat"))
        for field in ("slo", "ttft_ms", "central_wait_ms", "batch_mean"):
            assert abs(row[field] - float(by_run[key][field])) < 1e-8
        for cl in "ABC":
            for field in ("slo", "ttft_ms"):
                assert abs(row[f"{cl}_{field}"] - float(by_class[key + (cl,)][field])) < 1e-8
    reference = {(r["variant"], int(r["cap"]), r["mode"]): r
                 for r in csv_rows(ARCHIVE / "summary.csv")}
    for variant in SEEDS:
        for cap in (1, 8):
            for mode in MODES:
                for field in ("slo", "ttft_ms", "central_wait_ms", "batch_mean"):
                    assert abs(aggregate(records, variant, cap, mode, field)
                               - float(reference[variant, cap, mode][field])) < 1e-8
    write_csv("run_metrics_96.csv", records)
    return records, reports


def save(fig, name):
    path = FIG / f"e26f_dv_{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    FIGURES.append(path)


def bar(records):
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    x, width = np.arange(4), .35
    for i, variant in enumerate(SEEDS):
        values = [aggregate(records, variant, 8, mode, "slo") * 100 for mode in MODES]
        ranges = [np.array(seed_values(records, variant, 8, mode, "slo")) * 100 for mode in MODES]
        xs = x + (i - .5) * width
        color = ("#6A9FDA", "#2C5F9E")[i]
        ax.bar(xs, values, width, color=color, label=f"C_{variant}", zorder=2)
        ax.errorbar(xs, values, yerr=[[v - r.min() for v, r in zip(values, ranges)],
                                     [r.max() - v for v, r in zip(values, ranges)]],
                    fmt="none", ecolor="#333333", capsize=4, lw=1, zorder=3)
        for pos, value, mode, spread in zip(xs, values, MODES, ranges):
            ax.text(pos, max(value, spread.max()) + 2, f"{value:.1f}%", ha="center", fontsize=10, color=color)
            label = "\n".join(f"{cl} {aggregate(records, variant, 8, mode, cl + '_slo'):.0%}" for cl in "ABC")
            ax.text(pos, value * .47, label, color="white", ha="center", va="center", fontsize=9)
    ax.set(xticks=x, xticklabels=[LABEL[m] for m in MODES], ylim=(0, 102),
           ylabel="SLO 满足率（%）", title="图1｜E26f-DV 汇总：C_A / C_B × 四策略（max_num_seqs=8）")
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    fig.text(.5, .025, "柱高=先取路由重复中位数，再取输入种子中位数；线段=输入种子间范围，非置信区间。\n"
             "柱内白字=A/B/C 各类 SLO 独立中位数；C_B 仅一个输入种子。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, 1))
    save(fig, "slo_bar")


def selected(reports, variant):
    seed = SEEDS[variant][0]
    result = {}
    for mode in MODES:
        report = reports[variant, seed, 8, mode, 0]
        path = RAW / "runs" / f"D1f{variant}-s{seed}" / f"cap8-{mode}-r0" / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        starts = {(e["worker"], e["step_id"]): e for e in events if e["event"] == "step_started"}
        ends = {(e["worker"], e["step_id"]): e["t"] for e in events if e["event"] == "step_complete"}
        comp = [e for e in events if e["event"] == "layer_compute_started"]
        histogram = defaultdict(int)
        for e in starts.values():
            histogram[str(e["num_requests"])] += 1
        assert dict(histogram) == report["actual_batch"]["request_count_histogram"]
        for worker in report["workers"]:
            active = sum(ends[k] - e["t"] for k, e in starts.items() if k[0] == worker["id"])
            compute = sum(e["end_s"] - e["t"] for e in comp if e["worker"] == worker["id"])
            assert abs(compute - worker["compute_s"]) < 1e-9
            assert abs(active - compute - worker["stall_s"]) < 1e-9
            CHECKS["timeline_compute_checks"] += 1
        result[mode] = report, events, starts, ends, comp
        CHECKS["selected_event_runs"] += 1
    return result


def timeline(data, variant, zoom=False):
    horizon = .2 if zoom else max(value[0]["makespan_s"] for value in data.values())
    fig, axes = plt.subplots(4, 1, figsize=(16.5, 13.5), sharex=True)
    fig.subplots_adjust(left=.11, right=.99, bottom=.055, top=.91, hspace=.23)
    suffix = "前 200 ms 放大" if zoom else "full-sort-ms"
    fig.suptitle(f"图2｜C_{variant} 各策略执行时间线 · {suffix} · seed={SEEDS[variant][0]}, repeat=0", y=.985)
    fig.legend(handles=[Patch(color=STATE_COLOR[k], label=t) for k, t in
                        (("A", "A 计算"), ("B", "B 计算"), ("C", "C 计算（零读）"),
                         ("mixed", "混合计算"), ("io", "IO 等待"), ("idle", "无请求"))],
               ncol=6, loc="upper center", bbox_to_anchor=(.5, .962), frameon=False)
    for ax, mode in zip(axes, MODES):
        report, _, starts, ends, comp = data[mode]
        first = {wid: min(e["t"] for key, e in starts.items() if key[0] == wid) for wid in range(32)}
        order = sorted(first, key=lambda wid: (first[wid], wid))
        pos = {wid: 31 - i for i, wid in enumerate(order)}
        polys = {key: [] for key in STATE_COLOR}

        def rect(kind, start, end, y):
            start, end = max(0, start) * 1000, min(horizon, end) * 1000
            if end > start:
                polys[kind].append([(start, y - .45), (end, y - .45), (end, y + .45), (start, y + .45)])

        for wid in range(32):
            rect("idle", 0, horizon, pos[wid])
        for key, event in starts.items():
            rect("io", event["t"], ends[key], pos[key[0]])
        for event in comp:
            key = event["worker"], event["step_id"]
            classes = {report["requests"][rid]["class_id"] for rid in starts[key]["num_scheduled_tokens"]}
            kind = next(iter(classes)) if len(classes) == 1 else "mixed"
            rect(kind, event["t"], event["end_s"], pos[event["worker"]])
        for kind in ("idle", "io", "A", "B", "C", "mixed"):
            ax.add_collection(PolyCollection(polys[kind], facecolors=STATE_COLOR[kind],
                                            edgecolors="none", antialiased=False))
        end = report["makespan_s"]
        io = sum(w["stall_s"] for w in report["workers"]) / (32 * end)
        calc = sum(w["compute_s"] for w in report["workers"]) / (32 * end)
        ax.set(xlim=(0, horizon * 1000), ylim=(-.6, 31.6),
               ylabel=f"{LABEL[mode]}\n计算 {calc:.1%}\nIO 等待 {io:.1%}")
        ax.set_yticks([pos[wid] for wid in order], [str(wid) for wid in order], fontsize=6)
        ax.grid(axis="x", alpha=.14)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("仿真时间（ms）；每个策略独立按首次 step 启动时间从上到下排序，左侧数字为原 NPU 编号")
    save(fig, f"C{variant}_{'startup-sort-ms' if zoom else 'full-sort-ms'}")


def utilization(data, variant):
    values = [[w["compute_s"] / data[mode][0]["makespan_s"] * 100
               for w in data[mode][0]["workers"]] for mode in MODES]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.3), gridspec_kw={"width_ratios": [1, 1.6]})
    bp = left.boxplot(values, tick_labels=[LABEL[m].replace("+", "+\n") for m in MODES],
                      patch_artist=True, medianprops={"color": "black"}, widths=.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9DC3E6")
        patch.set_alpha(.8)
    left.scatter(range(1, 5), [np.mean(v) for v in values], color="#C44E52", marker="D", label="均值", zorder=3)
    left.set(ylim=(0, 100), ylabel="计算时间 / 本次 makespan（%）", title="32 NPU 纯计算利用率分布")
    left.tick_params(axis="x", labelsize=8)
    left.grid(axis="y", alpha=.22)
    left.legend(frameon=False)
    im = right.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    right.set_yticks(range(4), [LABEL[m] for m in MODES])
    right.set_xticks([0, 7, 15, 23, 31])
    right.set(xlabel="原 NPU 编号（此处不按启动时间排序）", title="利用率热图（亮=高）")
    fig.colorbar(im, ax=right, label="利用率（%）")
    fig.suptitle(f"图3｜C_{variant} 逐 NPU 利用率 · seed={SEEDS[variant][0]}, repeat=0 · IO 等待不计入计算", fontsize=12)
    fig.tight_layout(rect=(0, .02, 1, .93))
    save(fig, f"C{variant}_util_box_heatmap")


def bandwidth(data, variant):
    bin_s = .01
    horizon = max(value[0]["makespan_s"] for value in data.values())
    n = math.ceil(horizon / bin_s)
    fig, axes = plt.subplots(4, 1, figsize=(14.6, 10.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.985, bottom=.07, top=.9, hspace=.37)
    fig.suptitle(f"图4｜C_{variant} 存储带宽时序 · seed={SEEDS[variant][0]}, repeat=0\n"
                 "10 ms 精确区间积分；橙=申请需求，蓝=实际速率，黑虚线=120 GB/s", y=.98)
    curve_rows, summary = [], []
    maximum = 120.0
    for ax, mode in zip(axes, MODES):
        report, events, *_ = data[mode]
        demand, actual = np.zeros(n), np.zeros(n)
        intervals = [e for e in events if e["event"] == "storage_interval"]
        for e in intervals:
            assert e["actual_Bps"] <= 120e9 + 1
            for index in range(int(e["from"] / bin_s), min(n, math.ceil(e["t"] / bin_s))):
                overlap = max(0.0, min(e["t"], (index + 1) * bin_s) - max(e["from"], index * bin_s))
                actual[index] += overlap * e["actual_Bps"]
                demand[index] += overlap * e["demand_Bps"]
        assert abs(actual.sum() - report["bytes_read"]) / report["bytes_read"] < 1e-10
        CHECKS["bandwidth_integral_checks"] += 1
        actual, demand = actual / bin_s / 1e9, demand / bin_s / 1e9
        xs = (np.arange(n) + .5) * bin_s * 1000
        maximum = max(maximum, float(demand.max()))
        ax.plot(xs, np.maximum(demand, .5), color="#DD8452", lw=.9, label="需求 Σ申请率")
        ax.plot(xs, np.maximum(actual, .5), color="#4C72B0", lw=1, label="实际 Σ速率")
        ax.axhline(120, color="black", ls="--", lw=1)
        ax.set(yscale="log", ylabel=f"{LABEL[mode]}\nGB/s（对数）", xlim=(0, horizon * 1000))
        ax.grid(alpha=.2, which="both")
        over = sum((e["t"] - e["from"]) for e in intervals if e["demand_Bps"] > 120e9) / report["makespan_s"]
        sat = sum((e["t"] - e["from"]) for e in intervals if e["actual_Bps"] >= .9 * 120e9) / report["makespan_s"]
        avg = report["bytes_read"] / report["makespan_s"] / 1e9
        ax.set_title(f"需求超订 {over:.1%}｜实际饱和(≥0.9B) {sat:.1%}｜全程平均 {avg:.1f} GB/s", loc="right", fontsize=9)
        summary.append({"variant": variant, "seed": SEEDS[variant][0], "mode": mode,
                        "oversubscribed_fraction": over, "saturated_fraction": sat,
                        "average_GBps": avg, "bytes_integral": report["bytes_read"]})
        for t, q, rate in zip(xs, demand, actual):
            curve_rows.append({"mode": mode, "bin_mid_ms": t, "bin_width_ms": 10,
                               "demand_GBps": q, "actual_GBps": rate})
    for ax in axes:
        ax.set_ylim(.5, maximum * 1.55)
    axes[0].legend(loc="lower right", fontsize=8, framealpha=.95)
    axes[-1].set_xlabel("仿真时间（ms）；无读取区间速率为 0，对数图仅显示在 0.5 GB/s 底线，CSV 保留真实 0")
    write_csv(f"C{variant}_bandwidth_10ms.csv", curve_rows)
    write_csv(f"C{variant}_bandwidth_metrics.csv", summary)
    save(fig, f"C{variant}_bandwidth")


def cdf(data, variant, by_class=False):
    if by_class:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), sharey=True)
        classes = list("ABC")
    else:
        fig, ax = plt.subplots(figsize=(10.2, 6))
        axes, classes = [ax], [None]
    curve_rows = []
    for ax, cls in zip(axes, classes):
        deadline = None
        for mode in MODES:
            reqs = [q for q in data[mode][0]["requests"].values() if cls is None or q["class_id"] == cls]
            xs = np.sort([q["ttft_s"] * 1000 for q in reqs])
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(np.r_[0, xs], np.r_[0, ys], where="post", color=POLICY_COLOR[mode], lw=1.6,
                    label=f"{LABEL[mode]}（p50 {np.quantile(xs, .5):.0f} / p95 {np.quantile(xs, .95):.0f} ms）"
                    if not by_class else LABEL[mode])
            widths = np.sort([(q["deadline_s"] - q["arrival_s"]) * 1000 for q in reqs])
            if deadline is None:
                deadline = widths
            else:
                assert np.allclose(deadline, widths, rtol=0, atol=1e-9)
            for i, (x, y) in enumerate(zip(xs, ys)):
                curve_rows.append({"mode": mode, "class": cls or "all", "rank": i + 1,
                                   "ttft_ms": x, "cdf": y})
        if cls is None:
            ax.step(np.r_[0, deadline], np.r_[0, np.arange(1, len(deadline) + 1) / len(deadline)],
                    where="post", color="#444444", ls=":", lw=1.2, label="期限宽度分布（共享输入）")
        else:
            assert max(deadline) - min(deadline) < 1e-8
            ax.axvline(deadline[0], color="#444444", ls=":", lw=1.3,
                       label=f"期限 {deadline[0]:.1f} ms")
            ax.set_title(f"{cls} 类（120 请求）")
        ax.set(xlim=(0, None), ylim=(0, 1.025), xlabel="TTFT（ms，包含中央准入等待）")
        ax.grid(alpha=.22)
        ax.legend(loc="lower right", fontsize=8, framealpha=.95)
    axes[0].set_ylabel("累计分布 CDF")
    fig.suptitle(f"图5｜C_{variant} {'分类' if by_class else '总体'} TTFT 经验 CDF · seed={SEEDS[variant][0]}, repeat=0", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    kind = "cdf_by_class" if by_class else "cdf"
    write_csv(f"C{variant}_{kind}.csv", curve_rows)
    save(fig, f"C{variant}_{kind}")


def percent(value):
    return f"{100 * value:.1f}%"


def report_text(records, reports):
    def agg(v, mode, field, cap=8):
        return aggregate(records, v, cap, mode, field)

    def image(name, alt):
        return f"![{alt}](figure/e26f_dv_{name}.png)"

    text = """# 实验结果：TFS 接入 Dynamo/vLLM——E26f 零读第三类对齐实验

| 项目 | 约定 |
|---|---|
| 实验日 / 报告整理日 | 原实验 2026-10-19；本报告 2026-10-20 从冻结记录重新整理，没有重跑仿真 |
| 参照格式 | `status_aware_network/实验结果-E26f-零读第三类实验-20261011.md`，沿用汇总、甘特、利用率、带宽、CDF 的组织与主要配色 |
| 负载 | A 簇到达：簇间隔为均值 16/98 秒的指数分布，每簇 16 条，末簇补余；B/C 各自逐条泊松到达，间隔为均值 1/98 秒的指数分布；A/B/C 各 120 条，共 360 条 |
| 请求形状 | A=(h=42883,u=256)，B=(9399,4096)，C_A=(0,2657)，C_B=(0,8110)；h 为远端命中 token，u 为新计算 token |
| 资源 | 32 worker/NPU 槽位，8 层，120 GB/s 聚合存储，8192 新计算 token/step，逻辑内存每 worker 8 GB；单流初始申请上限 200 GB/s |
| 比较方案 | 直接提交 `immediate` / 仅等待预算排序 `slack_only` / 配额+先到先服务 `quota_fcfs` / TFS `tfs` |
| 输入种子 / 路由重复 | C_A：416/417/418；C_B：424。每个输入种子、策略、请求数上限运行 3 次；Dynamo 平局选择未固定随机种子 |
| 总矩阵 | 4 条输入 × 2 档 max_num_seqs(1/8) × 4 策略 × 3 次路由重复 = 96 次；主报告展示上限 8，附表展示上限 1 |
| 汇总口径 | 同一输入种子的 3 次路由重复先取中位数，再对输入种子取中位数。分类指标分别取中位数，不能用它们反推总指标 |
| 轨迹图口径 | 固定 C_A seed416、C_B seed424，repeat0、max_num_seqs=8；每方案 360 请求。单次图不代表全部种子的中位轨迹 |
| SLO（时延目标）与 TTFT（首 token 时延） | 首 token 时间 ≤ arrival+4T0。T0 为 SA 同源单请求参考服务时间；TTFT 从 arrival 计起，包含中央准入等待；不使用统一 1 秒阈值作为本报告主口径 |
| 执行边界 | 真实 Dynamo SelectionService、vLLM Scheduler/Allocator；逐层 IO、计算时间、模型输出为仿真。利用率指模拟计算时间占比，不是物理 NPU 硬件利用率 |
| 存储与配额 | SA 聚合 max_min；每批每层一条读取流，下一层与本层计算重叠；TFS 从准入占配额，到首 token 释放，本实验只输出一个 token，无后续 decode |
| 版本 | SA 来源提交 `3cd826961bb53b05d09eb7f6e6949e4222230605`；本批运行实现 SHA256 `06f3914df6016cbee868b7eab54450fb7dd4161060d26c773b4696982b121ea7` |

## 0. 总判决

**条件性通过：当前合成执行模型下，TFS 相对直接提交提高 SLO 满足率，并降低平均 TTFT。不能据此确认真实硬件收益幅度。**

"""
    for v in SEEDS:
        gain = (agg(v, "tfs", "slo") - agg(v, "immediate", "slo")) * 100
        text += (f"- **C_{v}**：总 SLO {percent(agg(v, 'immediate', 'slo'))} → **{percent(agg(v, 'tfs', 'slo'))}**，"
                 f"提高 **{gain:.1f} 个百分点**；平均 TTFT {agg(v, 'immediate', 'ttft_ms'):.1f} → **{agg(v, 'tfs', 'ttft_ms'):.1f} ms**。\n")
    text += """
- 配额机制是本矩阵中的主要差异来源：仅排序策略没有表现出相近收益；TFS 与“配额+先到先服务”更接近。这是策略对照支持的现象，不是对配额与排序做独立因果量化。
- TFS 仍有反例与取舍：C_A seed418 的表现低于同源 SA 控制，A 类远未全部达标。详见 §6、§7。

## 1. 汇总表与 SLO 柱状图

主口径：max_num_seqs=8。方括号为输入种子 416/417/418 各自的路由重复中位数；C_B 只有 seed424。

"""
    for v in SEEDS:
        text += f"### C_{v}：{'零读轻算 C，单请求纯计算约 48.17 ms' if v == 'A' else '零读重算 C，单请求纯计算约 228.73 ms'}\n\n"
        text += "| 方案 | 总 SLO [逐输入种子] | A-SLO | B-SLO | C-SLO | 平均 TTFT(ms) | 中央等待(ms) | makespan(ms) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
        for mode in MODES:
            seeds = "/".join(f"{100*x:.1f}" for x in seed_values(records, v, 8, mode, "slo"))
            text += f"| {LABEL[mode]} | **{percent(agg(v, mode, 'slo'))}** [{seeds}] | "
            text += " | ".join(percent(agg(v, mode, cl + "_slo")) for cl in "ABC")
            text += f" | {agg(v, mode, 'ttft_ms'):.1f} | {agg(v, mode, 'central_wait_ms'):.1f} | {agg(v, mode, 'makespan_ms'):.1f} |\n"
        text += "\n"
    text += image("slo_bar", "总 SLO 满足率柱状图：四策略与 C_A/C_B") + "\n\n"
    text += "**读法**：浅蓝 C_A，深蓝 C_B，与原报告汇总图一致。柱内 A/B/C 白字为分类 SLO 独立中位数。误差线只表示输入种子间最小/最大值，不是置信区间；C_B 不画输入种子变异。\n\n"
    text += """### 四种策略的区别

| 方案 | 中央排序 | 重请求配额 | 实例内组批 |
|---|---|---|---|
| 直接提交 | 到达顺序 | 无 | 真实 vLLM |
| 仅等待预算排序 | deadline−now−T0 | 无 | 真实 vLLM |
| 配额+先到先服务 | 到达顺序，受限重请求可跳过 | 有 | 真实 vLLM |
| TFS | deadline−now−T0，受限重请求可跳过 | 有 | 真实 vLLM |

这里的“仅等待预算排序”不是原 SA 报告中的 EDF；“配额+先到先服务”也不是原 SA FCFS。本轮没有重新运行 local/MPC，不把原报告旧数值混进本轮策略柱状图。

没有配额时，请求到达即被下发，排序通常只影响同刻到达的候选，不会主动等待未来请求形成跨时刻中央队列。直接提交与仅排序之间的小幅数值差异也包含随机路由变化，不应全部归因于排序。

## 2. 各方案 full-sort-ms 执行时间线

蓝=A 计算，橙=B 计算，绿=C 计算，紫=混合类别计算，红=当前 step 等待本层读取，灰=worker 无请求。单类多请求批次仍显示该类别颜色。每个 step 包含 8 层，不能把色块数量当作批次数。

每个策略独立按首次 step 开始时间从上到下排序，同刻按原 worker 编号排列；左侧保留原编号。排序只改变展示，不改变路由。四策略在同一工况内共用横轴，较早结束的策略末尾补灰。中央队列仍可能有请求，灰色不表示系统无请求。

"""
    for v in SEEDS:
        text += f"### C_{v}：完整轨迹\n\n" + image(f"C{v}_full-sort-ms", f"C_{v} 四策略完整启动排序时间线") + "\n\n"
        text += f"[查看 C_{v} 前 200 ms 放大图](figure/e26f_dv_C{v}_startup-sort-ms.png)\n\n"
        text += "| 方案 | 总 SLO（本次） | step 数 | 多请求 step | 混合类别 step | 平均批次请求数 |\n|---|---:|---:|---:|---:|---:|\n"
        for mode in MODES:
            report = reports[v, SEEDS[v][0], 8, mode, 0]
            hist = report["actual_batch"]["request_count_histogram"]
            n = sum(hist.values())
            multi = sum(count for k, count in hist.items() if int(k) > 1)
            ref = next(r for r in csv_rows(ARCHIVE / "runs.csv") if r["variant"] == v
                       and r["seed"] == str(SEEDS[v][0]) and r["cap"] == "8"
                       and r["mode"] == mode and r["repeat"] == "0")
            text += f"| {LABEL[mode]} | {percent(report['profile_deadline_attainment'])} | {n} | {multi} ({multi/n:.1%}) | {ref['mixed_steps']} | {report['actual_batch']['requests']['mean']:.3f} |\n"
        text += "\n"
    text += "**读法**：直接提交下，多请求已进入 worker 队列，当前 step 因读等待而变慢，后续 step 获得更多组批候选。TFS 将重请求积压留在中央，worker 内积压减少；这解释了 C_A 中 TFS 批次较小，但不能把‘批次小’本身当作性能优势。\n\n"
    text += "## 3. NPU 利用率分布与热力图\n\n主口径为 `worker.compute_s / 本次 makespan`，IO 等待不算计算。混合批次的计算区间在每个 worker 只累计一次，不按批内请求重复相加。箱线图与热图均对应上述固定单次轨迹；热图按原 NPU 编号排列，便于观察负载分布。\n\n"
    for v in SEEDS:
        text += image(f"C{v}_util_box_heatmap", f"C_{v} NPU 利用率分布与热图") + "\n\n"
        text += f"**C_{v} 利用率汇总**（双层中位数，与 §1 相同；图中为固定单次，数值不必相同）：\n\n"
        text += "| 方案 | min | p50 | p90 | max | 均值 | CV | 含 IO 等待的均值 | makespan(ms) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        for mode in MODES:
            text += f"| {LABEL[mode]} | " + " | ".join(f"{agg(v, mode, k):.1f}%" for k in ("util_min", "util_p50", "util_p90", "util_max", "util_mean"))
            text += f" | {agg(v, mode, 'util_cv'):.3f} | {agg(v, mode, 'busy_mean'):.1f}% | {agg(v, mode, 'makespan_ms'):.1f} |\n"
        text += "\n"
    text += "**读法**：红菱形为 32 NPU 纯计算均值；CV 为标准差/均值，越小表示该次运行中各 worker 的计算占比越接近。‘计算+IO 等待’只是槽位占用对照，不可当作有效计算利用率。不同策略 makespan 不同，利用率差异同时包含分母变化，不能解释为计算量或硬件算力增加。\n\n"
    text += "## 4. 存储带宽时序\n\n沿用原报告：橙色为需求 Σ申请率，蓝色为实际 Σ速率，黑虚线为 120 GB/s，上下采用相同对数纵轴。对每个真实区间与 10 ms 时间桶求交并积分，不把整段按起点塞进一个桶。每条实际速率曲线的字节积分与原报告读取字节数核对。\n\n"
    for v in SEEDS:
        text += image(f"C{v}_bandwidth", f"C_{v} 四策略存储带宽时序") + "\n\n"
        text += "| 方案（固定单次） | 需求超订时间占比 | 实际≥108 GB/s时间占比 | 全程平均实际带宽(GB/s) |\n|---|---:|---:|---:|\n"
        for row in csv_rows(DATA / f"C{v}_bandwidth_metrics.csv"):
            text += f"| {LABEL[row['mode']]} | {percent(float(row['oversubscribed_fraction']))} | {percent(float(row['saturated_fraction']))} | {float(row['average_GBps']):.1f} |\n"
        text += "\n"
    text += "**读法**：C 零读，不直接产生带宽需求；进入混批后，C 会改变本层计算时长，进而改变下一层读取申请率 `V_batch/c_batch`。因此图中带宽差异同时包含准入、批次组成、路由和读算重叠。SA 的低需求规则会把剩余带宽均分给在途流，因此实际速率可以高于申请率；申请率在此不是硬限速。超订/饱和占比均以本策略整个 makespan 为分母，包含无读取时间；对数图将真实 0 速率画在 0.5 GB/s 底线，仅为显示，CSV 保留 0。\n\n"
    text += "## 5. TTFT 累积分布 CDF\n\n四策略颜色固定：直接提交灰、仅排序红、配额先到先服务橙、TFS 绿，与原报告的基线灰、期限类红、TFS 绿保持接近。实线是实际 TTFT，灰点线是共享输入的期限宽度分布。CDF 包含中央排队，不只统计下发之后的耗时。\n\n"
    for v in SEEDS:
        text += image(f"C{v}_cdf", f"C_{v} 总体 TTFT CDF") + "\n\n"
        text += image(f"C{v}_cdf_by_class", f"C_{v} A/B/C 分类 TTFT CDF") + "\n\n"
    text += "**读法**：总体 TTFT CDF 与期限宽度 CDF 的相对位置只能辅助观察，不能据两条边际分布直接计算 SLO；SLO 必须逐请求比较 TTFT 与该请求自身的期限宽度。分类图的每类期限相同，可直接在垂直期限线处读取该类达标比例。图中 p50/p95 是固定单次全 360 请求分位数，不是多次运行分位数的中位数。\n\n"
    text += "## 6. 与同事 SA 结果的衔接\n\n原 2026-10-11 报告含旧存储分配与 local/MPC 等臂，不能与本轮 max_min 结果直接相减。本轮独立导出同源 SA 输入，并运行相同 max_min 的 SA FCFS/TFS 控制；下表仅比较这些同源控制。SA 为单请求执行模型，DV 使用真实 Dynamo 路由、每 worker 排队与 vLLM 组批，两者不应预期逐请求完全一致。\n\n"
    controls = csv_rows(ARCHIVE / "sa_controls.csv")
    text += "| 工况 | SA FCFS 总 SLO | SA TFS 总 SLO | DV 直接提交（cap8） | DV TFS（cap8） |\n|---|---:|---:|---:|---:|\n"
    for v in SEEDS:
        sa_values = [stats.median(float(r["slo"]) for r in controls if r["variant"] == v and r["mode"] == m) for m in ("fcfs", "tfs")]
        text += f"| C_{v} | {percent(sa_values[0])} | {percent(sa_values[1])} | {percent(agg(v, 'immediate', 'slo'))} | {percent(agg(v, 'tfs', 'slo'))} |\n"
    text += "\n**保留反例**：C_A seed418，SA TFS 达标率 71.1%，DV TFS cap1 路由重复中位数 64.2%、cap8 为 57.5%。不能将汇总收益表述成所有输入都与 SA 一致，也不能保证所有 A 请求因 TFS 而更快。\n\n"
    text += """## 7. 计算模型与结论限制

1. **512 token 是模型参数，不是实测硬件饱和点。** 本轮继承 SA `synthetic-v1`，每层 `c=launch+aN/η(N)+b·attention_pairs`，`η=min(1,max(1/16,N/512))`。B 与两种 C 的单请求新计算 token 都超过 512，因此额外组批的纯计算加速较小。
2. **组批仍能改变 IO 重叠。** 每批每层读取量按成员求和，下一层预取窗口为本批计算时长。A+B 可用较长计算窗口隐藏读取；B+C 在 B 原本已能隐藏 IO 时额外收益较小。不能仅凭纯计算耗时判断整体组批收益。
3. **混批同步等待。** 仿真一个 step 在本层读取完成后执行本层；C 本身零读，若进入有读取的混批仍会等待该批读取。紫色表示共同执行，不表示将类别串行调用。
4. 输入与资源对齐不等于真实硬件校准。没有目标 GPU/NPU 的批次形状测量，无法证明实际组批收益、策略排名或收益幅度与本报告一致。输出为一个合成 token，没有长期 decode 连续组批。
5. C_A 只有 3 个输入种子，C_B 只有 1 个。路由重复不等于新增输入种子。误差线不是显著性检验，固定单次轨迹不代表中位数运行。
6. 独立 SA 对拍覆盖单请求、串行与双 worker 并发，批次上限为 1；没有多请求混批的独立执行真值。相同画像的自洽和账本守恒不能代替硬件真实性验证。
7. 原始 96 次实验均无抢占或重复加载；本轮报告未覆盖多存储节点/路径、写回、故障、反馈延迟与预测误差。TFS 的配额占用是准入计数，不是物理带宽预留。

## 8. 附表：max_num_seqs=1 对照

此档只限制每个 step 最多一个请求，仍使用 Dynamo 路由与 worker 排队。口径与主表相同，不将 cap1 视为原 SA 引擎的完全替代。

| 工况 | 方案 | 总 SLO | A-SLO | B-SLO | C-SLO | 平均 TTFT(ms) | makespan(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    for v in SEEDS:
        for mode in MODES:
            text += f"| C_{v} | {LABEL[mode]} | " + " | ".join(percent(agg(v, mode, field, 1)) for field in ("slo", "A_slo", "B_slo", "C_slo"))
            text += f" | {agg(v, mode, 'ttft_ms', 1):.1f} | {agg(v, mode, 'makespan_ms', 1):.1f} |\n"
    text += """

## 9. 验证、数据与复现

本次只读已有记录，不改运行时、配置、调度策略或旧图。重新核对 384 个原始文件 SHA256，读取 96 份报告、34,560 条请求；重新计算总/分类 SLO 与 TTFT，并对照原汇总 CSV。全部 3,072 个 worker 时间账本核对；8 次固定轨迹的 256 个 worker 计算与 IO 区间核对；8 条带宽实际速率积分核对读取字节。误差阈值与检查结果见 [生成验收记录](report_data/validation.json)。本次没有重新运行仿真或原单元测试。

- [96 次运行完整指标 CSV](report_data/run_metrics_96.csv)
- [生成验收与文件哈希](report_data/validation.json)
- 各固定轨迹的带宽 10ms 数值、CDF 排名数值和利用率 CSV 位于 `report_data/`；PNG 位于 `figure/`，可连同本 Markdown 一起发给同事。
- 原始记录：`data/dynamo_vllm_simulation/dynamo_vllm_cpu_sim/results/sa_alignment_20261019/`；归档摘要：同级 `sa_alignment_summary_20261019/`。完整事件文件为本地实验资产，分享本报告不需要附上运行环境。
- 复现：在项目根目录执行 `data/dynamo_vllm_simulation/dynamo_vllm_cpu_sim/.venv/bin/python note/tfs_dv/build_e26f_report.py`。需保留原始事件资产；脚本只生成报告、图和 CSV。

## 10. 简要解读

- 直接提交允许请求进入 worker 队列，积压会增加后续组批候选，也会增加排队和共同等待。
- TFS 将重请求等待留在中央，减少 worker 内积压，并给 B/C 留出更及时执行的机会；A 的中央等待仍计入 TTFT。
- 本报告用于对齐当前仿真实验的输入、机制、观测结果和限制。要验证真实部署结论，下一步需要独立测量单类/混合批次计算画像，而不是直接调高或调低饱和 token 参数来获得预期结论。
"""
    path = OUT / "实验结果-TFS接入DV-E26f对齐-20261020.md"
    path.write_text(text)
    return path


def main():
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "font.size": 10, "axes.unicode_minus": False})
    records, reports = collect()
    print("384 文件哈希、96 报告与原汇总核对通过。", flush=True)
    bar(records)
    worker_rows = []
    for variant in SEEDS:
        data = selected(reports, variant)
        timeline(data, variant)
        timeline(data, variant, zoom=True)
        utilization(data, variant)
        bandwidth(data, variant)
        cdf(data, variant)
        cdf(data, variant, by_class=True)
        for mode in MODES:
            report = data[mode][0]
            for w in report["workers"]:
                worker_rows.append({"variant": variant, "seed": SEEDS[variant][0], "repeat": 0,
                                    "mode": mode, "worker": w["id"], "compute_s": w["compute_s"],
                                    "stall_s": w["stall_s"], "idle_s": w["idle_s"],
                                    "makespan_s": report["makespan_s"],
                                    "utilization_percent": 100 * w["compute_s"] / report["makespan_s"]})
        print(f"C_{variant}：四策略的时间线、利用率、带宽与 CDF 完成。", flush=True)
    write_csv("selected_worker_utilization.csv", worker_rows)
    report = report_text(records, reports)
    CHECKS.update({"figures": len(FIGURES), "simulation_rerun": False,
                   "aggregation": "median routing repeats per input seed, then median input seeds",
                   "selected_traces": "C_A seed416 / C_B seed424, cap8 repeat0, all four policies",
                   "file_hashes": {str(p.relative_to(OUT)): sha(p)
                                   for p in FIGURES + [report, Path(__file__).resolve()] + sorted(DATA.glob("*.csv"))},
                   "original_reference": str(WORKSPACE / "data/status_aware_network/实验结果-E26f-零读第三类实验-20261011.md")})
    (DATA / "validation.json").write_text(json.dumps(CHECKS, indent=2, ensure_ascii=False) + "\n")
    print(report, flush=True)
    print(json.dumps({k: v for k, v in CHECKS.items() if k != "file_hashes"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
