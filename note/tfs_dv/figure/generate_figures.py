"""从已归档轨迹重画六色时间线（毫秒、按首次 step 启动排序），不运行仿真、不改变已有结果。"""

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch


WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = WORKSPACE / "data/dynamo_vllm_simulation"
SOURCE = ROOT / "dynamo_vllm_cpu_sim/results/sa_alignment_summary_20261019/selected_runs"
DEST = Path(__file__).resolve().parent
COLORS = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868",
          "mixed": "#9974a5", "io": "#c44e52", "idle": "#c7c7c7"}


def load(variant, seed, mode):
    folder = SOURCE / f"C{variant}-s{seed}-cap8-{mode}-r0"
    report = json.loads((folder / "report.json").read_text())
    with gzip.open(folder / "events.jsonl.gz", "rt") as stream:
        events = [json.loads(line) for line in stream]
    starts = {(e["worker"], e["step_id"]): e for e in events
              if e["event"] == "step_started"}
    ends = {(e["worker"], e["step_id"]): e["t"] for e in events
            if e["event"] == "step_complete"}
    compute = [e for e in events if e["event"] == "layer_compute_started"]
    grouped = {key: [] for key in starts}
    for event in compute:
        grouped[event["worker"], event["step_id"]].append(event)
    # 对照逐 worker 账本，确保红色仅为真实执行 step 内未计算的时间。
    for worker in report["workers"]:
        keys = [key for key in starts if key[0] == worker["id"]]
        active = sum(ends[key] - starts[key]["t"] for key in keys)
        calc = sum(e["end_s"] - e["t"] for key in keys for e in grouped[key])
        assert abs(calc - worker["compute_s"]) < 1e-9
        assert abs(active - calc - worker["stall_s"]) < 1e-9
        assert abs(report["makespan_s"] - active - worker["idle_s"]) < 1e-9
        intervals = sorted((starts[key]["t"], ends[key]) for key in keys)
        assert all(a[1] <= b[0] + 1e-12 for a, b in zip(intervals, intervals[1:]))
    return report, starts, ends, compute


def draw(ax, data, horizon, label, reorder=False):
    report, starts, ends, compute = data
    first = {wid: min(e["t"] for key, e in starts.items() if key[0] == wid)
             for wid in range(32)}
    order = sorted(first, key=lambda wid: (first[wid], wid)) if reorder else list(range(32))
    # 启动顺序图从上到下排列；编号图保持 NPU0 在下、NPU31 在上。
    pos = {wid: 31 - index if reorder else wid for index, wid in enumerate(order)}
    polygons = {key: [] for key in COLORS}

    def rect(kind, start, end, bottom, top):
        start, end = max(0, start), min(horizon, end)
        if end > start:
            polygons[kind].append([(start * 1000, bottom), (end * 1000, bottom),
                                   (end * 1000, top), (start * 1000, top)])

    for wid in range(32):
        y = pos[wid]
        rect("idle", 0, horizon, y - .46, y + .46)
    for key, event in starts.items():
        y = pos[key[0]]
        rect("io", event["t"], ends[key], y - .46, y + .46)
    for event in compute:
        key = event["worker"], event["step_id"]
        classes = sorted({report["requests"][rid]["class_id"]
                          for rid in starts[key]["num_scheduled_tokens"]})
        y = pos[event["worker"]]
        kind = classes[0] if len(classes) == 1 else "mixed"
        rect(kind, event["t"], event["end_s"], y - .46, y + .46)
    for kind in ("idle", "io", "A", "B", "C", "mixed"):
        ax.add_collection(PolyCollection(polygons[kind], facecolors=COLORS[kind],
                                        edgecolors="none", antialiased=False))
    ax.set(xlim=(0, horizon * 1000), ylim=(-.6, 31.6), ylabel=label)
    if reorder:
        ax.set_yticks([pos[wid] for wid in order], [f"{wid}" for wid in order], fontsize=7)
    else:
        ax.set_yticks([0, 8, 16, 24, 31], ["NPU0", "8", "16", "24", "NPU31"])
    ax.set_xlabel("时间（ms）")
    ax.grid(axis="x", alpha=.14)
    ax.set_axisbelow(True)


def figure(variant, seed, data, zoom=False):
    horizon = .2 if zoom else max(d[0]["makespan_s"] for d in data.values())
    fig, axes = plt.subplots(2, 1, figsize=(17, 7.4), sharex=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.075, top=.87, hspace=.22)
    fig.suptitle(f"C_{variant} · seed {seed} · repeat 0 · max_num_seqs=8"
                 + ("｜前 200 ms，各策略按首次 step 启动顺序排列" if zoom else "｜各策略按首次 step 启动顺序排列"),
                 fontsize=14, y=.99)
    fig.legend(handles=[Patch(color=COLORS[k], label=t) for k, t in
                         [("A", "A 计算"), ("B", "B 计算"), ("C", "C 计算（零读）"),
                          ("mixed", "混合计算"), ("io", "IO 等待"), ("idle", "无请求")]],
               loc="upper center", bbox_to_anchor=(.5, .955), ncol=6, frameon=False)
    for ax, mode, label in zip(axes, ("immediate", "tfs"), ("直接提交", "TFS")):
        draw(ax, data[mode], horizon, label, reorder=True)
    path = DEST / f"C{variant}-{'startup-sorted-ms' if zoom else 'full-sorted-ms'}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})
    DEST.mkdir(parents=True, exist_ok=True)
    summaries = []
    for variant, seed in (("A", 416), ("B", 424)):
        data = {mode: load(variant, seed, mode) for mode in ("immediate", "tfs")}
        figure(variant, seed, data)
        figure(variant, seed, data, zoom=True)
        for mode, (report, starts, _, _) in data.items():
            n = len(starts)
            multi = sum(e["num_requests"] > 1 for e in starts.values())
            total = 32 * report["makespan_s"]
            pct = {k: 100 * sum(w[k] for w in report["workers"]) / total
                   for k in ("compute_s", "stall_s", "idle_s")}
            summaries.append(f"| C_{variant} | {'直接提交' if mode == 'immediate' else 'TFS'} "
                             f"| {n} | {multi}（{multi/n:.1%}） | {pct['compute_s']:.1f}% "
                             f"| {pct['stall_s']:.1f}% | {pct['idle_s']:.1f}% |")
    md = """# SA-E26f 执行时间线：混合计算紫色版

使用 2026-10-19 已归档的原始事件重新绘图。没有重跑仿真，没有修改输入或调度算法。

- 蓝：A 计算；橙：B 计算；绿：C 计算（零读）；紫：混合类别计算；红：IO 等待；灰：该 worker 无请求。
- 同一 step 包含两个或更多类别时，整个计算区间显示紫色。相同类别的多请求批次仍显示类别颜色。
- 红色只表示当前执行 step 等待本层读取。后台预取与计算重叠时仍显示计算色。
- 灰色按 worker 口径统计。中央准入队列可能仍有请求，因此灰色不代表系统没有请求。
- 每个 step 包含 8 层。色块数量不等于请求数或批次数。单类多请求批次仍显示该类颜色。
- 全部横轴使用 ms。完整图与放大图均按首次 step 开始时间从上到下排列；同刻按 worker 编号排列。左侧保留原 worker 编号。每个策略独立排序，不能按行比较相同 worker。
- 每对策略共用时间轴，较早结束策略的末尾补灰。排序只改变展示位置，不改变真实路由和执行时间。

## C_A：完整时间线

三类各 120 条，共 360 条；seed=416，repeat=0，每个 step 最多 8 请求。

![C_A 完整时间线](CA-full-sorted-ms.png)

## C_A：启动阶段放大

仅显示前 200 ms。每个策略独立按首次 step 开始时间从上到下排序，左侧数字为原 worker 编号。此图只用于观察启动先后，不用于按行比较同一 worker。

![C_A 前 200 ms 启动顺序](CA-startup-sorted-ms.png)

## C_B：完整时间线

三类各 120 条，共 360 条；seed=424，repeat=0，每个 step 最多 8 请求。

![C_B 完整时间线](CB-full-sorted-ms.png)

## C_B：启动阶段放大

![C_B 前 200 ms 启动顺序](CB-startup-sorted-ms.png)

## 原始计数与时间占比

占比来自各策略自身运行期间的 32 个 worker 时间总和；不包含完整图中为统一横轴而补出的末尾灰色。

| 工况 | 策略 | step 数 | 多请求 step | 计算时间占比 | IO 等待占比 | 无请求占比 |
|---|---|---:|---:|---:|---:|---:|
""" + "\n".join(summaries) + """

绘图前，逐 worker 核对计算、IO 等待、无请求三项时长与原 report.json 一致，误差小于 1e-9 秒，并检查 step 不重叠。所有 4 次运行、128 个 worker 均通过。

数据：`dynamo_vllm_cpu_sim/results/sa_alignment_summary_20261019/selected_runs/` 下的四份 `events.jsonl.gz` 和 `report.json`。

复现：在项目根目录执行 `data/dynamo_vllm_simulation/dynamo_vllm_cpu_sim/.venv/bin/python note/tfs_dv/figure/generate_figures.py`。
"""
    path = DEST / "README.md"
    path.write_text(md)
    print(path)
    print("4 次运行、128 个 worker 时间账本校验通过。")


if __name__ == "__main__":
    main()
