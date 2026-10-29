# 复现范围

## 重画与验证报告

在报告目录运行 `python build_report.py`，需要 matplotlib、numpy 和可用中文字体。
运行 `python verify_report.py` 核对小表、CDF、时间账本、带宽积分、源码指纹与相对链接，
并重建报告目录旁的ZIP。报告目录内的数据足够完成这些操作，不依赖原始实验目录。

## 复跑仿真

runtime_source.tar.gz保存本次执行的sim/scripts/configs/patched和适配器文件。
它是DV未提交工作树的精确小型快照，含新增trace_io.py；不能仅用source_head替代。
ABC/ABCD含冻结驱动、配置、实际执行入口、分配表和小型验收记录。
原始Mooncake的output_length和hash_ids未参与本次构造；人工缓存量不是实测命中。

完整重跑仍需要原工作区布局、匹配versions/source-fingerprints的上游、Python虚拟环境，
以及provenance.json指定SHA的加权block16 Dynamo原生构建。分享包不含这些大依赖，
因此不是解压后即可独立启动的完整运行环境。run.py中路径按工作区解释。
prepare拒绝覆盖已有实验；正式记录不可覆盖。若移植到新环境，需先单独完成依赖验收，
再修改本地实验输出路径并保留新旧指纹。不能把不同环境的新轮次混入本次结果。

本轮配置：单共享池90e9 B/s、K=1、12实例、每步最多4请求/8192新token，
78层流水、TP1/DP8/EP8、每方案72预热+360正式、各一次。
完整原始事件保留本机DV results/mooncake360_global4_90g_20261029，未放入ZIP。
