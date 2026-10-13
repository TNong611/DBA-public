"""DBA 记忆日常维护 + 组装（双脑架构）。

跑在计划任务 `DSH_Memory_Daily`（每日 00:05），由 memory-daily.ps1 调起。

按「单文件五级记忆」的旧实现会 consolidate / retire / dedupe，最后用
`mem_assembly.assemble_budgeted` 覆写 `_assembled_context.md`。
那两点在 DBA 架构下**必须改掉**，否则每天 00:05 会静默破坏记忆：

  1. 旧组装器读 fixed/long_term/summary/recent.md（已迁为 *.legacy），会用
     **过期内容覆盖**新组装产物 —— 注入的系统提示变成旧五级记忆，而且不报错。
     现在改为调用 `nb_bridge.rebuild_assembly`。
  2. consolidate / retire / dedupe 都作用于 L2/L3 那些已废弃文件。DBA 里
     stream/ws **不限长度**，L1→L2→L3 级联压缩这一步根本不存在。
     所以默认跳过，仅在显式传 --legacy 时才跑（回退或整理归档用）。

⭐ 2026-10-06：数据与脚本都搬到 `D:/AI/DBA`，本文件从 ~/.dsh/scripts 迁入。
   路径不再从 HERMES_HOME 推 —— DBA 的约定是「数据跟着代码走」，
   仓库根由 nb_config 从本文件位置反推，可用 DBA_HOME 覆盖。
"""
import sys
from datetime import datetime
from pathlib import Path

# 本文件位于 <repo>/scripts/，同级就是全部记忆工具链
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def run_script(name):
    """跑一个**已废弃**的旧脚本。只在 --legacy 路径下使用。"""
    import os
    parts = name.split()
    s = SCRIPTS / parts[0]
    if not s.exists():
        print(f"WARN: {s} not found")
        return
    cmd = f'"{sys.executable}" "{s}" {" ".join(parts[1:])} 2>&1'
    print(f"[{name}] {os.popen(cmd).read()[:200].strip()}")


def assemble_context():
    """按 DBA 三层组装注入产物。

    产出 `_core_context.md`（稳定段）+ `_hits_context.md`（变化段）
    + `_assembled_context.md`（拼接，供仍按单文件消费的调用方）。
    无 query 时用核心记忆里的活跃待办作为检索起点。
    """
    import nb_bridge
    res = nb_bridge.rebuild_assembly("")
    hits = len(res["retrieval"]["hits"]) if res.get("retrieval") else 0
    print(f"OK: dba assembly {res['bytes']}B "
          f"(core {res['core_kb']}KB, hits {hits}, index {res['index']}, "
          f"build {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')})")
    # 组装预算告警必须显式打出来：下游 DSH 插件超 64KB 会**整段丢弃**记忆，
    # 而它不报错。这里喊一声是最后一道能被看见的机会。
    if res.get("budget_warning"):
        print(f"WARN: {res['budget_warning']}")
    return res


def rebuild_vectors():
    """向量索引日常维护。

    ⚠️ 只在索引**陈旧**时才重建：全量重建 511 段约 12s（要过 Ollama），
    每天无条件跑一次是浪费；但也不能不跑 —— 陈旧索引会让新写的记忆
    在语义通道里永远召不回，表现为「刚记的东西搜不到」。
    """
    try:
        import nb_vec
    except Exception as e:
        print(f"WARN: 向量层不可用（{type(e).__name__}: {e}）→ 跳过，关键词通道不受影响")
        return
    try:
        if not nb_vec.stale():
            print(f"OK: 向量索引已是最新（{nb_vec.stats()}）")
            return
        r = nb_vec.build(verbose=False)
        print(f"OK: 向量索引重建 {r['count']} 段 × {r['dim']} 维，{r['elapsed_s']}s")
    except Exception as e:
        # 向量层挂了**不能**让整个维护脚本失败：关键词通道仍是可用的。
        # 但这句 WARN 必须写进日志，否则就是又一次静默降级。
        print(f"WARN: 向量重建失败（{type(e).__name__}: {e}）→ 关键词通道照常")


if __name__ == "__main__":
    legacy = "--legacy" in sys.argv
    if legacy:
        # 仅用于回退五级架构或整理旧归档；日常不再执行
        run_script("mem_consolidate.py")
        run_script("mem_retire.py")
        run_script("mem_dedupe.py --apply")
    else:
        print("DBA 双脑架构：跳过 L2/L3 级联压缩步骤"
              "（stream/ws 不限长度；旧脚本仅 --legacy 时运行）")
    assemble_context()
    rebuild_vectors()
