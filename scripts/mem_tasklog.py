"""任务日志模块 — 管理每日任务记录与工作区。

- worklog: 记录当天任务(doing/done/todo)
- next-day compress: 把昨天的日志转成**待摘要素材**，交由云端模型并入 core
- workspace: 管理当天工作区目录

⚠️ 2026-10-06 迁移到 DBA 后改了三点：
  1. 不再 `from mem_config import get_config`（那是五级记忆的配置模块，没跟着
     搬到 DBA，直接 import 会 ModuleNotFoundError）。实测它在 DBA/scripts 下
     一跑就崩，而崩溃表现是计划任务里 STEP 1/2 exit=1 —— 所以必须去掉。
     （`get_config` 本来也没被用到，是死 import。）
  2. 路径不再从 HERMES_HOME 推。记忆数据已迁到 DBA 仓库，路径统一问 nb_config。
     ⚠️ 但 **workspace 仍留在 hermes 下** —— 它装的是当天的临时文件和截图，
     是工作区不是记忆，没有版本化的价值，不该进 git 仓库。
  3. tasklog 仍在 hermes/memories 下（同上：它是当日流水，不是长期记忆）。
     DBA 这边的 **memory/ 只放三层记忆 + 索引**，保持仓库干净。
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

# ── 记忆数据（DBA）：交给 nb_config 决定，它读 DBA_HOME ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import nb_config as _C
except Exception:  # pragma: no cover - 仅在 nb_config 缺失时
    _C = None

# ── 当日工作区与任务日志（仍在 hermes 下，不进 DBA 仓库） ──
HH = os.environ.get("HERMES_HOME") or (
    os.environ.get("USERPROFILE", "C:\\Users\\tnong") + "/AppData/Local/hermes"
)
MEM = HH + "/memories"
WORK = HH + "/workspace"

# ── Task Log ──

TASKLOG_PATH = MEM + "/tasklog.md"

def today_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def init_tasklog():
    """Ensure today's tasklog section exists."""
    if not os.path.exists(TASKLOG_PATH):
        with open(TASKLOG_PATH, "w", encoding="utf-8") as f:
            f.write(f"# 任务日志\n_自动管理，当日记录，次日压缩到 F0_\n\n---\n\n## {today_tag()}\n\n")
        print(f"  tasklog: created with today tag {today_tag()}")
        return
    content = open(TASKLOG_PATH, "r", encoding="utf-8").read()
    if f"## {today_tag()}" not in content:
        with open(TASKLOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n## {today_tag()}\n\n")

def write_tasklog_entry(status: str, content: str):
    """Write a task entry under today's section. status: doing/done/todo."""
    init_tasklog()
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"- **{ts}** [{status}] {content}\n"
    lines = open(TASKLOG_PATH, "r", encoding="utf-8").read().split("\n")
    # find today's section and insert after it
    for i, line in enumerate(lines):
        if line.strip() == f"## {today_tag()}":
            # find the next section or end
            insert_at = i + 1
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("## "):
                    insert_at = j
                    break
                insert_at = j + 1
            lines.insert(insert_at, entry)
            break
    open(TASKLOG_PATH, "w", encoding="utf-8").write("\n".join(lines))
    print(f"  tasklog: added [{status}] {content}")

def compress_yesterday_to_fixed():
    """把昨天的任务日志并入核心记忆（core.md）。

    2026-10-05 迁移：目标从 `fixed.md`（五级记忆的 F0，已废弃）改为
    `notebook/core.md`（记事本三层的核心记忆）。上限取 `nb_config.CORE_MAX_KB`
    （15KB），不再是硬编码的 7KB。

    ⚠️ 这里**不做**「取前 40 行」式的搬运。用户明确定义过：压缩必须是 LLM
    摘要/提炼，把原始内容整段搬到下一级不叫压缩，会被当场指出。
    所以本函数只把昨天的原始日志**转存为待摘要素材**，真正的摘要由云端模型
    在做 core 维护时完成（core 按分工本就由云端模型写）。
    绝不把原始日志直接灌进 core。

    ⚠️ 旧实现还有一处性质更差的问题：它一边写 `fixed.md`、一边用
    `_trim_fixed_to_7kb` 按「整段丢弃最旧的块」来腾地方 —— 那是**丢记忆**，
    不是压缩。新实现不再丢块，超限时只告警，交由云端模型决定怎么并。
    """
    tasklog = TASKLOG_PATH
    if not os.path.exists(tasklog):
        print("  tasklog: not found, nothing to compress")
        return

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    content = open(tasklog, "r", encoding="utf-8").read()
    sections = content.split("\n## ")

    yday_section = None
    remaining = []
    for sec in sections:
        if sec.startswith(yesterday):
            yday_section = "## " + sec
        else:
            remaining.append(sec)

    if not yday_section:
        print(f"  tasklog: no section for {yesterday}, nothing to compress")
        return

    raw_bytes = len(yday_section.encode("utf-8"))

    # 素材落盘（不是 core）：留给云端模型摘要用
    try:
        import nb_config as _C
        _C.ensure_dirs()
        staging = _C.MEMORY / "_pending_core.md"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(staging, "a", encoding="utf-8") as f:
            f.write(f"\n\n## 待并入核心记忆 | tasklog-{yesterday} | {stamp}\n\n")
            f.write(yday_section.strip() + "\n")
        # ⚠️ 上限判断用**原始字节**，不用 `size/1024` 的浮点近似：
        # 15361 字节 → 15.0KB（保留一位小数后）→ 与 15 比较为 False →
        # **刚超限的一刻静默通过**。同一处 bug 本项目已有 4 个副本。
        core_b = _C.core_bytes() if hasattr(_C, "core_bytes") else (
            os.path.getsize(_C.CORE) if os.path.exists(_C.CORE) else -1)
        over_b = _C.over_core_limit(core_b) if hasattr(_C, "over_core_limit") else 0
        print(f"  tasklog: {yesterday} ({raw_bytes}B 原文) → 待摘要素材 "
              f"{staging.name}；core 当前 {core_b / 1024:.2f}KB/{_C.CORE_MAX_KB}KB"
              f"（{core_b}B/{_C.core_limit_bytes() if hasattr(_C, 'core_limit_bytes') else _C.CORE_MAX_KB * 1024}B）")
        if over_b > 0:
            print(f"  ⚠️ core 已超上限 {over_b} 字节，需云端模型摘要压缩后再并入")
    except Exception as e:
        print(f"  tasklog: staging failed: {e}")
        return

    # 从 tasklog 移除昨天的段落（已转存为素材，避免重复并入）
    with open(tasklog, "w", encoding="utf-8") as f:
        f.write("\n## ".join(remaining))


def _trim_fixed_to_7kb(path: str):
    """【已废弃，保留仅为兼容旧调用】

    旧实现按「整段丢弃最旧的块」把 F0 压到 7KB —— 那是丢记忆而非压缩，
    且 7KB 是旧的硬编码上限（现为 nb_config.CORE_MAX_KB=15KB）。
    现由云端模型负责 core 的摘要与合并，本函数不再被 compress 流程调用。
    """
    if not os.path.exists(path):
        return
    sz = os.path.getsize(path)
    print(f"  F0: [已废弃] 不再按丢弃整块的方式裁剪（{sz/1024:.1f}KB），"
          f"改由云端模型摘要维护")

# ── Workspace ──

def init_workspace():
    """Ensure today's workspace directory exists, clean up yesterday's."""
    today = today_tag()
    today_dir = f"{WORK}/{today}"
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_dir = f"{WORK}/{yesterday}"
    
    # Clean yesterday's workspace
    if os.path.exists(yesterday_dir):
        shutil.rmtree(yesterday_dir)
        print(f"  workspace: cleaned {yesterday}")
    
    # Create today's
    os.makedirs(today_dir, exist_ok=True)
    print(f"  workspace: ready → {today_dir}")
    return today_dir

def save_to_workspace(filename: str, content: bytes | str):
    """Save a file to today's workspace."""
    today_dir = init_workspace()
    path = f"{today_dir}/{filename}"
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as f:
        f.write(content)
    print(f"  workspace: saved {filename}")
    return path

def list_workspace() -> list[str]:
    """List files in today's workspace."""
    today_dir = f"{WORK}/{today_tag()}"
    if not os.path.exists(today_dir):
        return []
    return os.listdir(today_dir)

# ── CLI ──

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "init":
        init_tasklog()
        init_workspace()
    elif action == "compress":
        compress_yesterday_to_fixed()
    elif action == "add":
        status = sys.argv[2] if len(sys.argv) > 2 else "todo"
        text = sys.argv[3] if len(sys.argv) > 3 else "(no detail)"
        write_tasklog_entry(status, text)
    elif action == "workspace-list":
        for f in list_workspace():
            print(f)
    elif action == "status":
        init_tasklog()
        init_workspace()
        print(f"  workspace files: {len(list_workspace())}")
        print(f"  tasklog exists: {os.path.exists(TASKLOG_PATH)}")
    else:
        print(f"Unknown action: {action}")
        print("Usage: mem_tasklog.py [init|compress|add <status> <text>|workspace-list|status]")
