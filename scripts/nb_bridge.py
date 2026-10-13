"""记事本 ↔ 旧宿主 桥接层。

旧接入点（Hermes 插件 snowfox-memory、DSH 的 snowfox-inject.mjs）都消费
`memories/_assembled_context.md`。本模块让这些接入点不改协议就能吃新模式：
    写入端 → nb_assemble.write_assembly(query) 产出同一文件
    读取端 → 旧代码原样读该文件，无需改动

检索 query 的来源有三种，按优先级：
    1) 显式传入的 query 参数
    2) `notebook/.query` 文件（宿主把本轮用户输入写进去，再拉起本脚本）
       为什么用文件而不是命令行参数：用户输入含引号/换行/HTML/中文标点时，
       命令行转义极易出错（Windows shell 尤甚），文件传递无转义问题。
    3) 无 query → nb_assemble 退回「核心记忆里的活跃待办」

索引维护：assemble 只读索引；索引由 nb_index / nb_write 重建。
本模块提供 ensure_index()，在索引缺失或明显过期（输入文件比索引新）时重建，
避免「新写的记忆检索不到」这一最常见故障。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_config as C

# 宿主写入本轮用户输入的约定路径
QUERY_FILE = C.MEMORY / ".query"


def read_query_file() -> str:
    """读宿主写入的本轮查询；不存在或读取失败返回空串。"""
    try:
        if QUERY_FILE.exists():
            return QUERY_FILE.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        pass
    return ""


def write_query_file(text: str) -> None:
    """宿主写入本轮用户输入，供后续 assemble 取用。"""
    try:
        C.ensure_dirs()
        QUERY_FILE.write_text(text or "", encoding="utf-8")
    except OSError:
        pass


def index_stale() -> bool:
    """索引需要重建吗？

    三种情况都算「需要」：
      1. 索引不存在
      2. 索引**损坏或为空**（仅看 mtime 会把写坏的索引判成新鲜 → 永久静默
         返回 0 命中，与「真的没命中」不可区分。2026-10-05 审查实测）
      3. 任一数据文件比索引新
    """
    if not C.INDEX.exists():
        return True
    # 先验可解析性，再比 mtime —— 顺序不能反：损坏的索引可能 mtime 最新
    try:
        import nb_index
        nb_index.load()
    except Exception:
        return True
    try:
        idx_m = C.INDEX.stat().st_mtime
    except OSError:
        return True
    sources = [C.CORE, C.STREAM, *C.WS_DIR.glob("*.md")] if C.WS_DIR.exists() else [C.CORE, C.STREAM]
    for p in sources:
        try:
            if p.exists() and p.stat().st_mtime > idx_m:
                return True
        except OSError:
            continue
    return False


def ensure_index(force: bool = False) -> str:
    """索引缺失/过期则重建。返回可读状态串。"""
    import nb_index
    if force or index_stale():
        if C.INDEX.exists():
            try:
                C.INDEX.unlink()
            except OSError:
                pass
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            nb_index.build(verbose=False)
        return "rebuilt"
    return "fresh"


def rebuild_assembly(query: str = "") -> dict:
    """对外唯一入口：确保索引新鲜，然后按 query 组装并写盘。

    query 为空时自动退回 .query 文件，再退回活跃待办。
    """
    if not query.strip():
        query = read_query_file()
    status = ensure_index()
    import nb_assemble
    res = nb_assemble.write_assembly(query=query)
    res["index"] = status
    return res


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = _s.argv[1:]
    # --set-query 从 stdin 读入本轮用户输入并只写文件（不组装），
    # 供宿主在 turn 开始时低成本地登记查询。
    if args and args[0] == "--set-query":
        text = _s.stdin.read()
        write_query_file(text)
        print(f"query set ({len(text)} chars)")
        _s.exit(0)
    # 无参数时读 stdin（宿主用管道把用户输入送进来）——
    # 走管道而非命令行参数：用户输入含引号/换行/中文标点时命令行转义极易出错。
    q = " ".join(args)
    if not q and not _s.stdin.isatty():
        try:
            q = _s.stdin.read()
        except Exception:
            q = ""
    r = rebuild_assembly(q)
    print(f"index={r['index']}  bytes={r['bytes']}  core={r['core_kb']}KB  "
          f"hits={len(r['retrieval']['hits'])}  query={r['retrieval'].get('query', '')[:60]!r}")

