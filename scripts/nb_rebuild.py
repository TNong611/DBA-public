"""DBA 一键重建索引 —— 克隆仓库后的第一条命令。

═══════════════════════════════════════════════════════════════
为什么需要这个脚本
═══════════════════════════════════════════════════════════════
`.gitignore` 把 `memory/.index/` 与 `memory/.vec/` 排除在版本库外（它们是整体
重写的二进制/大 JSON，每次重建都是全新文件，进 git 会撑爆历史）。

于是**刚 clone 下来的仓库是没有索引的** —— 这时检索返回 0 命中，而
「0 命中」与「记忆里真没这件事」在旧实现里长得一模一样。那正是本项目反复
踩的坑：静默失败冒充正常结果。所以克隆后需要一个**明确、唯一、可说出口**
的动作把索引建起来，而不是指望使用者自己想起来有哪几个脚本要手跑。

    py nb_rebuild.py            # 重建关键词 + 向量
    py nb_rebuild.py --kw       # 只重建关键词（零依赖、零 GPU，秒级）
    py nb_rebuild.py --if-stale # 只在陈旧时重建（给计划任务用）

关键词索引零依赖，一定能成；向量索引需要 Ollama + GPU，失败**不影响**
关键词检索，但会明确报告（不静默）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nb_config as C  # noqa: E402


def rebuild_keywords(verbose: bool = True) -> dict:
    """重建关键词倒排索引（纯 Python 零依赖）。"""
    import nb_index
    t0 = time.time()
    idx = nb_index.build(verbose=False)
    r = {
        "docs": len(idx["docs"]),
        "terms": len(idx["postings"]),
        "elapsed_s": round(time.time() - t0, 2),
        "bytes": C.INDEX.stat().st_size,
    }
    if verbose:
        print(f"关键词: {r['docs']} 段 / {r['terms']} 词，{r['elapsed_s']}s，"
              f"{r['bytes'] / 1024:.1f}KB → {C.INDEX}")
    return r


def rebuild_vectors(verbose: bool = True, if_stale: bool = False) -> dict:
    """重建向量索引（需 Ollama + GPU）。

    返回**结构化状态** {"status": "built"|"skipped"|"failed", ...}。

    ⚠️ 不能用 None 表示「没做」：那样「无需重建（跳过）」和「重建失败」就是
    同一个值，主流程只能靠 `C.VEC_INDEX.exists()` 去猜 —— 于是**已有旧索引
    而本次重建失败**时，会报告「向量为既有索引，本次未重建」，把失败掩盖成
    一次正常跳过。这正是本项目「哨兵值不得与无值共用返回通道」的第 N 次复发，
    2026-10-06 复核时发现，所以拆成三态。
    """
    try:
        import nb_vec
    except Exception as e:
        print(f"⚠️ 向量层不可用（{type(e).__name__}: {e}）→ 跳过，关键词通道不受影响")
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    try:
        if if_stale and not nb_vec.stale():
            if verbose:
                print(f"向量: 已是最新，跳过（{nb_vec.stats()}）")
            return {"status": "skipped", "reason": "already-fresh"}
        t0 = time.time()
        r = nb_vec.build(verbose=False)
        r["elapsed_s"] = round(time.time() - t0, 1)
        if verbose:
            print(f"向量: {r['count']} 段 × {r['dim']} 维，{r['elapsed_s']}s，"
                  f"{r['bytes'] / 1024 / 1024:.1f}MB → {C.VEC_INDEX}")
            print(f"      GPU: {nb_vec.gpu_status()}")
        return dict(r, status="built")
    except Exception as e:
        # 明确报错，不装成功。
        print(f"🚨 向量重建失败（{type(e).__name__}: {e}）")
        print(f"   关键词通道仍可用；修好后重跑：py nb_rebuild.py --vec")
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"}


def main(argv: list[str]) -> int:
    only_kw = "--kw" in argv
    only_vec = "--vec" in argv
    if_stale = "--if-stale" in argv

    print(f"DBA 重建索引  root={C.ROOT}")
    C.ensure_dirs()

    failed = 0
    if not only_vec:
        try:
            rebuild_keywords()
        except Exception as e:
            print(f"🚨 关键词重建失败（{type(e).__name__}: {e}）")
            failed += 1

    vec_status = None
    if not only_kw:
        vec_status = rebuild_vectors(if_stale=if_stale)["status"]

    # 结论必须明确：让调用方一眼看出「检索可用吗、语义可用吗」。
    # ⚠️ 判定**只看结构化的 status**，不看「索引文件在不在」—— 文件可能是
    # 上一次留下的旧索引，拿它冒充本次成功就是掩盖失败。
    print()
    if failed:
        print("结果: ❌ 关键词索引未能建立 —— 检索不可用")
        return 1
    if not C.INDEX.exists():
        print("结果: ❌ 关键词索引仍不存在")
        return 1
    if only_kw:
        print("结果: ✓ 关键词可用（未处理向量）")
    elif vec_status == "built":
        print("结果: ✓ 关键词 + 向量均可用")
    elif vec_status == "skipped":
        print("结果: ✓ 关键词 + 向量均可用（向量已是最新，未重建）")
    elif vec_status == "failed":
        # 关键区分：**本次失败了**，不要说成「用了既有索引」。
        print("结果: ⚠️ 关键词可用；向量**本次重建失败**"
              "（若已有旧索引则仍是旧内容，新记忆召不回）")
        return 1
    else:
        print("结果: ⚠️ 关键词可用；向量未处理")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
