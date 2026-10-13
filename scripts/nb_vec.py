"""DBA 向量记忆层 —— 语义检索通道，与关键词索引**零耦合**。

═══════════════════════════════════════════════════════════════════
为什么要有这一层（以及为什么它必须和关键词索引分开）
═══════════════════════════════════════════════════════════════════
关键词索引（nb_index.py）是**精确 token 匹配**：查「记忆架构」只命中含这四个
字（及其二元组）的段落。它的强项是精准、可解释、零依赖；弱项是**同义改写**：

    查「怎么让模型记住我」  → 命不中标题写着「记忆连续性方案」的段落

向量检索补的正是这个洞：把 query 和段落都映到同一个语义空间，靠**余弦距离**
找近邻，同义改写也能命中。两者是互补的，不是替代关系 —— 所以它们各自独立
建索引、独立检索、各自可单独工作。任一方挂了，另一方照常返回结果。

⚠️ 这个「零耦合」不是洁癖，是可靠性要求：向量层依赖 Ollama 进程 + GPU，而
   关键词层是纯 Python 零依赖。如果用同一个入口把两者串起来，Ollama 一挂
   就等于**整个记忆检索全哑**，连原本能用的精确检索也没了。

═══════════════════════════════════════════════════════════════════
存储格式
═══════════════════════════════════════════════════════════════════
    .vec/vectors.npy   float32 [N, 1024]，第 i 行 = meta 里第 i 段的向量
    .vec/meta.json     {"model":..,"dim":..,"count":..,"items":[{...}]}
                       items[i] = {id, file, line, scope, head, text}

⚠️ 两文件必须**同时**写入且行数一致。旧实现（关键词侧）踩过的坑：索引半写
   后被判「新鲜」，于是永久静默返回 0 命中。这里用「先写临时文件再原子替换，
   且 meta 的 count 与 npy 的行数不一致就整份判损坏」来防。
   哨兵值绝不与真实取值共用通道 —— 这条在整个项目里已经踩过三次。

═══════════════════════════════════════════════════════════════════
GPU 要求
═══════════════════════════════════════════════════════════════════
嵌入一律走 Ollama（用户硬性要求：不许退到 CPU）。这里**不**做「GPU 不可用就
换 CPU」的兜底 —— 那只会让慢得不可用的路径悄悄生效，用户以为在用 GPU。
拿不到 GPU 就报错，让人看见。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import nb_config as C


class VecUnavailable(RuntimeError):
    """向量层不可用（Ollama 没起、模型没拉、GPU 拿不到…）。

    ⚠️ 独立异常类型是**故意**的：调用方要能把「向量层挂了」与「向量层正常但
    没有近邻」分开处理。若两者都表现为「返回空列表」，就会把「检索不到」
    误判成「没记过」，这类静默降级在本项目里已经造成过 96 条记录只剩 1 条
    有日期的后果。
    """


class VecCorrupt(RuntimeError):
    """向量索引损坏（npy 与 meta 不一致、维度不符、JSON 坏掉）。"""


# ── 嵌入后端 ─────────────────────────────────────────────────────

def _embed(texts: list[str], model: str | None = None) -> "list[list[float]]":
    """调 Ollama 取句向量。一次多段，走 /api/embed（批量）。"""
    import urllib.error
    import urllib.request

    model = model or C.VEC_MODEL
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{C.VEC_OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise VecUnavailable(
            f"连不上 Ollama（{C.VEC_OLLAMA_URL}）：{e}\n"
            f"  起服务: ollama serve\n  拉模型: ollama pull {model}"
        ) from e
    except TimeoutError as e:
        # ⚠️ 别把这条当"模型太大"就完事：最常见的原因是**上一个请求把服务挂住了**
        #   （单请求 token 总量过载），此时客户端超时、服务端仍在算，后面全部排队。
        #   处置是重启服务 + 调小 char_budget，不是换模型。
        raise VecUnavailable(
            f"Ollama 嵌入超时（{model}）：{len(texts)} 条 / {sum(len(t) for t in texts)} 字符。\n"
            f"  多半是单请求过载挂住了服务 → 重启：taskkill /IM ollama.exe /F 然后 ollama serve\n"
            f"  然后调小 build(char_budget=...)，默认 1800"
        ) from e

    embs = body.get("embeddings")
    if not embs:
        raise VecUnavailable(f"Ollama 返回里没有 embeddings：{str(body)[:200]}")
    return embs


def gpu_probe() -> dict:
    """GPU 占用的**结构化**状态 —— 给「必须做判断」的调用方用。

    返回 {"loaded": bool, "gpu_percent": float|None, "raw": str}

    ⚠️ 为什么不能只提供 gpu_status() 那句人话：判断「在不在 GPU 上」的人
    只能去 `in` 字符串（例如 `"CPU" in gpu`），那是**猜**：
    - "未加载（首次检索时才载入）" 里没有 "CPU"，于是被判为「没问题」；
    - 多模型共存时只报一个百分比，无法定位到 bge-m3；
    - 文案一改，所有基于字符串的判断全部静默失效。

    这是本项目的老毛病换了个位置：**拿给人看的文案去做决定**。
    所以把原始数字单独开一条通道（gpu_percent），文案只用于显示；
    拿不到就返回 None —— **None 表示「无法确认」，不等于「没问题」**，
    调用方必须显式处理（doctor 会直接判失败）。
    """
    import urllib.request

    out = {"loaded": False, "gpu_percent": None, "raw": "未知"}
    try:
        with urllib.request.urlopen(f"{C.VEC_OLLAMA_URL}/api/ps", timeout=10) as r:
            ps = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        out["raw"] = f"未知（{type(e).__name__}）"
        return out

    models = ps.get("models") or []
    if not models:
        out["raw"] = "未加载（首次检索时才载入）"
        return out

    out["loaded"] = True
    parts, pcts = [], []
    for m in models:
        size = m.get("size") or 0
        vram = m.get("size_vram") or 0
        if not size:
            continue
        pct = round(vram / size * 100, 1)
        name = m.get("name") or "?"
        parts.append(f"{name} {pct:.0f}% GPU")
        # 只把**目标嵌入模型**的百分比作为判据，避免别的模型拉高结论
        if C.VEC_MODEL.split(":")[0] in str(name):
            pcts.append(pct)
    out["raw"] = " | ".join(parts) if parts else "未加载"
    if pcts:
        out["gpu_percent"] = min(pcts)  # 取最低：任一实例掉到 CPU 都要暴露
    elif parts:
        # 目标模型没在列表里（可能被换掉了）：给不出结论，留 None 让调用方判
        out["gpu_percent"] = None
    return out


def gpu_status() -> str:
    """当前 Ollama 是否真的在用 GPU（**仅用于显示**）。

    ⚠️ 不要拿这个字符串做判断 —— 判断请用 gpu_probe() 的 gpu_percent。
    字符串里出现 "未加载" 时它既不是 CPU 也不是 GPU，用 `in` 去猜必然出错。
    """
    return gpu_probe()["raw"]


# ── 段落切分（与关键词索引共用同一套逻辑，保证两层的单元一致） ──

def collect_items() -> list[dict]:
    """把 core / stream / ws 切成待嵌入的单元。

    复用 nb_index.collect_docs()：**两层必须切得一样**，否则同一个问题在两边
    命中不同粒度的段落，合并结果时会出现重复或缺漏。
    字段名对齐 nb_index 的既有约定（title/heading/id），不要在这里另起一套。
    """
    import nb_index

    docs = nb_index.collect_docs()
    items = []
    for d in docs:
        items.append({
            "id": d["id"],
            "file": d["file"],
            "scope": d["scope"],
            "title": d.get("title", ""),
            "head": d.get("heading", ""),
            "text": d["text"],
        })
    return items


def embed_text_of(item: dict) -> str:
    """喂给嵌入模型的文本：标题 + 正文。

    标题单独喂是必要的：`ws/记忆系统.md` 里的段标题（如「向量层与关键词索引
    零耦合」）本身就是最浓缩的语义,丢标题会让这些段落只能靠正文长句被召回。
    截断到 4000 字符：bge-m3 有效上下文有限，超长段落尾部是噪声。
    """
    head = (item.get("head") or "").strip()
    text = (item.get("text") or "").strip()
    s = f"{head}\n{text}" if head else text
    return s[:4000]


# ── 建索引 ────────────────────────────────────────────────────────

def build(batch: int = 16, verbose: bool = True, char_budget: int = 1800) -> dict:
    """全量重建向量索引。返回统计信息。

    ⚠️ **为什么要按字符预算分组，而不是按条数**（2026-10-13 实测）：
    Ollama 的 `/api/embed` 在**单请求 token 总量过大时不报错、而是挂住** ——
    客户端超时后它还在算，后续请求排队同样超时，整个嵌入通道看起来"死了"。
    旧实现 `batch=16` × 每条 4000 字符 = 单请求可达 3 万 token（bge-m3 上限才 8192）。
    实测：1 条 4000 字符 ✅ 0.3s；4 条各 4000 字符 ❌ 挂死 75s+；
    而 16 条**短**文本（合计 ~1200 字符）✅ 0.1~1.5s —— **瓶颈是字符总量，不是条数**。
    所以这里按累计字符数开组，条数只作为上限兜底。
    """
    import numpy as np

    C.ensure_dirs()
    items = collect_items()
    if not items:
        raise VecCorrupt("没有任何可嵌入的段落：core/stream/ws 都是空的？")

    # 按字符预算分组（单条超预算时自成一组，不会把别的挤掉）
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for it in items:
        t = embed_text_of(it)
        if cur and (cur_len + len(t) > char_budget or len(cur) >= batch * 4):
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(t)
        cur_len += len(t)
    if cur:
        groups.append(cur)

    vecs: list[list[float]] = []
    t0 = time.time()
    for gi, chunk in enumerate(groups):
        embs = _embed(chunk)
        if len(embs) != len(chunk):
            raise VecCorrupt(f"嵌入条数不匹配：要 {len(chunk)} 条，返回 {len(embs)} 条")
        vecs.extend(embs)
        if verbose:
            done = sum(len(g) for g in groups[:gi + 1])
            print(f"  [{done}/{len(items)}] {time.time() - t0:.1f}s", end="\r", flush=True)
    if verbose:
        print()

    arr = np.asarray(vecs, dtype="float32")
    # L2 归一化：之后余弦相似度退化成点积，检索快且省内存
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    meta = {
        "model": C.VEC_MODEL,
        "dim": int(arr.shape[1]),
        "count": int(arr.shape[0]),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "items": items,
    }

    # 原子替换：先写 .tmp 再 replace，避免半写文件被判「新鲜」。
    # ⚠️ 临时名不能用 with_suffix(".npy.tmp")：np.save() 见到路径末尾不是 .npy
    #    会**自动补上 .npy**，于是文件落在 `vectors.npy.tmp.npy`，而 replace 找的是
    #    `vectors.npy.tmp` → FileNotFoundError（实测踩过，2026-10-06）。
    #    所以临时名自己写成 .npy 结尾，别让 numpy 替我们决定。
    C.VEC_DIR.mkdir(parents=True, exist_ok=True)
    tmp_npy = C.VEC_DIR / "vectors.tmp.npy"
    tmp_meta = C.VEC_DIR / "meta.tmp.json"
    np.save(str(tmp_npy), arr)
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    tmp_npy.replace(C.VEC_INDEX)
    tmp_meta.replace(C.VEC_META)

    return {
        "count": int(arr.shape[0]),
        "dim": int(arr.shape[1]),
        "elapsed_s": meta["elapsed_s"],
        "bytes": C.VEC_INDEX.stat().st_size,
    }


# ── 载入（损坏必须能被分辨出来） ──────────────────────────────────

def load() -> tuple["object", dict]:
    """返回 (矩阵, meta)。缺失抛 VecUnavailable，损坏抛 VecCorrupt。

    ⚠️ 「不存在」和「损坏」必须走**不同**的异常：前者该重建，后者也该重建但
    更该被人知道（说明上次写入被打断了）。共用一种「返回 None」会让两者
    都表现为「没结果」。
    """
    import numpy as np

    if not C.VEC_INDEX.exists() or not C.VEC_META.exists():
        raise VecUnavailable("向量索引不存在，先建：py nb_vec.py build")
    try:
        meta = json.loads(C.VEC_META.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise VecCorrupt(f"meta.json 读不出：{e}") from e

    for k in ("model", "dim", "count", "items"):
        if k not in meta:
            raise VecCorrupt(f"meta.json 缺字段 {k!r}（半写文件？）")
    if not meta["items"]:
        raise VecCorrupt("meta.items 为空")

    try:
        arr = np.load(C.VEC_INDEX)
    except (OSError, ValueError) as e:
        raise VecCorrupt(f"vectors.npy 读不出：{e}") from e

    if arr.ndim != 2:
        raise VecCorrupt(f"vectors.npy 维度不对：{arr.shape}（应为二维）")
    if arr.shape[0] != len(meta["items"]):
        # 最危险的半写状态：npy 换了、meta 没换（或反之）→ 行与元数据错位，
        # 检索会返回**内容对不上的段落**，比直接报错危险得多。
        raise VecCorrupt(
            f"行数不一致：npy {arr.shape[0]} 行 vs meta {len(meta['items'])} 条"
            f" → 行与元数据已错位，必须重建"
        )
    if arr.shape[1] != meta["dim"]:
        raise VecCorrupt(f"维度不一致：npy {arr.shape[1]} vs meta {meta['dim']}")
    return arr, meta


def load_or_none():
    """给「不想区分缺失/损坏」的调用方用；两种都返回 None。"""
    try:
        return load()
    except (VecUnavailable, VecCorrupt):
        return None


def stats() -> str:
    """一行摘要，给 nb_config.summary() 用。不允许抛异常。"""
    if not C.VEC_ENABLED:
        return "关闭"
    if not C.VEC_INDEX.exists() or not C.VEC_META.exists():
        return f"未建（{C.VEC_MODEL}）→ py nb_vec.py build"
    try:
        arr, meta = load()
    except VecCorrupt as e:
        return f"🚨 损坏：{e}"
    except VecUnavailable as e:
        return f"不可用：{e}"
    age_days = (time.time() - C.VEC_INDEX.stat().st_mtime) / 86400
    return (f"{meta['model']} {arr.shape[0]} 段 × {arr.shape[1]} 维"
            f"（建 {age_days:.1f} 天前）| {gpu_status()}")


def stale() -> bool:
    """向量索引是否需要重建（数据比索引新，或索引不存在/损坏）。"""
    if not C.VEC_INDEX.exists() or not C.VEC_META.exists():
        return True
    try:
        load()
    except (VecUnavailable, VecCorrupt):
        return True
    try:
        m = C.VEC_INDEX.stat().st_mtime
    except OSError:
        return True
    sources = [C.CORE, C.STREAM]
    if C.WS_DIR.exists():
        sources += list(C.WS_DIR.glob("*.md"))
    for p in sources:
        try:
            if p.exists() and p.stat().st_mtime > m:
                return True
        except OSError:
            continue
    return False


# ── 检索 ──────────────────────────────────────────────────────────

def search(query: str, top_k: int | None = None,
           min_score: float | None = None, scope: str | None = None) -> dict:
    """语义检索。返回 {query, hits, bytes, model, note}。

    与 nb_search.search() 的返回形状**刻意保持一致**（query/hits/bytes/note），
    这样上层合并两个通道的结果时不必写两套解析。hits 每项含 score（余弦，
    [-1,1]，越高越近）与 source="vec" 以区分来源。
    """
    import numpy as np

    top_k = C.VEC_TOP_K if top_k is None else top_k
    min_score = C.VEC_MIN_SCORE if min_score is None else min_score
    q = (query or "").strip()
    if not q:
        return {"query": "", "hits": [], "bytes": 0, "model": C.VEC_MODEL, "note": "查询为空"}

    arr, meta = load()
    qv = np.asarray(_embed([q])[0], dtype="float32")
    n = np.linalg.norm(qv)
    if n == 0:
        raise VecCorrupt("query 向量为零向量")
    qv = qv / n

    if qv.shape[0] != arr.shape[1]:
        raise VecCorrupt(
            f"query 维度 {qv.shape[0]} 与索引维度 {arr.shape[1]} 不符"
            f" → 嵌入模型换了？重建：py nb_vec.py build"
        )

    # 矩阵已 L2 归一化 → 点积即余弦
    sims = arr @ qv
    order = np.argsort(-sims)
    hits = []
    for i in order:
        score = float(sims[i])
        if score < min_score:
            break
        it = meta["items"][i]
        # ⚠️ 默认排除 core，与 nb_search.search() 的策略**保持一致**：
        # core.md 是每轮全量注入的，再把它召回一遍等于同一内容占两份预算，
        # 还会挤掉真正的检索名额。两边策略不一致曾导致同一个查询在两条通道
        # 上命中范围不同（core 7/508 段），合并去重时看着像重复却没有键可去。
        if scope is None:
            if it.get("scope") == "core":
                continue
        elif it.get("scope") != scope:
            continue
        hits.append({
            "score": round(score, 4),
            "source": "vec",
            "file": it["file"],
            "id": it.get("id", ""),
            "scope": it["scope"],
            "head": it.get("head", ""),
            "text": it["text"],
        })
        if len(hits) >= top_k:
            break

    total = sum(len(h["text"].encode("utf-8")) + len(h.get("head", "").encode("utf-8")) for h in hits)
    note = "" if hits else f"无近邻（阈值 {min_score}）"
    return {"query": q, "hits": hits, "bytes": total, "model": meta["model"], "note": note}


def render(res: dict) -> str:
    """人读格式。"""
    if not res["hits"]:
        return f"[向量] {res['note']}"
    out = [f"[向量] '{res['query']}' → {len(res['hits'])} 段（{res['model']}，{res['bytes']}B）"]
    for h in res["hits"]:
        out.append(f"  {h['score']:.3f}  {h['file']}  {h.get('head','')[:46]}")
    return "\n".join(out)


# ── CLI ───────────────────────────────────────────────────────────

def _main() -> int:
    import sys

    args = sys.argv[1:]
    if not args:
        print("用法: nb_vec.py build | query <文本> [top_k] | stats | scopes")
        return 2
    cmd = args[0]

    if cmd == "build":
        r = build()
        print(f"向量索引: {r['count']} 段 × {r['dim']} 维，"
              f"{r['elapsed_s']}s，{r['bytes'] / 1024 / 1024:.1f}MB → {C.VEC_INDEX}")
        print(f"  GPU: {gpu_status()}")
        return 0

    if cmd == "query":
        if len(args) < 2:
            print("需要查询文本")
            return 2
        k = int(args[2]) if len(args) > 2 and args[2].isdigit() else None
        print(render(search(args[1], top_k=k)))
        return 0

    if cmd == "stats":
        print(stats())
        print(f"  陈旧: {stale()}")
        print(f"  GPU : {gpu_status()}")
        return 0

    if cmd == "scopes":
        _, meta = load()
        from collections import Counter
        c = Counter(it["scope"] for it in meta["items"])
        for k, v in c.most_common():
            print(f"  {k}: {v} 段")
        return 0

    print(f"未知子命令: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
