"""DBA 统一门面 —— 任何 Agent 接记忆只需要这一个文件。

═══════════════════════════════════════════════════════════════════
用法（三行接入）
═══════════════════════════════════════════════════════════════════
    import sys; sys.path.insert(0, r"D:\\AI\\DBA\\port")
    import dba

    dba.recall("用户之前提过的代理地址")        # 检索 → 给模型看的文本
    dba.recall("...", mode="ladder")           # 三级阶梯：二次检索+时间权重，不清晰则主动再检索
    dba.remember_core("新事实", ...)           # 写核心记忆（云端模型）
    dba.remember_stream("干过什么", ...)        # 写记忆流（本地模型）
    dba.remember_ws("工作区", "细节", ...)      # 写工作区册

命令行：
    py dba.py status
    py dba.py recall "关键词" [--mode auto|kw|vec|both|ladder]
    py dba.py reindex

本文件**只做编排**，不重复实现索引/检索/组装逻辑 —— 那些在 scripts/nb_*.py 里，
由本门面统一调用。这样「Agent 接口」和「实现」不会各改一套、慢慢漂移。

═══════════════════════════════════════════════════════════════════
为什么要有门面层（而不是让每个 Agent 直接 import nb_search）
═══════════════════════════════════════════════════════════════════
1. **路径自举**：接入方不该关心 scripts/ 在哪、索引叫什么。门面自己定位仓库根。
2. **降级语义统一**：两个检索通道（关键词 / 向量）可能单边不可用。门面负责把
   「一个挂了另一个照常」这件事做对，而不是让每个接入方各写一遍 try/except。
   ⚠️ 这里是本项目反复出错的点：**降级必须是「明确说出来的降级」**。
   若向量层挂了就静默返回关键词结果、连个标记都不留，使用者会以为「语义检索
   也说没记过」，进而重复问同一件事或以为记忆丢了。所以结果里永远有
   `degraded` 字段说明哪条通道没跑。
3. **写入者约束**：三层各有指定写入者（core/ws 归云端模型，stream 归本地模型）。
   门面把这个约束做成函数名，而不是靠人记文档。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── 路径自举 ──────────────────────────────────────────────────────
# 本文件位于 <repo>/port/dba.py，scripts/ 与 memory/ 都在上一级。
ROOT = Path(os.environ.get("DBA_HOME") or Path(__file__).resolve().parent.parent).resolve()
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

__all__ = [
    "recall", "recall_both", "remember_core", "remember_stream", "remember_ws",
    "context", "status", "reindex", "ROOT", "ingest_turn", "recall_turns",
]


# ── 读取 ──────────────────────────────────────────────────────────

_VALID_SCOPES = (None, "core", "stream", "ws")


def _validate(top_k: int, scope: str | None) -> None:
    """入口参数校验 —— 参数写错必须当场炸，不能让调用方拿着错结果走。

    两个都踩过：
    - `top_k=0/-1` 各通道行为**不一致**（关键词返空、向量可能返 1 条），
      调用方会拿到自相矛盾的结果；
    - `scope="wss"` 这类拼错会**静默返回空命中**，看起来像「记忆里没这事」，
      实际是参数错。这两者在排查时方向完全相反，不能同形。
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k 必须是 >=1 的整数，得到 {top_k!r}")
    if scope not in _VALID_SCOPES:
        raise ValueError(
            f"未知 scope={scope!r}；只接受 None（全部，排除 core）| 'core' | 'stream' | 'ws'"
        )


def _ladder(query: str, top_k: int, scope: str | None) -> dict:
    """三级检索阶梯（关键词侧）。返回结构与 _kw_only 一致，另加时间/放宽信息。"""
    try:
        import nb_search
    except Exception as e:
        return {
            "query": query, "hits": [], "text": "", "channels": ["kw"],
            "degraded": [f"kw:ImportError: {type(e).__name__}: {e}"],
            "bytes": 0, "narrowed": False, "relaxed": [], "weak": True,
            "weak_reasons": ["nb_search 导入失败"], "note": "",
        }
    r = nb_search.search_ladder(query, top_k=top_k, scope=scope)
    hits = []
    for h in r["hits"]:
        h = dict(h)
        h.setdefault("source", "kw")
        if "head" not in h:
            h["head"] = h.get("heading", "")
        hits.append(h)
    degraded = []
    if r.get("error"):
        degraded.append(f"kw:{r['error']}")
    out = {
        "query": query,
        "hits": hits,
        "text": _render(hits),
        "channels": ["kw"],
        "degraded": degraded,
        "bytes": sum(len((h.get("text") or "").encode("utf-8")) for h in hits),
        "narrowed": r.get("narrowed", False),
        "note": r.get("note", ""),
        # ⚠️ 这三个字段是阶梯模式的**意义所在**，必须原样透出去：
        #   relaxed       —— 放宽过哪些条件（空列表 = 一次就查到了）
        #   weak          —— 放宽后是否仍不清晰（云端模型据此决定自己再查）
        #   time          —— 时间权重是否真的生效了（含「等于没生效」的情形）
        # 埋在 nb_search 里不往外传，等于让调用方无法判断「这条结果可信吗」。
        "relaxed": r.get("relaxed", []),
        "relax_notes": r.get("relax_notes", []),
        "weak": r.get("weak", False),
        "weak_reasons": r.get("weak_reasons", []),
    }
    if r.get("time"):
        out["time"] = r["time"]
    # ⚠️ 子串扩展**必须透出**。它是「数字片段查到了完整 ID」的唯一证据 ——
    # 只用 note 说一句是不够的：调用方要能**判断**这次命中是精确的还是擦边的，
    # 否则程序化调用方（不会读中文 note）会把片段匹配当成精确命中。
    # 用 `setdefault` 式写法保证**字段恒在**：上面那句「四种 mode 一律齐全」
    # 是承诺过的契约，缺字段会让调用方 `KeyError`（实测 mode="kw" 就漏过一次）。
    # 空 dict = 这次没用到扩展。
    out["expansion"] = r.get("expansion") or {}
    return out


def recall(query: str, top_k: int = 5, scope: str | None = None,
           mode: str = "auto") -> dict:
    """检索记忆。

    返回字段（**四种 mode 一律齐全**，调用方不必按 mode 分支取值）：
        query      原查询
        hits       [{id,file,scope,head,text,score,source}]  source ∈ {"kw","vec"}
        text       渲染好的、可直接塞进 prompt 的文本
        channels   **实际被查询过**的通道，如 ["kw"] 或 ["kw","vec"]
        degraded   被查询过但**失败**的通道，形如 ["vec:VecUnavailable: ..."]
        bytes      命中文本的字节数
        expansion  子串扩展记录 {原词: [扩展出的词]}；空 dict = 未用到扩展
                   ⚠️ 有了它，调用方才能区分「精确命中」与「片段擦边命中」。
                   典型场景：查 `3055` 召回 `202301230000` —— 结果是**对的**，
                   但性质是片段匹配。程序化调用方必须看这个字段，
                   不能只依赖中文 `note`。

    ⚠️ `channels` 与 `degraded` 是**两个不同的问题**，不要混：
        channels  回答了「这轮问了谁」——用于判断结论的可信范围
        degraded  回答了「谁答不上来」——用于判断是否降级
    所以 **`degraded == []` 不能证明向量层健康**，只能证明「这轮没有通道失败」。
    `auto` 模式在关键词命中充足时**根本不会去问向量层**，此时 channels == ["kw"]、
    degraded 为空 —— 这是正常的省算力行为，不是故障，但也**不构成健康检查**。
    要巡检向量层，必须显式用 mode="vec" 或 "both"。
    （曾把这两件事混为一谈：只看 degraded 为空就以为一切正常，而实际上
     auto 模式压根没调用向量层 —— 健康探针自己不说话，是另一种静默降级。）

    mode:
        "auto"  = 关键词优先，命中不足时补向量（默认；快且省 GPU）
        "kw"    = 只走关键词（零依赖、零 GPU，离线可用）
        "vec"   = 只走向量（语义，同义改写好用；需 Ollama + GPU）
        "both"  = 两条都跑，按排名交错合并
        "ladder"= 三级检索阶梯：一次检索 → 内容太多则二次检索（收窄+时间权重）
                  → 结果不清晰则主动再检索（逐级放宽，最多 2 次）

    `ladder` 与 `auto` 的区别只在**要不要主动再检索**：两者都走关键词优先，
    但 `ladder` 在结果「不清晰」时会自己放宽条件重查一遍，并把每次放宽
    如实写进 `relaxed`。`auto` 不重查。
    """
    _validate(top_k, scope)
    if mode == "kw":
        return _kw_only(query, top_k, scope)
    if mode == "vec":
        return _vec_only(query, top_k, scope)
    if mode == "both":
        return recall_both(query, top_k, scope)
    if mode == "ladder":
        return _ladder(query, top_k, scope)
    if mode != "auto":
        # ⚠️ 不许静默落回 auto：调用方把 "vecc" 拼错时会**得到一份看似合理
        # 的结果**（其实是关键词检索），而它以为在用语义检索。这正是本项目
        # 最忌讳的静默降级 —— 参数写错必须当场炸，而不是换个策略继续跑。
        raise ValueError(
            f"未知 mode={mode!r}；只接受 'auto' | 'kw' | 'vec' | 'both' | 'ladder'"
        )

    # auto：先关键词，明显不足再补向量
    kw = _kw_only(query, top_k, scope)
    if len(kw["hits"]) >= max(3, top_k // 2):
        # 关键词已够用 → 不问向量层。channels 如实写 ["kw"]，让调用方知道
        # 向量这一路**没被问过**（而不是「问过且正常」）。
        kw["channels"] = ["kw"]
        return kw

    vec = _vec_only(query, top_k, scope)
    merged = _merge(kw["hits"], vec["hits"], top_k)
    # ⚠️ 两条通道的失败都要收：旧写法只收 vec 的错，于是「关键词自己崩了、
    # 向量也没补上」这种情形返回 degraded=[]，看起来一切正常而实际什么都没有。
    degraded = list(kw.get("degraded") or []) + list(vec.get("degraded") or [])
    return {
        "query": query,
        "hits": merged,
        "text": _render(merged),
        "channels": ["kw", "vec"],
        "degraded": degraded,
        "bytes": sum(len((h.get("text") or "").encode("utf-8")) for h in merged),
        "note": kw.get("note", ""),
        "expansion": kw.get("expansion") or {},
        # `narrowed` 在 kw 分支有、这里原缺 → 同一件事在两种 mode 下字段名不同，
        # 调用方按 mode 分支取值。统一补上（向量通道不收窄，恒为 False）。
        "narrowed": kw.get("narrowed", False),
    }


def recall_both(query: str, top_k: int = 5, scope: str | None = None) -> dict:
    """两条通道都跑并合并。巡检用这个（或 mode="vec"），别用 auto。"""
    _validate(top_k, scope)
    kw = _kw_only(query, top_k, scope)
    vec = _vec_only(query, top_k, scope)
    merged = _merge(kw["hits"], vec["hits"], top_k)
    degraded = list(kw.get("degraded") or []) + list(vec.get("degraded") or [])
    return {
        "query": query,
        "hits": merged,
        "text": _render(merged),
        "channels": ["kw", "vec"],
        "degraded": degraded,
        "bytes": sum(len((h.get("text") or "").encode("utf-8")) for h in merged),
        "note": "",
        "expansion": kw.get("expansion") or {},
        "narrowed": kw.get("narrowed", False),
    }


def _kw_only(query: str, top_k: int, scope: str | None) -> dict:
    """关键词通道。**无论成败都返回完整的字段集**（channels/degraded/hits/text）。

    ⚠️ 不能直接把 nb_search.search() 的返回原样交出去：它没有 channels 字段，
    于是 `mode="kw"` 下调用方取 r["channels"] 会 KeyError —— 而文档又承诺这个
    字段存在（2026-10-06 由文档编写者实测发现）。门面的职责就是抹平底层的
    字段差异，不能把差异透传给调用方。
    """
    try:
        import nb_search
        r = nb_search.search(query, top_k=top_k, scope=scope)
        hits = []
        for h in r["hits"]:
            h = dict(h)
            h.setdefault("source", "kw")
            # 字段归一：关键词侧叫 heading，向量侧叫 head。
            # 两边都补上 head，调用方只认一个名字即可（heading 保留兼容）。
            if "head" not in h:
                h["head"] = h.get("heading", "")
            hits.append(h)
        # ⚠️ nb_search 用 error 字段报告**故障**（索引不存在/损坏且重建失败），
        # 与 note（说明，含「无命中」这种正常信息）分开。不把 error 转成
        # degraded 的话，「检索此刻不可用」会呈现为「空命中 + degraded=[]」，
        # 与「记忆里真的没这件事」完全同形 —— 2026-10-06 复核发现的泄漏点。
        degraded = []
        if r.get("error"):
            degraded.append(f"kw:{r['error']}")
        return {
            "query": query,
            "hits": hits,
            "text": _render(hits),
            "channels": ["kw"],
            "degraded": degraded,
            "bytes": sum(len((h.get("text") or "").encode("utf-8")) for h in hits),
            "narrowed": r.get("narrowed", False),
            "note": r.get("note", ""),
            # 子串扩展的如实记录（空 dict = 未用到）。这个分支同样必须带 ——
            # 实测漏掉它会让 `mode="kw"` 直接 KeyError，而 kw 恰恰是最常用的通道。
            "expansion": r.get("expansion", {}),
        }
    except Exception as e:
        # 关键词层是纯 Python 零依赖，它挂说明代码/索引真坏了 —— 必须说出来
        return {"query": query, "hits": [], "text": "", "channels": ["kw"],
                "degraded": [f"kw:{type(e).__name__}: {e}"], "bytes": 0,
                "narrowed": False, "error": f"{type(e).__name__}: {e}",
                "expansion": {}, "note": "关键词通道不可用"}


def _vec_only(query: str, top_k: int, scope: str | None) -> dict:
    """向量通道。同样保证完整字段集。"""
    try:
        import nb_vec
        r = nb_vec.search(query, top_k=top_k, scope=scope)
        hits = [dict(h) for h in r["hits"]]
        return {
            "query": query,
            "hits": hits,
            "text": _render(hits),
            "channels": ["vec"],
            "degraded": [],
            "bytes": sum(len((h.get("text") or "").encode("utf-8")) for h in hits),
            "model": r.get("model", ""),
            "expansion": {},
            # 向量通道不做行窗口收窄，恒 False；补上是为了让字段集在各 mode 间一致。
            "narrowed": False,
            "note": r.get("note", ""),
        }
    except Exception as e:
        return {"query": query, "hits": [], "text": "", "channels": ["vec"],
                "degraded": [f"vec:{type(e).__name__}: {e}"], "bytes": 0,
                "error": f"{type(e).__name__}: {e}", "expansion": {},
                "narrowed": False, "note": "向量通道不可用"}


def _merge(kw_hits: list[dict], vec_hits: list[dict], top_k: int) -> list[dict]:
    """交错合并两条通道的结果，按内容去重。

    为什么交错而不是拼接：拼接会让关键词通道（每次都能返回满 top_k）把名额
    全部占掉，向量通道的语义命中永远看不到。
    为什么去重：同一段落很可能两条通道都召回，不去重的话 top_k 会被同一段
    的副本占满 —— 看起来命中很多，信息量却没增加。
    """
    seen: set = set()
    out: list[dict] = []
    i = j = 0
    while len(out) < top_k and (i < len(kw_hits) or j < len(vec_hits)):
        took = False
        if i < len(kw_hits):
            h = dict(kw_hits[i]); i += 1; took = True
            if _key(h) not in seen:
                seen.add(_key(h)); out.append(h)
        if len(out) >= top_k:
            break
        if j < len(vec_hits):
            h = dict(vec_hits[j]); j += 1; took = True
            if _key(h) not in seen:
                seen.add(_key(h)); out.append(h)
        if not took:
            break
    return out


def _key(h: dict) -> tuple:
    """去重键：优先用段落 id，退化为文本前 80 字符。"""
    return (h.get("id") or (h.get("text") or "")[:80],)


def _render(hits: list[dict]) -> str:
    """把命中渲染成给模型读的文本。"""
    if not hits:
        return ""
    out = []
    for h in hits:
        tag = h.get("source", "kw")
        src = h.get("file", "?")
        head = (h.get("head") or h.get("heading") or "").strip()
        body = (h.get("text") or "").strip()
        score = h.get("score")
        label = f"[{tag} {score}]" if score is not None else f"[{tag}]"
        out.append(f"{label} {src} :: {head}\n{body}")
    return "\n\n".join(out)


# ── 写入（按写入者分工命名） ──────────────────────────────────────

def remember_core(text: str, title: str = "追加") -> str:
    """写核心记忆。**云端模型**用。

    核心记忆每轮全量注入，所以只放「硬事实」：身份、路径、配置值、代理地址、
    待办、反复踩的坑。不放流水账（那是 stream）、不放长细节（那是 ws）。

    ⚠️ title 的默认值是「追加」，与 nb_write.append_core 保持一致 —— 门面
    刻意**不**在这里另起一套参数名（曾用 heading 导致 TypeError）。
    门面只做编排，签名跟着实现走，不要在这里发明第二套命名。
    """
    import nb_write
    return nb_write.append_core(text, title=title)


def remember_stream(text: str, tag: str = "", at: str | None = None) -> str:
    """写记忆流。**本地模型**用。只记「何时干过什么」，一行一条，不写细节。

    写入有内容闸门（拒收对话残片/系统报错/空话），细则请走 remember_ws。
    """
    import nb_write
    return nb_write.append_stream(text, at=at, tag=tag)


def remember_ws(area: str, text: str, title: str | None = None,
                at: str | None = None) -> str:
    """写工作区册。**云端模型**用。存细节、结论、报错与修法。"""
    import nb_write
    return nb_write.append_ws(area, text, title=title, at=at)


# ── 联合轮次记忆（本地模型写入；云端按需读取） ────────────────

def ingest_turn(event: dict, spawn: bool = False) -> dict:
    """Persist a turn without starting a console-capable background worker."""
    import nb_turn
    key = nb_turn.Pipeline().enqueue(event)
    result = {"persisted": True, "key": key}
    if spawn:
        try:
            nb_turn.Pipeline().spawn()
            result["worker"] = "spawned"
        except Exception as exc:
            result["worker_error"] = f"{type(exc).__name__}: {exc}"
    return result


def recall_turns(query: str, top_k: int = 5) -> dict:
    """Semantic recall of complete sanitized user+assistant turn archives."""
    import nb_turn
    return nb_turn.Pipeline().recall(query, top_k)


# ── 组装（给注入用） ──────────────────────────────────────────────

def context(query: str = "", top_k: int = 5) -> dict:
    """产出可直接注入 prompt 的两段文本。

    返回 {core_text, hits_text, text, bytes, core_kb, budget_warning, ...}
    ⚠️ core_text 与 hits_text **必须分两段注入**，别自己拼成一段：
       core 几天才变一次（可缓存前缀），hits 每轮都变（必须排最末）。
       拼成一段会让检索一命中就把 system prompt 后半段的缓存全废掉
       （用户此前实测：命中率从 92.7% 掉到 0%）。
    """
    import nb_assemble
    return nb_assemble.assemble(query, top_k=top_k)


def reindex(vectors: bool = True, keywords: bool = True) -> dict:
    """重建索引。"""
    out = {}
    if keywords:
        import nb_write
        out["kw"] = nb_write.reindex()
    if vectors:
        import nb_vec
        out["vec"] = nb_vec.build(verbose=False)
    return out


# ── 状态 ──────────────────────────────────────────────────────────

def status() -> str:
    """人读状态：核心记忆大小、各层文件、两条检索通道健康度、GPU。"""
    import nb_config as C
    lines = [C.summary(), "", "检索通道:"]
    try:
        import nb_search  # noqa: F401
        lines.append(f"  kw  : 可用（{C.INDEX.name} {C.kb(C.INDEX):.1f}KB）")
    except Exception as e:
        lines.append(f"  kw  : 🚨 {type(e).__name__}: {e}")
    try:
        import nb_vec
        lines.append(f"  vec : {nb_vec.stats()}")
    except Exception as e:
        lines.append(f"  vec : 🚨 {type(e).__name__}: {e}")
    lines.append("")
    lines.append("写入分工: core/ws ← 云端模型 | stream ← 本地模型")
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(status())
    elif args[0] == "status":
        print(status())
    elif args[0] == "recall":
        rest = args[1:]
        # ⚠️ `--mode X` 必须**从查询词里摘掉**，不能只判断「在不在」。
        # 只判断不移除的话，查询词会变成 '关键词 --mode ladder'，
        # tokenize 会把 mode/ladder 也当查询词喂进 BM25 —— 结果看着像对的
        # （确实有命中），但排序被凭空多出的词污染，而且很安静。
        # nb_search.py 踩过同一个坑（--ladder），这里照同一规矩办。
        # ⚠️ 而且必须处理**重复出现**：sol6.1 实测 `recall --mode kw --mode`
        # 只摘掉第一组，剩下的 `--mode` 被当成查询词喂进检索并返回了结果
        # （退出码 0，看起来完全正常）。悬空的选项必须报错，不能悄悄变成搜索词。
        mode = "both"
        while "--mode" in rest:
            i = rest.index("--mode")
            if i + 1 >= len(rest):
                print("--mode 需要一个取值（auto/kw/vec/both/ladder）", file=sys.stderr)
                sys.exit(2)
            mode = rest[i + 1]
            del rest[i:i + 2]
        # 摘完若还残留 `--` 开头的悬空选项（如 `recall --mode kw --mode` 里
        # 最后那个被当成取值的 `--mode`），说明参数本身就畸形。
        for a in rest:
            if a.startswith("--"):
                print(f"无法识别的选项或悬空的 --mode 取值: {a!r}", file=sys.stderr)
                sys.exit(2)
        if not rest or not " ".join(rest).strip():
            print("recall 需要一个查询词", file=sys.stderr)
            sys.exit(2)
        try:
            r = recall(" ".join(rest), mode=mode)
        except ValueError as e:
            print(f"参数错误: {e}", file=sys.stderr)
            sys.exit(2)
        print(f"模式 {mode}  通道 {r['channels']}  降级 {r['degraded'] or '无'}  {r['bytes']}B")
        # 阶梯模式专有的信息：放宽了什么、还清不清晰、时间权重生效没
        if mode == "ladder":
            print(f"放宽 {r['relaxed'] or '无（一次就查到）'}  仍不清晰 {r['weak']}")
            # ⚠️ 「试过但没采纳」的放宽也要显示：否则系统动过手而使用者看不见，
            # 会以为「一次就查到」。
            if r.get("relax_notes"):
                print(f"  放宽未奏效: {r['relax_notes']}")
            if r["weak"]:
                print(f"  原因: {r['weak_reasons']}")
            t = r.get("time")
            if t:
                if t.get("span_days"):
                    print(f"  时间权重: 跨度 {t['span_days']} 天，"
                          f"提升 {t['promoted']} 条，中性权重 {t['neutral_weight']}")
                else:
                    print("  时间权重: 未生效（语料无日期跨度）")
        print(r["text"] or "(无命中)")
    elif args[0] == "reindex":
        print(reindex())
    else:
        print(__doc__)
