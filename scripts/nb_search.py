"""记事本关键词检索 — 三级检索阶梯。

检索只作用于 stream.md（记忆流）与 ws/*.md（工作区册）；
core.md（核心记忆）是全量注入的，**默认排除在检索之外**，
避免同一份内容既全量注入又重复召回、白占注入预算。
（确需检索核心记忆时显式 --scope core。）

三级检索（用户 2026-10-06 定的规则：本地模型先检索 → 内容太多则二次详细检索
（加时间权重，越近权重越大）→ 云端模型觉得不清晰可主动再次检索）：

    ① 一次检索：查询 → BM25 top_k 段，**纯相关度，不掺时间**。
       为什么第一级不掺时间：它决定「谁能进候选集」。掺了时间，同一个查询
       在不同日子会给出不同候选集，检索不可复现 —— 排查「上次为什么召回不到」
       就无从下手。

    ② 二次检索：命中总量 > RETRIEVAL_MAX_KB（=「内容太多」）时触发。
       两件事同时做：
         · 收窄：把命中段落切成行窗口，只返回含查询词的行 ± NARROW_WINDOW 行，
           把注入量压进预算；
         · **时间权重重排**：候选池放大 TIME_POOL_FACTOR 倍，按
           相关度 × 时间权重 重排，让新记忆在预算裁剪中活下来。
       权重 = 0.5 ** (距最新天数 / 半衰期)，**下限 TIME_DECAY_MIN_WEIGHT**。
       无日期段落取**中性权重**（有日期段权重的中位数）—— 「不知道」既不加分
       也不减分；不是 1.0（那会假装它最新），也不是最小值（那会假装它最旧）。

    ③ 主动再检索（`--ladder`，或门面 mode="ladder"）：结果**不清晰**时逐级放宽。
       「不清晰」落成三条可计算的硬信号（见 _looks_weak）：0 命中 / 最高分过低 /
       命中段落与查询无共享词（仅二元组擦边命中）。逐级放宽最多 2 次：
       放开 scope → top_k 翻倍。每次放宽都记进 `relaxed`。

命令行：
    py nb_search.py "查询词" [更多查询词...] --top-k N
    py nb_search.py "查询词" --scope ws        # 只查工作区
    py nb_search.py "查询词" --ladder          # 三级阶梯（含主动再检索）
    py nb_search.py "查询词" --json            # 输出 JSON（供程序消费）

⚠️ top_k 只能走 `--top-k N`，**不再支持** `nb_search.py "查询词" 5`：
旧写法用「最后一个位置参数是纯数字就当 top_k」判定，于是**纯数字查询被整句吃掉**——
实测 `nb_search.py "2023"` → query 变成空串，静默返回 `No results for ''`（不报错）。
学号 202301233055、Clash 端口 7897、年份 2026 全是这一类查询；
`nb_search.py "记忆架构" "2026"` 则把年份当 top_k 丢掉，查询词凭空少一个。
代价是命令变长一点，但「数字是查询还是参数」不再靠猜。

⚠️ 时间权重**失效时和正常时长得一模一样**：没有 date 字段 → 权重全 1.0 →
排序分毫不变，而检索照常返回、不报任何错。所以 `nb_doctor.py` 专门查
「索引里有多少段带日期、跨度多少天」—— 不查就没人会发现它没在工作。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_config as C
import nb_index as I


# 单条正文超预算时的截断标记。刻意写成显眼中文：静默截断记忆正文
# 会变成下一个「事后无从追查」的坑，读的人必须一眼看出这段不完整。
_TRUNC_MARK = "…（超预算已硬截断）"


def _clip_to_bytes(text: str, limit: int) -> str:
    """把单条正文硬截断到 limit 字节内，结果仍是合法 UTF-8。

    ⚠️ 不能按**字符**截断：预算单位是字节，而中文一字 3 字节，
    按字符数砍会按 3 倍超预算。
    ⚠️ 也不能直接 `text.encode()[:limit].decode()`：可能砍在一个多字节字符
    中间，产生 UnicodeDecodeError 或替换字符，所以配 errors="ignore" 丢掉残字节。
    ⚠️ 截断标记的字节数**也要计入**预算，否则截完照样超。
    """
    if limit <= 0:
        return ""
    if len(text.encode("utf-8")) <= limit:
        return text
    mark_b = len(_TRUNC_MARK.encode("utf-8"))
    if limit <= mark_b:
        # 预算连标记都放不下 → 只能硬切，不附标记
        return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
    room = limit - mark_b
    return text.encode("utf-8")[:room].decode("utf-8", errors="ignore") + _TRUNC_MARK


def _hit_lines(lines: list[str], terms: set[str]) -> set[int]:
    """哪些行含 terms 里的词（子串判定）。terms 为空返回空集。"""
    if not terms:
        return set()
    hits: set[int] = set()
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(t in low for t in terms):
            hits.add(i)
    return hits


def _narrow(text: str, query_terms: set[str], window: int = C.NARROW_WINDOW) -> str:
    """二次检索：只保留含查询词的行 ± window 行。

    ⚠️ 命中判定必须**优先用 bigram**（长度 >= 2 的词），只有 bigram 一个都不中
    才退回单字。旧实现直接拿全部词（含单字）判命中，单字于是把大量无关行拉进来：
    实测 core.md「用户」段 13 行命中 8 行，其中 **4 行仅靠单字**才中
    （query=记忆架构 时，「记」「忆」等单字散落各处），收窄基本失效——
    而收窄失效正是「单条超预算压不回去」的直接原因（行行命中则收窄率 100%，
    实测 120 行全命中的文档收窄后仍是 100% 体积）。
    退回单字这一级是**故意保留**的：查询本身就是单字（如「机」）时 bigram 为空集，
    此时只能靠单字，否则会一行都留不下。
    """
    lines = text.split("\n")
    bigrams = {t for t in query_terms if len(t) >= 2}
    hit_lines = _hit_lines(lines, bigrams)
    if not hit_lines:
        # bigram 全不中 → 退回单字（单字查询 / 词形被拆散的情形）
        hit_lines = _hit_lines(lines, query_terms)
    if not hit_lines:
        # 词形未逐字命中（如跨行），退回首段窗口
        return "\n".join(lines[: window * 2 + 1]).strip()

    keep: set[int] = set()
    for i in hit_lines:
        keep.update(range(max(0, i - window), min(len(lines), i + window + 1)))

    out: list[str] = []
    prev = -2
    for i in sorted(keep):
        if i != prev + 1 and out:
            out.append("  …")
        out.append(lines[i])
        prev = i
    return "\n".join(out).strip()


def search(
    query: str,
    top_k: int = C.RETRIEVAL_TOP_K,
    scope: str | None = None,
    max_bytes: int | None = None,
) -> dict:
    """返回 {query, hits:[...], narrowed:bool, bytes:int, note:str, error:str}

    ⚠️ **失败必须走 error 字段，不能只写 note**（2026-10-06 复核发现）。
    note 是**说明**（给人看的、可能是"无命中"这种正常信息），error 是**故障**。
    调用方（port/dba.py 的 _kw_only）靠 error 决定是否计入 degraded。
    若只有 note，则「索引损坏且重建失败」会呈现为「空命中 + degraded=[]」，
    与「记忆里真的没这件事」**完全同形** —— 这正是本项目反复栽的坑。
    """
    result = {"query": query, "hits": [], "narrowed": False, "bytes": 0,
              "note": "", "error": ""}
    # 子串扩展的如实记录（原词 → 扩展出的词）。空 dict = 这次检索没用到扩展。
    # 放在这里而不是函数末尾，是为了让中途每一条 return 都带上它 ——
    # 否则「扩展发生了但结果里查不到证据」，又是一次静默。
    result["expansion"] = {}
    if not query or not query.strip():
        result["note"] = "查询为空"
        return result

    # ⚠️ load() 现在把「不存在」和「损坏」分成两条通道：不存在返回 None，
    # 损坏抛 IndexCorrupt。旧写法只判 `is None`，于是损坏的索引会让本函数
    # **直接抛异常**，调用方（含 port/dba.py 门面）看到的是「检索崩了」——
    # 而文件其实存在、内容只是坏的，重建即可。所以这里把损坏也接住并重建，
    # 而不是让它冒泡成崩溃。
    try:
        index = I.load()
    except I.IndexCorrupt as e:
        result["note"] = f"索引损坏（{e}），已尝试重建"
        try:
            index = I.build(verbose=False)
        except Exception as e2:
            # 重建也失败 → 这是**故障**，不只是说明。必须进 error 才不会被
            # 门面当成「正常但没命中」。
            result["note"] = f"索引损坏且重建失败：{e} / {e2}"
            result["error"] = f"IndexCorrupt: 索引损坏且重建失败（{e}; {e2}）"
            return result
    if index is None:
        # 索引**不存在**同样是故障：检索此刻不可用，而不是「没有相关记忆」。
        result["note"] = "索引不存在，请先运行 nb_rebuild.py --kw"
        result["error"] = "IndexUnavailable: 索引不存在"
        return result

    terms = I.tokenize(query)
    term_expanded: dict[str, list[str]] = {}

    # ⚠️ 数字/ASCII 词是**整词**入索引的 → 用它的任何片段都查不到（实测
    # `2023`/`3055` 均 0 命中，只有完整的 `202301233055` 能召回）。
    # 但子串扩展**不能无条件做**（第一版就是这么错的，被独立对抗复核否掉）：
    # 只要精确词有命中，扩展词就会**累加 BM25** 压过精确匹配 ——
    # 实测 `bom ing` 会把 `bom` 的精确命中挤到第 7 名，夺魁的文档里 `bom` 词频为 0。
    # 那违背了「精确匹配必须胜过片段擦边」，而检索的意义正在于此。
    # 所以设**闸门**：只有精确词一个候选都取不到，才启用扩展。
    # 这样有精确命中时扩展根本不参与，精确结果不可能被污染。
    if I.missing_terms(terms, index) and not I.exact_candidates(terms, index):
        terms, term_expanded = I.expand_terms(terms, index)

    scored = I.score(terms, index)
    if not scored:
        result["note"] = "无命中"
        return result
    if term_expanded:
        result["expansion"] = term_expanded
        # **必须如实说明**：靠片段匹配找到的，不等于精确命中。不说的话，
        # 使用者会把「子串擦边」当成「这文档里有这个号码」。
        # ⚠️ 用 `+=` 而不是 `=`：这个位置之后的代码还会**覆盖** note
        # （收窄说明、时间权重重排说明），实测那样会把这条提示整条吃掉
        # ——「扩展影响了结果却没有任何提示」，正是要避免的静默。
        pairs = "；".join(f"{k}→{'/'.join(v[:3])}{'…' if len(v) > 3 else ''}"
                        for k, v in term_expanded.items())
        result["note"] += f"含子串扩展命中（{pairs}）"

    docs = index["docs"]

    # ── 阈值：按**本次检索的最高分**归一，而不是绝对 BM25 分数 ──
    # 旧代码是 `if sc < C.RETRIEVAL_MIN_SCORE: continue`（绝对 0.8），实测为空操作：
    # 13 个真实查询的 top-5 候选无一低于 0.8，一个都没被它滤掉（详见 nb_config 注释）。
    # 归一后默认 0.0 = 关闭；一旦有人把它调大，阈值立刻有实际过滤行为。
    # 取 scored[0] 是因为 I.score() 已按分数降序排好，无需再扫一遍。
    top_score = scored[0][1]
    min_score = top_score * C.RETRIEVAL_MIN_SCORE_RATIO if C.RETRIEVAL_MIN_SCORE_RATIO > 0 else 0.0

    # ── 顺序：**先按 scope 过滤，再截断到 top_k** ──
    # ⚠️ 这两步的顺序必须是这个次序，不能交换。旧实现把 `len(picked) >= top_k` 的
    # 判据放在 scope 过滤**之后**（判据用已过滤的 picked 计数），字面上已经是对的；
    # 但那个写法极易被误读成「先截断后过滤」，而且一旦有人把 break 的判据改成
    # 全库扫描计数，`--scope core` 就会要求 core 先挤进**全库**前 top_k 才可见——
    # 实测 core 文档在全库的排名普遍在 11~103 位（记忆架构=33、模型接入=103、求职=36），
    # 那种写法下 8 个查询里 7 个直接返回 0 命中。
    # 现在把「先过滤后截断」写成两段独立代码，语义不再依赖 break 的位置。
    cands = [(docs[i], sc) for i, sc in scored if sc >= min_score]
    if scope is None:
        # 默认排除 core：它是全量注入的，再召回等于同一内容占两份预算
        cands = [(d, sc) for d, sc in cands if d.get("scope") != "core"]
    else:
        cands = [(d, sc) for d, sc in cands if d.get("scope") == scope]
    picked = cands[:top_k]

    if not picked:
        # 区分「词库里没有」与「有候选但全被归一阈值滤掉」：后者调低
        # RETRIEVAL_MIN_SCORE_RATIO 就能救回来，前者不能。两者都显示
        # 「无命中」的话，用户会以为记忆里真的没这件事。
        if scored and not cands:
            result["note"] = (f"无命中（候选全被阈值滤掉："
                              f"最高分 {top_score:.2f} × {C.RETRIEVAL_MIN_SCORE_RATIO:.2f}"
                              f" 归一阈值）")
        else:
            result["note"] = "无命中"
        return result

    limit = max_bytes if max_bytes is not None else C.RETRIEVAL_MAX_KB * 1024
    total = sum(len(d["text"].encode("utf-8")) for d, _ in picked)
    # 初检量另存：二次检索会把 picked 换成重排后的三元组，之后再从 picked
    # 反推初检量就会算错（或直接拆包失败）。
    init_bytes = total

    # —— 二次检索判定：命中总量超预算 ——
    narrow = total > limit
    result["narrowed"] = narrow

    # ═══ 二次检索：内容太多 → 放开候选池 + 叠加时间权重（越近权重越大）═══
    #
    # 规则（2026-10-06）：一次检索纯相关度；命中**内容太多**时走二次检索，
    # 此时按「相关度 × 时间权重」重排，让新记忆在预算裁剪中活下来。
    #
    # 为什么只在二次检索里施加时间权重：一次检索决定谁能进候选集，纯相关度
    # 更可复现（掺了时间，同一个查询在不同日子候选集就不同，没法排查
    # 「上次为什么召回不到」）。也正因如此，只有**确实太挤**的时候才值得
    # 让时间因素参与进来 —— 内容装得下时不需要它为谁让位。
    #
    # ⚠️ 权重必须**相乘**而不是相加：BM25 是浮动量纲（实测同一语料里
    # 中位数 1.6~5.8、最大 23.4），加上一个 (0,1] 的权重等于没加，只有相乘
    # 才是「打折」。这也和跨通道合并的规矩一致：**不同量纲不许相加**。
    time_info = None
    if narrow:
        corpus = index.get("corpus_dates") or {}
        pool_n = max(top_k * C.TIME_POOL_FACTOR, top_k + 4)
        pool = cands[:pool_n]
        before = {d["id"] for d, _ in picked}
        n_dated = n_undated = 0
        reranked = []
        for d, sc in pool:
            w, applied = I.time_weight(d, corpus)
            if applied:
                n_dated += 1
            else:
                n_undated += 1
            # ⚠️ 排序键与输出值必须是**同一个取整结果**。
            #   若用原始浮点 sc*w 排序、却输出 round(sc,2)*round(w,3)，
            #   两者会偶发地给出不同次序 —— 表现为「输出里的第 2 条分数
            #   比第 3 条低」，看的人立刻会怀疑算错。实测 12 个查询没撞上，
            #   但那只是没撞上，不是不会发生。所以先把两个因子取整，
            #   再用它们算出唯一的 rank，排序与显示共用这一个数。
            sw = round(sc, 2)
            ww = round(w, 3)
            # ⚠️ `applied` 必须一路带到输出（第六个元素）。它才是「这条用了
            # 真日期权重」的唯一权威判据 —— 早期版本在渲染处用
            # `bool(d.get("date"))` 重算，于是**日期畸形**（如 `2026-13-45`）时
            # 报 dated=True，而实际拿到的是中性权重：显示与实际矛盾。
            # sol6.1 复验抓到这一点。**能算出就是能算出，不能靠字段非空去猜。**
            reranked.append((d, sc, sw, ww, round(sw * ww, 2), applied))
        # 稳定排序：rank 相同时按纯相关度降序（原次序对相关度更忠实）
        reranked.sort(key=lambda x: (-x[4], -x[2]))
        picked = [(d, sc, sw, ww, rank, ap)
                  for d, sc, sw, ww, rank, ap in reranked[:top_k]]
        time_info = {
            # ⚠️ 区分「按日期算了权重」与「用了中性权重」：二者都让排序变化，
            # 但前者是「新的优先」，后者只是「不知道的不吃亏」。混成一个布尔值
            # 等于把两种完全不同的推理过程说成同一件事。
            "dated": n_dated,
            "undated": n_undated,
            # ⚠️ 精度必须与逐条命中的 `time_weight` 一致（都是 3 位）。
            # 索引里存 4 位（0.7236），命中里是 3 位（0.724）—— 若这里原样透出
            # 4 位，人一比对就以为无日期段的权重「不等于」中性权重（复核时确实
            # 这么报了）。同一个量在两处出现，就必须长得一样。
            "neutral_weight": (None if corpus.get("neutral_weight") is None
                               else round(float(corpus["neutral_weight"]), 3)),
            "min_weight": corpus.get("min_weight"),
            "halflife_days": C.TIME_DECAY_HALFLIFE_DAYS,
            "pool": len(pool),
            "promoted": sum(1 for d, *_ in picked if d["id"] not in before),
            "corpus_dated": corpus.get("dated", 0),
            "corpus_undated": corpus.get("undated", 0),
        }
        # ⚠️ 如实报告「跨度」：时间权重的参照点是语料内最新日期，语料只跨 1 天时
        # 所有段落权重都是 1.0 —— **它就是个空操作**。不说的话，人会以为
        # 「加了时间权重」就等于结果在按时间排序，而实际排序完全没变。
        # `_days_between` 现在对畸形日期返回 None（不再抛异常），
        # 所以这里可能拿到 None —— 那就是「跨度算不出」，不是 0 天。
        if corpus.get("oldest") and corpus.get("newest"):
            time_info["span_days"] = I._days_between(corpus["oldest"], corpus["newest"])
        else:
            time_info["span_days"] = None  # 语料无任何日期 → 更谈不上跨度
        result["time"] = time_info
    else:
        # 未触发二次检索：补齐成同一个六元组形状，下游不必分支。
        # rank 就等于纯相关度（没有时间权重参与），sw/ww/ap 是为了让形状一致；
        # ap=False 正确表达「这条没吃时间权重」。
        picked = [(d, sc, round(sc, 2), 1.0, round(sc, 2), False) for d, sc in picked]

    qset = set(terms)

    for d, sc, sw, ww, rank, ap in picked:
        text = _narrow(d["text"], qset) if narrow else d["text"]
        hit = {
            "id": d["id"],
            "file": d["file"],
            "scope": d.get("scope", ""),
            "heading": d["heading"],
            # `score` 恒为**纯 BM25 相关度**，语义不随时间权重改变 ——
            # 否则同一个分数在不同轮次含义不同，无法比较。
            "score": sw,
            "text": text,
            "truncated": narrow,
        }
        if narrow:
            # 重排后的真实依据单列出来，让「为什么它排在前面」可查。
            # ⚠️ `time_weight` 相同不代表原因相同：有日期的是「按新旧算出来的」，
            # 没日期的是「中性权重（不知道）」。`dated` 字段就是区分标志 ——
            # 不要用权重数值去猜，那正是「拿数值替代语义」的坑。
            #
            # ⚠️ rank 直接取排序时算好的那个数（见上面的排序键注释）：
            # 绝不在这里重算，否则排序依据与显示值可能不一致。
            hit["time_weight"] = ww
            hit["rank_score"] = rank
            # ⚠️ 用算出来的 `ap`，**不要**用 `bool(d.get("date"))` 重算：
            # 日期畸形时字符串非空但算不出权重，两者会给出相反的答案。
            hit["dated"] = ap
            if ap and d.get("date"):
                hit["date"] = d["date"]
        result["hits"].append(hit)

    # 收窄后仍超预算 → 按**重排后的次序**截断尾部。这一步是时间权重真正
    # 起作用的地方：被丢掉的现在是最低「相关度×权重」的那条，也就是
    # 同样相关时，**较旧的先被丢**。
    while len(result["hits"]) > 1:
        total = sum(len(h["text"].encode("utf-8")) for h in result["hits"])
        if total <= limit:
            break
        result["hits"].pop()

    # ⚠️ 循环到只剩 1 条时**永远丢不掉**，于是单条本身就超预算的情况会原样返回。
    # 旧代码在这里直接写 note「已二次检索收窄到行窗口」——**收了窄也可能仍旧超**，
    # 声称「已收窄」等于把没收窄的事实盖住（实测 max_bytes=1024 时返回 1154B，
    # 超 1.1 倍而 note 仍说已收窄）。现在：真溢出的最后一条要么硬截断进预算，
    # 要么在 note 里如实说明「单条命中超预算，未截断」——不许含糊。
    over_single = False
    total = sum(len(h["text"].encode("utf-8")) for h in result["hits"])
    if total > limit and len(result["hits"]) == 1:
        only = result["hits"][0]
        if C.RETRIEVAL_CLIP_OVERBUDGET_HIT:
            only["text"] = _clip_to_bytes(only["text"], limit)
            only["truncated"] = True
            total = len(only["text"].encode("utf-8"))
        else:
            over_single = True
    result["bytes"] = total    # note 必须描述**最终**状态，不是初始状态。踩到的坑：初始命中量超预算时
    # narrow=True，但收窄/丢弃之后产物可能已经回到预算内，此时再说「命中 X 超预算」
    # 就是假话（实测 max_bytes=2048 时最终 1154B < 2048B，note 却仍写超预算）。
    # 三档分开写：还在超 / 已收窄回预算内 / 单条超预算未截断。
    if over_single:
        # `+=`：note 里可能已有子串扩展的如实说明，赋值会吃掉它。
        result["note"] += (f"命中 {total / 1024:.1f}KB 超预算，且**单条命中本身超预算，未截断**"
                           f"（该条已收窄到行窗口，但仍大于 {limit / 1024:.1f}KB）")
    elif narrow:
        # ⚠️ 「初检量」必须用**收窄前**的池子算，但 `picked` 在二次检索里已被
        #    换成「重排后的三元组」，长度不再是初检命中数。旧写法直接拆两元组
        #    会 `ValueError: too many values to unpack`（实测触发：一个弱查询
        #    走到第 2 级放宽后崩掉）。初检量另存一份，别从 picked 反推。
        head = (f"初检 {init_bytes / 1024:.1f}KB 超预算，"
                f"已二次检索收窄到行窗口 → {total / 1024:.1f}KB")
        if total > limit:
            head = f"命中 {total / 1024:.1f}KB 仍超预算（已收窄到行窗口）"
        if len(result["hits"]) == 1 and result["hits"][0]["text"].endswith(_TRUNC_MARK):
            head += "，末条已硬截断"
        elif len(picked) > len(result["hits"]):
            head += f"（并按分数丢弃了 {len(picked) - len(result['hits'])} 条低分命中）"
        # ⚠️ 时间权重的作用必须**如实**写出来，包括「它等于没起作用」的情况。
        # 语料只跨 1 天时权重全是 1.0，排序与不加时间完全一致；若不说明，
        # 人会以为结果在按新旧排 —— 那是把「没发生」说成「发生了」。
        if time_info:
            span = time_info.get("span_days")
            if span is None:
                head += "；时间权重**未生效**（语料无任何日期）"
            elif span == 0:
                head += "；时间权重**未改变排序**（语料仅跨 0 天，权重全为 1.0）"
            else:
                head += (f"；已按时间权重重排（跨度 {span} 天，"
                         f"半衰期 {time_info['halflife_days']:.0f} 天"
                         f"，下限 {time_info['min_weight']}"
                         f"，无日期段取中性权重 {time_info['neutral_weight']}"
                         f"（本池 {time_info['undated']} 段无日期）")
                if time_info["promoted"]:
                    head += f"，{time_info['promoted']} 条新记忆升入"
                head += "）"
        # ⚠️ 用 `+=` 而非 `=`：note 里可能已经躺着**子串扩展的如实说明**，
        # 直接赋值会把它整条吃掉 —— 实测「扩展实际影响了结果，却没有任何提示」。
        # note 是给人看的说明通道，凡是**改变了结果**的事都必须留在里面。
        result["note"] += head
    return result


def _looks_weak(res: dict, query: str) -> tuple[bool, list[str]]:
    """判断「这次检索结果是不是不够用」→ (是否需再检索, 理由列表)。

    规则里的第三段：**云端模型如果觉得不清晰可以主动再次检索**。
    但「觉得不清晰」若只靠模型主观判断，它就会：要么从不触发（形同虚设），
    要么频繁触发（每次多烧一轮）。所以把「不清晰」落成**可计算的信号**：

      1. 0 命中 —— 最明确的不清晰。（单独就足以判定）
      2. 命中段落与查询**没有共享的词** —— 只靠二元组擦边命中，
         即查询的字被拆开散落在不同位置，语义上压根不挨着。
      3. 最高分过低 —— **但这一条单独不算数**，理由见下。

    ⚠️ 第 3 条为什么不单独算数（实测抓到的误判）：
      短查询的 BM25 分数**天然低** —— 查询词少 → 命中的词少 → 累加项少。
      实测 `BOM`（1 个词）最高分 5.68、`Codex`（1 个词）4.25，两者的命中
      其实**都很精准**（记忆里全是相关内容），却都会被 6.0 的阈值判成「弱」。
      那是把「词少」误读成「查不到」。所以：**只有查询词数 >=
      RETRIEVAL_WEAK_MIN_TERMS 时**，分数才有可比性，可单独判弱；
      短查询则必须配上「零共享词」才能判弱。

    ⚠️ 第 2 条是本项目特有的坑：二元分词下「记忆系统」会切成 记忆/忆系/系统，
      于是含「系统」的无关段落也能命中。共享词检查能把这批噪声识别出来。
      实测它准确圈住了 `量子纠缠光谱仪`、`xyz 量子`、`超导磁悬浮轴承`
      （分数 5.31~6.91，看着不低，但零共享词）。
    """
    reasons = []
    hits = res.get("hits") or []
    if not hits:
        reasons.append("0 命中")
        return True, reasons

    top = hits[0].get("score", 0.0)
    # 共享「整词」检查：把查询与命中文本各自分词，看有没有交集。
    qterms = set(I.tokenize(query))
    # ⚠️ 把**子串扩展出的词**也算进查询词。反例（独立对抗复核实测）：
    #   查询 `3055` 扩展成 `202301233055` 并命中了正确文档，但共享词检查
    #   拿的是原词 `3055`，而文档里只有 `202301233055` → 判「无共享词」→
    #   把一次**成功的**召回说成「仅二元组擦边命中」并触发无谓放宽。
    #   扩展后的词才是这次检索**实际用的**查询词，比对必须用它们。
    for _k, _vs in (res.get("expansion") or {}).items():
        qterms.update(_vs)
    # 只比「有区分度」的词：单字不算（否则「系统」这类高频字几乎总能找到交集）。
    qsig = {t for t in qterms if len(t) >= 2}
    # ⚠️ 但**单字查询本身是合法的**（`机`、`钱`），此时 qsig 必然为空集
    # （分词出来的就是那个单字），旧代码于是跳过共享检查、`shared` 恒为 False，
    # 再撞上「短查询且无共享词」分支 → **把精准的单字查询判成弱**并触发无谓放宽。
    # （sol6.1 反证 agent 实测：query='机'、命中文本='机'、score=5 → 判弱。）
    # 既然查询本身就是单字，那个字就是唯一的区分度来源，直接用它比对。
    if not qsig:
        qsig = {t for t in qterms if t}
    shared = False
    if qsig:
        hit_terms = set()
        for h in hits[:3]:
            hit_terms |= set(I.tokenize(h.get("heading", "") + " " + h.get("text", "")))
        shared = bool(qsig & hit_terms)

    # 信号 ②：零共享词 —— 单独就足以判弱（分数再高也拦不住，见 docstring）
    if qsig and not shared:
        reasons.append("命中段落与查询无共享词（仅二元组擦边命中）")

    # 信号 ③：分数低 —— 仅在**长查询**（词数够多）时单独成立；
    # 短查询必须配上「零共享词」，否则会把精准的短查询误判成弱。
    if top < C.RETRIEVAL_WEAK_TOP_SCORE:
        multi_term = len(qsig) >= C.RETRIEVAL_WEAK_MIN_TERMS
        if multi_term:
            reasons.append(f"最高分 {top:.2f} < {C.RETRIEVAL_WEAK_TOP_SCORE}")
        elif not shared:
            reasons.append(f"最高分 {top:.2f} < {C.RETRIEVAL_WEAK_TOP_SCORE}（短查询且无共享词）")
        # else：短查询但命中了真词 → 只是词少，不算不清晰
    return bool(reasons), reasons


def search_ladder(
    query: str,
    top_k: int = C.RETRIEVAL_TOP_K,
    scope: str | None = None,
    max_bytes: int | None = None,
    max_rounds: int = C.RETRIEVAL_MAX_RELAX_ROUNDS,
) -> dict:
    """三级检索阶梯 —— 规则的第二、三段落地。

        ① 一次检索：纯相关度，top_k。
        ② 二次检索：命中内容太多 → 收窄 + 时间权重（在 search() 内部完成）。
        ③ 主动再检索：**结果不清晰时**逐级放宽，最多 max_rounds 次。

    阶段③的放宽手段必须**逐级**且**有记录**，否则就等于把「没查到」悄悄
    演成「查到了别的」。三级放宽：
        第 1 级：范围放宽 —— scope 限定放开（`--scope ws` 查不到时看全部）
        第 2 级：条数放宽 —— top_k 翻倍
        第 3 级：彻底放宽 —— 丢掉 max_bytes 预算，并纳入 core

    ⚠️ 每次放宽都写进 `relaxed` 列表，最终结果里带着「这是放宽后才找到的」
    这一事实。**放宽本身不是问题，放宽而不说才是问题** —— 使用者拿到的每条
    结果都应该知道它是第几级找出来的，否则会把「勉强沾边」当成「精确命中」。
    """
    def _run(k: int, sc: str | None, mb: int | None) -> dict:
        return search(query, top_k=k, scope=sc, max_bytes=mb)

    res = _run(top_k, scope, max_bytes)
    relaxed: list[str] = []
    # ⚠️ 与 `relaxed`（放宽**成功**并已采纳）分开记：这里的每一条都是
    # 「试过了、但没采纳」。两者混在一起的话，「放宽过吗」这个问题就无法回答 ——
    # 而使用者要知道的恰恰是「这次结果是不是放宽来的」。
    relax_notes: list[str] = []
    weak, reasons = _looks_weak(res, query)

    if weak and max_rounds > 0:
        # 第 1 级：放开 scope
        if scope is not None:
            r2 = _run(top_k, None, max_bytes)
            if r2["hits"]:
                relaxed.append(f"放开 scope（原 {scope}）")
                res = r2
                weak, reasons = _looks_weak(res, query)

        # 第 2 级：条数翻倍
        if weak and max_rounds > 1:
            r3 = _run(top_k * 2, None, max_bytes)
            # ⚠️ 判据是「**结果集变了**」，不是「条数变多了」。
            # sol6.1 独立复核实测：旧代码只比 len()，于是「条数相同但命中的是
            # 别的段落」会被整份丢弃 —— 检索其实找到了不一样的东西，
            # 而 `relaxed` 里一个字都没有，使用者看到的是「什么都没发生」。
            # 放宽本身不是问题，**放宽而不说**才是。
            old_ids = {h["id"] for h in res["hits"]}
            new_ids = {h["id"] for h in r3["hits"]}
            # ⚠️ 判据是「**结果集变了**」+「**没变少**」两条一起。
            # sol6.1 反证 agent 指出：只看 `new_ids != old_ids and len >=` 仍会
            # 丢弃「换了内容但条数更少」的结果 —— 而条数变少未必是预算截断，
            # 也可能是 top_k 翻倍后**候选池重排**，把原来勉强沾边的挤掉了。
            # 但那属于**放宽失败**（放宽本该带来更多/更好的候选），
            # 采纳它反而会缩小视野；所以改为：变了就采纳，**但若变少则说明
            # 放宽没带来收益**，如实记一条「放宽后结果反而减少」而不采纳。
            if new_ids != old_ids:
                if len(r3["hits"]) >= len(res["hits"]):
                    relaxed.append(f"top_k {top_k} → {top_k * 2}")
                    res = r3
                    weak, reasons = _looks_weak(res, query)
                else:
                    # 不采纳，但**也不隐瞒**：这次放宽试过了、结果反而更少。
                    # 静默跳过会让人以为「没放宽过」，而实际上系统动过手。
                    relax_notes.append(
                        f"top_k {top_k} → {top_k * 2} 试过但命中更少"
                        f"（{len(res['hits'])}→{len(r3['hits'])}），未采纳")

        # 第 3 级：丢掉预算 + 纳入 core
        if weak and max_rounds > 2:
            r4 = _run(top_k * 2, "core", None)
            if r4["hits"]:
                relaxed.append("纳入 core 且忽略预算")
                # 合并而非替换：core 的命中补在前面，保留已有结果
                seen = {h["id"] for h in res["hits"]}
                add = [h for h in r4["hits"] if h["id"] not in seen]
                if add:
                    res = dict(res)
                    res["hits"] = (add + res["hits"])[: top_k * 2]
                    res["bytes"] = sum(len(h["text"].encode("utf-8")) for h in res["hits"])
                    weak, reasons = _looks_weak(res, query)

    res["relaxed"] = relaxed
    res["relax_notes"] = relax_notes
    res["weak"] = weak
    res["weak_reasons"] = reasons
    # 把「为什么需要再检索」也讲清楚，供云端模型决定要不要自己再查一次。
    if relaxed:
        res["note"] = (res.get("note", "") + f"；**已主动再检索**（{ '、'.join(relaxed) }）").lstrip("；")
    # 试过但没采纳的放宽也要说 —— 否则「系统动过手」这件事就消失了。
    if relax_notes:
        res["note"] = (res.get("note", "") + f"；放宽未奏效（{ '、'.join(relax_notes) }）").lstrip("；")
    if weak:
        res["note"] = (res.get("note", "") + f"；仍不清晰：{ '、'.join(reasons) }").lstrip("；")
    return res


def render(res: dict, max_chars: int = 500) -> str:
    """人类可读输出。"""
    if not res["hits"]:
        extra = ""
        if res.get("relaxed"):
            extra = f"（已放宽 {'、'.join(res['relaxed'])} 仍无命中）"
        return f"No results for {res['query']!r}. {res.get('note', '')}{extra}"
    lines = [f"## 记事本检索: {res['query']!r}  ({len(res['hits'])} 命中, {res['bytes'] / 1024:.1f}KB)"]
    if res.get("note"):
        lines.append(f"_{res['note']}_")
    # ⚠️ 时间权重生效时把「依据」摆到标题下方，而不是只藏在 JSON 里：
    #   同一批结果里，`w=0.724/—` 和 `w=1.000/2026-10-06` 的含义完全不同 ——
    #   前者是「不知道日期，按中性算」，后者是「确知是最新的」。只给人看分数
    #   会让人以为排序纯按相关度，从而误判某条旧记忆「应该」排前面。
    if res.get("narrowed") and any("time_weight" in h for h in res["hits"]):
        lines.append("_排序依据 = 相关度×时间权重；`—` 表示该段无日期（取中性权重）_")
    for i, h in enumerate(res["hits"], 1):
        # 行首标签：收窄时给「相关度×权重=rank」和日期/—，未收窄时保持原样
        if "rank_score" in h:
            tag = f"[{h['score']}×{h.get('time_weight', 1.0)}={h.get('rank_score')}]"
            when = h.get("date", "—") if h.get("dated") else "—"
            tag += f" {when}"
        else:
            tag = f"[{h['score']}]"
        lines.append(f"\n{i}. {tag} {h['file']} :: {h['heading'][:60]}")
        body = h["text"] if len(h["text"]) <= max_chars else h["text"][:max_chars] + " …"
        lines.append(body)
    return "\n".join(lines)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = [a for a in sys.argv[1:]]
    if not args:
        print(__doc__)
        sys.exit(1)

    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    # ⚠️ `--ladder` 必须**一并从 args 里摘掉**，不能只判断「在不在」。
    # 实测踩到：只判断不移除，查询词变成 '量子纠缠光谱仪 --ladder'，
    # 于是 tokenize 把 "ladder" 也当成查询词喂进 BM25 —— 结果肉眼看着像对的
    # （确实返回了命中），但排序被一个凭空多出来的词影响，且搜索标题里
    # 赫然带着 `--ladder`。这类「参数混进查询文本」的错必须整类防住，
    # 所以下面的 --scope / --top-k 也都在解析后 `del args[i:i+2]`。
    use_ladder = "--ladder" in args
    args = [a for a in args if a != "--ladder"]
    scope = None
    if "--scope" in args:
        i = args.index("--scope")
        if i + 1 >= len(args):
            # 与缺陷1同族：缺值时**不能**静默按「无 scope」继续跑。
            # 旧写法 scope=None 会让 `--scope` 退化成「查全库（除 core）」，
            # 用户以为限定了范围，实际范围变了，且没有任何提示。
            print("--scope 需要一个作用域名（core / stream / ws）", file=sys.stderr)
            sys.exit(2)
        scope = args[i + 1]
        del args[i:i + 2]

    top_k = C.RETRIEVAL_TOP_K
    # ⚠️ top_k 只认显式 `--top-k N`；位置参数**一律**当查询文本。
    # 旧实现是 `if args and args[-1].isdigit(): top_k = int(args.pop())`，
    # 直接把「2023」「7897」「2026」「5」这类纯数字查询吃成空串（见文件头注释）。
    # 别为了少打几个字把它加回来：它会继续咬人，而且咬得非常安静。
    if "--top-k" in args:
        i = args.index("--top-k")
        if i + 1 >= len(args):
            print("--top-k 需要一个整数参数", file=sys.stderr)
            sys.exit(2)
        try:
            top_k = int(args[i + 1])
        except ValueError:
            print(f"--top-k 需要一个整数，收到 {args[i + 1]!r}", file=sys.stderr)
            sys.exit(2)
        if top_k < 1:
            print(f"--top-k 必须 >= 1，收到 {top_k}", file=sys.stderr)
            sys.exit(2)
        del args[i:i + 2]

    query = " ".join(args)
    # `--ladder` 走三级阶梯（主动再检索）。默认**关**：它会多跑 1~2 次检索，
    # 而且放宽后的结果与放宽前不是同一批 —— 日常用默认路径，需要「查不到就
    # 自己想办法」时才开。规格里这一步是「云端模型主动再次检索」，
    # 所以默认交给调用方决定，不替它决定。
    if use_ladder:
        res = search_ladder(query, top_k=top_k, scope=scope)
    else:
        res = search(query, top_k=top_k, scope=scope)
    if as_json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(render(res))


if __name__ == "__main__":
    main()
