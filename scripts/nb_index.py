"""记事本倒排索引 — 纯 Python，零第三方依赖。

为什么不用 SQLite FTS5 trigram（实测结论，写下来免得以后又走回头路）：
    FTS5 的 trigram 分词器对中文是 3 字符滑窗，**2 字词永远匹配不到**。
    实测：'五级' → 0 命中，'五级记' → 1 命中。中文里 2 字词占比极高，
    所以 trigram 不可用；而 jieba / sklearn 在本机未安装。
    → 自建 CJK bigram 倒排索引：2 字、3 字、长词都能命中，且零依赖。

分词规则：
    CJK 连续串 → 一元 + 二元（二元保证 2 字词可命中；一元让单字查询也能召回）
    ASCII 数字串 → 整词（长度 >= 2），按非字母数字切分（RWKV-7 → rwkv, 7）
    打分用 BM25，一元词因文档频率高而被 idf 自动降权，无需手工加权。

索引结构（JSON，可直接肉眼检查）：
    {version, docs:[{id,file,title,heading,text}], postings:{term:[doc_idx]},
     tf:{doc_idx:{term:count}}, df:{term:count}, len:{doc_idx:n}, avg_len}
"""

import datetime as _dt
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_config as C


class IndexCorrupt(Exception):
    """索引存在但不可用（JSON 损坏 / 结构缺字段 / 内容为空）。"""

TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
VERSION = 1

# ── 日期提取（时间权重的前提） ──
# ⚠️ 时间权重在 2026-10-06 之前是**没法做的**：索引 docs 里根本没有日期字段，
#   段落只有 {id,file,scope,title,heading,text}，无从判断新旧。
#   要做时间权重，第一步就是让每个段落带上日期。这里就是那一步。
DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")

# 日期只从**条目开头**取，不从正文任意位置搜。
# 理由：正文里的日期常常是「提到另一个日期」（如「2026-09-18 实测有效」写在
# 一段 10-06 的记录里），拿它当段落时间会把新旧判反。而条目开头的日期是写入时
# 打的戳（stream 的 `2026-10-06 12:02:42 | tag=x`、ws 的 `_section(f"{_stamp} | {title}")`），
# 它才是这段记忆的**诞生时间**。取不到就是取不到 —— 返回 None，
# **绝不猜、绝不填假日期**（本项目 `date_of` 曾返回 '未标日期' 当哨兵，
# 结果 96 条记忆只剩 1 条有日期，同类坑不许再犯）。
DATE_HEAD_CHARS = 40


def _valid_date(s: str) -> bool:
    """真的是一个存在的日历日？`2026-13-45` / `2026-02-31` 都**不是**。

    ⚠️ 为什么必须校验（实测抓到的崩）：正则只保证「四位-两位-两位」这个**形状**，
    月 13、日 45、日 99 都能通过。而这些字符串一旦进了 `date` 字段，
    `_days_between` 里的 `date()` 就会抛 `ValueError` —— 后果是
    **记忆里只要有一条手滑打错的日期，整个关键词索引就建不出来**
    （实测 stream.md 加一行 `2026-13-45` 即复现）。
    更糟的是检索的自愈路径是「损坏→重建」，重建同样会崩，
    于是从「静默 0 命中」直接升级成「崩」。
    校验放在**入口**，坏日期就进不了索引。
    """
    try:
        y, m, d = (int(x) for x in s.split("-"))
        _dt.date(y, m, d)
        return True
    except Exception:
        return False


def extract_date(heading: str, text: str = "") -> tuple[str | None, str | None]:
    """从条目开头取 `YYYY-MM-DD`。

    返回 `(date, rejected)`：
      · 取到合法日期 → ("2026-10-06", None)
      · 压根没有日期 → (None, None)          「不知道」
      · 有日期形状但不合法 → (None, "2026-13-45")  「原样回吐被拒的那个串」

    ⚠️ 三个通道而不是两个（本项目的老规矩：失败不能和合法取值共用通道）：
    「没写日期」和「写了但写错了」是两件不同的事 —— 前者无事可做，
    后者是**记忆数据里的错误**，应当能被数出来并报出去（见 build 的
    `corpus_dates.malformed`）。若两者都返回 None，一条手滑的日期就只是
    「安静地没有时间权重」，永远没人知道记忆里存着一个错。
    """
    for src, limit in ((heading or "", None), (text or "", DATE_HEAD_CHARS)):
        if not src:
            continue
        m = DATE_RE.search(src if limit is None else src[:limit])
        if m:
            s = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            if _valid_date(s):
                return s, None
            return None, s
    return None, None


def _days_between(a: str, b: str) -> int | None:
    """两个 `YYYY-MM-DD` 相差天数（b - a）；任一不合法返回 **None**。

    ⚠️ 旧实现直接 `date(...)` 解析，畸形日期抛 `ValueError`——把一个
    「数据里有个坏日期」的局部问题升级成「整个索引建不出来」。
    这里改为返回 None，让调用方自己决定怎么处理（当前：不施加时间权重）。
    返回 None 而不是 0：0 的意思是「同一天」，是个合法答案，不能兼职当错误码。
    """
    try:
        ya, ma, da = (int(x) for x in a.split("-"))
        yb, mb, db = (int(x) for x in b.split("-"))
        return (_dt.date(yb, mb, db) - _dt.date(ya, ma, da)).days
    except Exception:
        return None

# BM25 参数
K1 = 1.5
B = 0.75


# ── 子串扩展（数字/ASCII 片段召回）的参数 ──
# ⚠️ 常量定义在 `nb_config`（本模块**反过来 import 它**，放这里会构成循环导入）。
# 这里只是把名字接过来，保持既有引用点不变。详见 nb_config 处的说明。
MIN_SUBSTR_LEN = C.MIN_SUBSTR_LEN
MAX_SUBSTR_EXPANSION = C.MAX_SUBSTR_EXPANSION
MAX_SUBSTR_TERMS = C.MAX_SUBSTR_TERMS


def tokenize(text: str) -> list[str]:
    """CJK 一元+二元，ASCII 整词。"""
    terms: list[str] = []
    for m in TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        if tok[0].isascii():
            if len(tok) >= 2:
                terms.append(tok)
            continue
        if len(tok) == 1:
            terms.append(tok)
        else:
            terms.extend(tok[i:i + 1] for i in range(len(tok)))
            terms.extend(tok[i:i + 2] for i in range(len(tok) - 1))
    return terms


def parse_sections(text: str) -> list[tuple[str, str]]:
    """按 '## ' 切段 → [(heading, body)]。首个 ## 之前的内容归入 '(前言)'。"""
    lines = text.split("\n")
    out: list[tuple[str, list[str]]] = []
    cur_head = "(前言)"
    cur: list[str] = []
    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            if cur:
                out.append((cur_head, cur))
            cur_head = line[3:].strip()
            cur = [line]
        else:
            cur.append(line)
    if cur:
        out.append((cur_head, cur))
    return [(h, "\n".join(b).strip()) for h, b in out if "\n".join(b).strip()]


def parse_lines(text: str) -> list[tuple[str, str]]:
    """按「行」切 → [(heading, line)]，用于单行一条的记忆流。

    为什么记忆流不能用 parse_sections：stream.md 的条目是 `- 时间 | 干了什么`
    的单行 bullet，全文件没有 `## ` 标题，按段切会把**整个文件**变成一个
    巨型检索单元 —— 粒度丢失，任何查询都只能召回这一整坨。
    实测（2026-10-05 冒烟）：所有查询都命中同一个 '(前言)'，检索失效。
    → 记忆流按行切，工作区册按段切（该处正文确实是多行块）。
    """
    out: list[tuple[str, str]] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("_") or s == "---":
            continue
        # 用行首的「时间戳 | tag=x」前缀做标题，便于阅读与定位
        head = s.lstrip("-* ").strip()
        head = head[:60]
        out.append((head, s))
    return out


def collect_docs() -> list[dict]:
    """扫描三层结构，产出可索引文档单元。

    粒度：记忆流=每行一条；工作区册=每 `## ` 段；核心记忆=每段（scope=core,
    默认不参与检索，因为它是全量注入的）。
    """
    docs: list[dict] = []

    def add(path: Path, scope: str, per_line: bool) -> None:
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return
        units = parse_lines(text) if per_line else parse_sections(text)
        rel = str(path.relative_to(C.MEMORY)).replace("\\", "/")
        for i, (head, body) in enumerate(units):
            # 标题行不重复进正文，避免污染词频
            clean = "\n".join(
                ln for ln in body.split("\n") if not ln.startswith("## ")
            ).strip()
            if not clean:
                continue
            # 时间权重的前提：每段带日期。三个通道分开（见 extract_date）。
            dt, rejected = extract_date(head, clean)
            docs.append({
                "id": f"{scope}:{path.stem}:{i}",
                "file": rel,
                "scope": scope,
                "title": path.stem,
                "heading": head,
                "text": clean,
                "date": dt,
                # 日期形状对但日历上不存在 → 记下来，别让它安静地消失
                "date_rejected": rejected,
            })

    add(C.CORE, "core", False)        # 可被 --scope core 检索，默认不参与
    add(C.STREAM, "stream", True)     # 单行一条 → 按行切
    excluded = set(getattr(C, "EXCLUDED_WS_FILES", ()))
    for p in sorted(C.WS_DIR.glob("*.md")):
        # ⚠️ 排除**只**发生在这里 —— 但 `nb_write.ws()` 也要用同一个判据，
        # 否则「写入时说在某区、检索时找不到」。排除理由见 nb_config。 
        if p.name in excluded:
            continue
        add(p, "ws", False)           # 多行详情 → 按段切
    return docs


def build(verbose: bool = True) -> dict:
    C.ensure_dirs()
    docs = collect_docs()
    postings: dict[str, list[int]] = {}
    tf: dict[str, dict[str, int]] = {}
    df: dict[str, int] = {}
    length: dict[str, int] = {}

    for i, d in enumerate(docs):
        terms = tokenize(d["heading"] + "\n" + d["text"])
        counts: dict[str, int] = {}
        for t in terms:
            counts[t] = counts.get(t, 0) + 1
        tf[str(i)] = counts
        length[str(i)] = len(terms)
        for t in counts:
            postings.setdefault(t, []).append(i)
            df[t] = df.get(t, 0) + 1

    avg_len = (sum(length.values()) / len(length)) if length else 0.0

    # 语料日期范围：时间权重的参照点（见 nb_config 的说明，不用挂钟）。
    # dated / undated 分别计数，让「多少段落有时间信息」成为**可见**的数字 ——
    # 否则时间权重会在「大部分段落没日期」的情况下悄悄退化为空操作，
    # 而使用者以为它在工作。
    dated = [d["date"] for d in docs if d.get("date")]
    newest = max(dated) if dated else None
    malformed = [d for d in docs if d.get("date_rejected")]
    corpus_dates = {
        "newest": newest,
        "oldest": min(dated) if dated else None,
        "dated": len(dated),
        "undated": len(docs) - len(dated) - len(malformed),
        # ⚠️ 「无日期」与「日期写错了」分开计数，不合成一个数字：
        # 前者是信息缺失（无事可做），后者是**记忆数据里的错误**（该去改）。
        # 合成一个数，那个错就永远只是「安静地没有时间权重」。
        "malformed": len(malformed),
        "malformed_samples": [d["date_rejected"] for d in malformed[:5]],
        # 无日期段落的中性权重 = 有日期段落权重的**中位数**。
        # 在**建索引时**算好存下来，而不是检索时现算：一是省去每次检索重算，
        # 二是让这个数字变成可检查的产物 —— 它决定三分之二记忆的排序待遇，
        # 藏在检索路径里没人看得见就太危险了（`nb_doctor` 会打印它）。
        "neutral_weight": None,
    }
    if newest and dated:
        ws_ = []
        for dt in dated:
            age = _days_between(dt, newest)
            # 任一日期不合法 → 不参与中位数（_days_between 返回 None）
            if age is None:
                continue
            if age < 0:
                age = 0
            w = 1.0 if age == 0 else max(C.TIME_DECAY_MIN_WEIGHT,
                                         0.5 ** (age / C.TIME_DECAY_HALFLIFE_DAYS))
            ws_.append(w)
        # ⚠️ 全部日期都不合法时 ws_ 为空，中位数无从谈起 → 保持 None，
        # 让 time_weight 走「语料无日期」分支。绝不能拿空列表去取中位数。
        if ws_:
            ws_.sort()
            mid = len(ws_) // 2
            corpus_dates["neutral_weight"] = round(
                ws_[mid] if len(ws_) % 2 else (ws_[mid - 1] + ws_[mid]) / 2, 4)
        corpus_dates["min_weight"] = round(C.TIME_DECAY_MIN_WEIGHT, 4)
        corpus_dates["halflife_days"] = C.TIME_DECAY_HALFLIFE_DAYS

    index = {
        "version": VERSION,
        "docs": docs,
        "postings": postings,
        "tf": tf,
        "df": df,
        "len": length,
        "avg_len": avg_len,
        "corpus_dates": corpus_dates,
    }
    C.INDEX.parent.mkdir(parents=True, exist_ok=True)
    C.INDEX.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"indexed: {len(docs)} sections, {len(postings)} terms, "
              f"{C.INDEX.stat().st_size / 1024:.1f}KB -> {C.INDEX}")
        print(f"dates: {corpus_dates['dated']} 有日期 / {corpus_dates['undated']} 无日期"
              f"（{corpus_dates['oldest']} ~ {corpus_dates['newest']}）")
        # 畸形日期只有在**存在**时才打印：它是记忆数据里的错误，值得一行醒目提示；
        # 不存在的场合（正常情况）不必占一行。
        if corpus_dates.get("malformed"):
            print(f"⚠️ {corpus_dates['malformed']} 段日期不合法（日历上不存在）："
                  f"{'、'.join(str(s) for s in corpus_dates.get('malformed_samples', [])[:3])}"
                  f"　← 这些段落拿不到时间权重，请修正源文件")
        nw = corpus_dates["neutral_weight"]
        print(f"时间权重: 半衰期 {C.TIME_DECAY_HALFLIFE_DAYS:.0f} 天，"
              f"下限 {C.TIME_DECAY_MIN_WEIGHT}，"
              f"无日期段取中性权重 {nw}"
              f"　← 中性权重取自有日期段的中位数："
              f"「不知道」既不加分也不减分（不是 1.0，那会假装它最新）")
    return index


def load() -> dict | None:
    """读索引。**不存在**返回 None；**损坏**抛 IndexCorrupt。

    ⚠️ 旧实现让「文件不存在」和「JSON 损坏」都返回 None，于是调用方无法
    区分两者，一律报「索引不存在，请先运行 nb_index.py」——**文件其实存在**，
    提示是假的（2026-10-05 审查实测）。更糟的是 nb_bridge.index_stale()
    只看 exists()+mtime，一份写坏的索引会被判为「新鲜」，永不重建，
    检索永久静默返回 0 命中，与「真的没命中」不可区分。

    现在把「损坏」做成可区分的信号：调用方要么重建，要么明确报错。
    哨兵值绝不能和「无值」共用同一个返回通道 —— 这条教训在项目里已经
    犯过多次（date_of 的 '未标日期' 曾让 96 条只活下 1 条日期）。
    """
    if not C.INDEX.exists():
        return None
    text = C.INDEX.read_text(encoding="utf-8", errors="replace")
    try:
        idx = json.loads(text)
    except Exception as e:
        raise IndexCorrupt(f"索引 JSON 解析失败: {e}") from e
    if not isinstance(idx, dict):
        raise IndexCorrupt("索引根不是对象")
    for key in ("docs", "postings", "tf", "df", "len", "avg_len"):
        if key not in idx:
            raise IndexCorrupt(f"索引缺少字段: {key}")
    # ⚠️ `corpus_dates` 与 doc 里的 `date` 是时间权重的前提，**必须**列为必需字段：
    #   旧索引（2026-10-06 之前建的）没有它们。若按「可选」处理，时间权重就会
    #   静默退化成空操作 —— 检索照常返回结果，只是排序里从来没有时间因素，
    #   而没有任何迹象表明这件事。列为必需 → 旧索引直接报
    #   IndexCorrupt 并触发重建（nb_search 已有「损坏→重建」路径），
    #   失败是**可见**的，代价只是重建一次。
    if "corpus_dates" not in idx:
        raise IndexCorrupt("索引缺少字段: corpus_dates（旧版索引，重建即可获得时间权重）")
    # ⚠️ 光「字段存在」不够 —— sol6.1 独立复核实测：手改过的索引把 corpus_dates
    # 写成 `[]` 或 `{'newest': None}` 也能通过旧检查，于是 time_weight() 一律返回
    # `(1.0, False)`：**排序里从来没有时间因素，而检索照常返回结果**。
    # 这与「索引缺字段」是同一个故障（时间权重静默失效），只是更隐蔽 ——
    # 字段在、值是垃圾。既然承认它是必需字段，就得连**形状**一起验。
    if not isinstance(idx["corpus_dates"], dict):
        raise IndexCorrupt("索引 corpus_dates 不是对象（形状损坏，重建即可）")
    # ⚠️ 「有 docs 却全无日期」是合法的（新建库还没记日期）；但「有日期段却没算出
    # newest/neutral_weight」自相矛盾 —— 那必然是被改坏了，因为 build() 是一起写的。
    cd = idx["corpus_dates"]
    # ⚠️ **先把 docs 验干净，再拿它做一致性判断** —— 顺序反了的话，
    # `sum(... d.get("date") ...)` 会先撞上非 dict 的 doc 抛 AttributeError
    # （旧代码用 `"date" in doc` 子串测试时，`docs=['date']` 这种字符串也能过）。
    if not isinstance(idx["docs"], list) or not idx["docs"]:
        raise IndexCorrupt("索引 docs 不是非空列表")
    bad_doc = next((i for i, d in enumerate(idx["docs"])
                    if not isinstance(d, dict) or "date" not in d), None)
    if bad_doc is not None:
        raise IndexCorrupt(
            f"索引第 {bad_doc} 条 doc 不是对象或缺 date 字段"
            "：旧版或损坏的索引，重建即可获得时间权重")
    # ⚠️ 判据不能依赖 `dated` 字段本身存在 —— sol6.1 复验实测：一份
    # `{"newest": None, "neutral_weight": None}`（**根本没有 dated 键**）的
    # corpus_dates 照样通过，因为旧检查写的是 `if cd.get("dated") and ...`，
    # dated 缺失时 `None` 为假、整个检查被跳过，时间权重静默失效。
    # 正确判据是「**docs 里到底有没有日期**」，直接数，不信任元数据自述：
    n_dated = sum(1 for d in idx["docs"] if d.get("date"))
    # 有日期段 → 必须算得出参照点与中性权重；无日期段 → newest/neutral 都该是 None。
    if n_dated:
        if not cd.get("newest") or cd.get("neutral_weight") is None:
            raise IndexCorrupt(
                f"索引自相矛盾：docs 里有 {n_dated} 条带日期，但 "
                f"newest={cd.get('newest')!r} / neutral_weight={cd.get('neutral_weight')!r}"
                "（时间权重会静默失效，重建即可）")
    elif cd.get("newest") or cd.get("neutral_weight") is not None:
        raise IndexCorrupt(
            f"索引自相矛盾：docs 里没有一条带日期，但 "
            f"newest={cd.get('newest')!r} / neutral_weight={cd.get('neutral_weight')!r}")
    # ⚠️ **newest 自己也必须是合法日期**（flash 复测找到，我复验确认）：
    # 只验「字段非空」不够 —— 手改成 `2026-13-45` 也非空，于是 time_weight 里
    # `_valid_date(newest)` 为假 → **每一个有日期的段落都被当成「没日期」**，
    # 全体拿到中性权重 0.74。排序完全一样了（时间维度整体失效），
    # 而 corpus_dates 里 dated=205 看着还挺健康。这是最隐蔽的一类失效：
    # 数据在、字段在、数值都在，只是**参照点坏了**，于是一切照旧却全无时间性。
    if cd.get("newest") and not _valid_date(cd["newest"]):
        raise IndexCorrupt(f"索引 corpus_dates.newest 不是合法日期: {cd['newest']!r}"
                           "（时间权重的参照点坏了会静默全失效，重建即可）")
    if cd.get("oldest") and not _valid_date(cd["oldest"]):
        raise IndexCorrupt(f"索引 corpus_dates.oldest 不是合法日期: {cd['oldest']!r}")
    # ⚠️ neutral_weight 必须能被消费：它决定 63% 无日期记忆的排序待遇。
    # 坏值会让 time_weight 回退到 1.0，**把无日期段落集体提到「最新」**，
    # 反过来挤掉有日期的真记忆。（实测 neutral=0 或 nan 时确实回退成 1.0。）
    # 用与消费端**同一个判据** `_safe_neutral` 验，避免两侧标准不一致。
    if cd.get("neutral_weight") is not None and _safe_neutral(cd) is None:
        raise IndexCorrupt(
            f"索引 neutral_weight 不可用: {cd['neutral_weight']!r}"
            f"（合法域 [{C.TIME_DECAY_MIN_WEIGHT}, 1.0]；坏值会把无日期段落提到最新，重建即可）")
    if not isinstance(idx["docs"], list) or not idx["docs"]:
        raise IndexCorrupt("索引 docs 不是非空列表")
    bad_doc = next((i for i, d in enumerate(idx["docs"])
                    if not isinstance(d, dict) or "date" not in d), None)
    if bad_doc is not None:
        raise IndexCorrupt(
            f"索引第 {bad_doc} 条 doc 不是对象或缺 date 字段"
            "：旧版或损坏的索引，重建即可获得时间权重")
    # ⚠️ docs 非空但 postings 为空 = 检索永久 0 命中（合法而空的索引是坏索引）。
    # 旧检查只看 docs，于是这种索引被放过，检索静默返回「没有相关记忆」。
    if not idx.get("postings"):
        raise IndexCorrupt("索引 postings 为空（检索会永久 0 命中，重建即可）")
    return idx

def load_or_none() -> dict | None:
    """只用于「拿不到就当没有」的场景；损坏同样返回 None（调用方需自行处理降级）。"""
    try:
        return load()
    except IndexCorrupt:
        return None


def idf(df: int, n: int) -> float:
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def _safe_neutral(corpus) -> float | None:
    """取中性权重，**校验类型与范围**；不可信就返回 None（= 走「未生效」分支）。

    ⚠️ sol6.1 独立复核实测：旧代码直接 `float(neutral)` 回吐，于是被改坏的索引
    能让无日期段落拿到 `2.0`（比最新还新）、`nan`、甚至 `-1.0`。
    这三种都会**污染重排**而不报错 —— 一个 NaN 混进排序键，`sorted` 的结果
    就再无意义，而检索看起来完全正常。权重是乘性因子，合法域就是
    [min_weight, 1.0]，越界即数据损坏，不能当合法值用。
    """
    if not isinstance(corpus, dict):
        return None
    nw = corpus.get("neutral_weight")
    if nw is None or isinstance(nw, bool):
        return None
    try:
        w = float(nw)
    except (TypeError, ValueError):
        return None
    # NaN 与 ±inf 都不满足 `lo <= w <= 1.0`（NaN 的比较恒为 False），一并挡掉。
    if not (C.TIME_DECAY_MIN_WEIGHT <= w <= 1.0):
        return None
    return w


def time_weight(doc: dict, corpus) -> tuple[float, bool]:
    """段落的时间权重 ∈ [min_weight, 1.0]。越新越接近 1.0。

    返回 `(weight, applied)` —— **第二个返回值是必须的**：
    `applied=False` 表示「这段没有日期，用的是中性权重」，而不是「它是最新的」。
    若只返回一个数，调用方/人就分不清「它最新」和「它没有日期」——
    前者是事实，后者是信息缺失，两者在排序里可能同分但含义完全相反。
    这正是本项目反复栽的「哨兵值与真实取值共用通道」的同一个坑。

    ⚠️ 两处刻意的设计（都由实测标定，别轻易改）：

    **1. 参照点是语料内最新日期，不是挂钟。** 用挂钟意味着系统时间一错、
    或记忆停更一段时间，所有权重一起塌掉，而此时权重看起来仍在正常工作
    （都能算出数），属于静默失效。用语料自身最新日期可自我校准。

    **2. 无日期段用中性权重（中位数），不是 1.0，也不是最小值。**
    1.0 = 假装它最新（当前 331/523 段无日期，等于让三分之二记忆凭空
    获得最新优先权，把有日期的真记忆挤掉）；最小值 = 假装它最旧
    （把整个 ws 册压到不可见）。中位数是唯一诚实的取值。

    **3. 权重有下限** `TIME_DECAY_MIN_WEIGHT`：时间只许倾斜排序，不许压倒
    相关度。实测跨度 101 天时最老段落权重 0.0969 → 需 10.3 倍 BM25 才能
    排前面，那等于「新」压过「准」，与「不丢旧事实」的目标相反。
    """
    if not C.TIME_DECAY_ENABLED:
        return 1.0, False
    # ⚠️ corpus 可能是任何东西（索引是磁盘 JSON，可被手改）→ 不假设它的类型。
    if not isinstance(corpus, dict):
        corpus = {}
    d = doc.get("date") if isinstance(doc, dict) else None
    newest = corpus.get("newest")
    lo = C.TIME_DECAY_MIN_WEIGHT
    neutral = _safe_neutral(corpus)
    # ⚠️ `_valid_date` 再挡一道：date 字段理论上已经校验过（extract_date），
    # 但索引是磁盘上的 JSON，可能被手改、或由旧版本写出。
    newest_ok = bool(newest) and _valid_date(newest)

    # ⚠️ **语料级健康检查必须整体判断，不能逐段各判各的**（这一条是修 bug 的核心）。
    # 若参照点或中性权重任一不可用 → 时间权重**整体不生效**，全员 1.0。
    # 反面案例（我复验过的实际行为）：newest 合法但 neutral_weight 被改坏时，
    # 旧代码让**有日期**的段落照常算出真权重、而**无日期**的段落退回 1.0 ——
    # 于是 63% 的无日期记忆被集体抬到「最新」档，反过来把有日期的真记忆挤下去。
    # 那正是整套设计要避免的失效模式（「个别段坏掉」变成了「静默改变全体排序」）。
    # 宁可不生效并如实上报（applied=False），也不做半个生效的时间权重。
    if not newest_ok or neutral is None:
        return 1.0, False

    # 到这里语料元数据是健康的：参照点合法、中性权重合法。
    # 无日期（或日期不合法）的段落 → 中性权重。**不是 1.0**（那会说成最新）。
    if not d or not _valid_date(d):
        return neutral, False
    age = _days_between(d, newest)
    if age is None:
        # 理论上不可达（上面刚校验过），但真发生了也必须是「没日期」而不是崩。
        return neutral, False
    if age < 0:
        age = 0          # 段落日期比语料最新还新：按最新处理，不放大
    hl = C.TIME_DECAY_HALFLIFE_DAYS
    if hl <= 0 or age == 0:
        return 1.0, True
    return max(lo, 0.5 ** (age / hl)), True


def missing_terms(query_terms: list[str], index: dict) -> list[str]:
    """查询里**精确查不到**、且长度够扩展的词（已去重，顺序稳定）。"""
    postings = index.get("postings") or {}
    seen: dict[str, None] = {}
    for t in query_terms:
        if len(t) < MIN_SUBSTR_LEN or t in postings:
            continue
        seen.setdefault(t, None)
    return list(seen)


def exact_candidates(query_terms: list[str], index: dict) -> set[int]:
    """只用**精确命中的词**取候选文档。用于判断「还需要不扩展」。

    与 `score()` 的候选集算法一致（同一份 postings），所以两者不会互相矛盾。
    """
    postings = index.get("postings") or {}
    cands: set[int] = set()
    for t in set(query_terms):
        if t in postings:
            cands.update(postings[t])
    return cands


def expand_terms(query_terms: list[str], index: dict) -> tuple[list[str], dict[str, list[str]]]:
    """把「精确查不到」的词按**子串**扩展成索引里真实存在的词。

    ⚠️ 修的是一处真实且致命的召回缺口：ASCII/数字词是**整词**入索引的
    （`tokenize` 里 ASCII 段整体保留，CJK 走一元+二元），所以中文能靠二元组
    部分命中，**数字不能**。实测（修复前）：
        `202301233055` → 3 条命中（整词恰好一致）
        `2023`         → **0 条**   ← 学号前四位，人最可能这么查
        `3055`         → **0 条**   ← 后四位，同样是最自然的查法
    一个只存在于 `202301233055` 里的学号，用它的任何片段都查不到 ——
    而「用片段去查」正是人的默认行为。这不是「排序不好」，是**查不到**。

    ⚠️⚠️ **但第一版实现有严重副作用，已被独立对抗复核否掉并重写**。第一版是
    无条件给每个查不到的词扩展，于是「一个词精确命中、另一个词是垃圾」时，
    扩展出的词**累加 BM25** 压过精确匹配。实测：
        `bom ing`  →  `bom` 精确命中文档被挤到第 7 名，
        `ing` 扩展出的 being/naming/owning/working/routing 因累加项多而夺魁
        （赢家文档里 `bom` 的词频是 0）
    这直接违背了「精确匹配必须胜过片段擦边」—— 而检索的全部意义就在于此。
    候选数落也有同类问题：按长度截断前 8 个时，`20260` 真正相关的那条文档
    靠**较长的**文件名 token 命中，被截掉了（未截断时它排第一）。

    因此现在的规则是**保守 + 有闸门**的：
      · **闸门：只有当查询的精确词一个候选都取不到时才扩展**（由调用方
        `search()` 用 `exact_candidates` 判断）。只要精确路径有任何命中，
        就走纯粹的精确路径，扩展完全不参与 —— 精确结果不可能被污染。
      · 只展开「精确查不到」的词，长度 ≥ `MIN_SUBSTR_LEN`。
      · **去重**（重复词不再重复扫描，也不重复累加）。
      · 候选按 `(长度, 词)` 排序 —— 等长时按词**字典序**，保证
        **确定性**：同一查询任何时候都给出同一候选集。
        （旧版只按长度排，等长项的顺序取决于 postings 的插入顺序，
        实测同一组键在插入顺序变化后给出不同的前 8 项。）
      · 上限 `MAX_SUBSTR_EXPANSION` 放宽到 64：要能装下 `2023` 这类
        常见四位数字的全部匹配键（实测 16 个）。截成 8 个会静默丢掉相关项。
      · 返回扩展映射，供上层**如实报告**「这是靠子串扩展才找到的」。

    返回 (terms, {原词: [扩展出的词...]})；无需扩展时映射为空 dict。
    """
    postings = index.get("postings") or {}
    terms: list[str] = []
    expanded: dict[str, list[str]] = {}
    n_expanded = 0
    for t in query_terms:
        if t in postings or len(t) < MIN_SUBSTR_LEN:
            terms.append(t)
            continue
        if t in expanded:
            # 已扩展过的重复词直接复用结果，不重复扫描（实测 1000 个重复词
            # 原本会扫 1000 遍全表 → 300+ms；去重后 1ms）
            terms.extend(expanded[t])
            continue
        if n_expanded >= MAX_SUBSTR_TERMS:
            # 超出上限 → 原样保留（走「查不到」路径），不再扫表。
            # 正常查询远达不到这个数，这里只为挡住畸形长输入。
            terms.append(t)
            continue
        n_expanded += 1
        hits = sorted((k for k in postings if t in k and k != t), key=lambda k: (len(k), k))
        if hits:
            hits = hits[:MAX_SUBSTR_EXPANSION]
            expanded[t] = hits
            terms.extend(hits)
        else:
            terms.append(t)
    return terms, expanded


def score(query_terms: list[str], index: dict) -> list[tuple[int, float]]:
    """BM25 打分 → [(doc_idx, score)]，降序。"""
    n = len(index["docs"])
    if n == 0:
        return []
    postings, tf, dfmap, lens = index["postings"], index["tf"], index["df"], index["len"]
    avg_len = index["avg_len"] or 1.0

    # 候选文档 = 至少命中一个查询词的文档（避免全库扫描）
    candidates: set[int] = set()
    for t in set(query_terms):
        candidates.update(postings.get(t, []))
    if not candidates:
        return []

    results: list[tuple[int, float]] = []
    qset = set(query_terms)
    for idx in candidates:
        dl = lens.get(str(idx), 0)
        s = 0.0
        doc_tf = tf.get(str(idx), {})
        for t in qset:
            f = doc_tf.get(t, 0)
            if not f:
                continue
            df = dfmap.get(t, 0)
            s += idf(df, n) * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / avg_len))
        if s > 0:
            results.append((idx, s))
    results.sort(key=lambda x: -x[1])
    return results


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        idx = load()
        if not idx:
            print("no index; run nb_index.py first")
            return
        print(f"docs={len(idx['docs'])} terms={len(idx['postings'])} avg_len={idx['avg_len']:.0f}")
        for d in idx["docs"][:5]:
            print(f"  {d['id']}  {d['heading'][:40]}")
        return
    build()


if __name__ == "__main__":
    main()
