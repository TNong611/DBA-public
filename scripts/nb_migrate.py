"""旧五级记忆 → 记事本三层迁移。

分配规则（用户定的写入者分工，迁移时按内容归属反向落位）：
    fixed.md + USER.md        → core.md        （核心记忆，≤15KB，全量注入）
    recent.md + summary.md    → stream.md      （记忆流，日级「何时干过什么」，不详细）
    summary/long_term/archive → ws/<工作区>.md （工作区册，详载细节）
    全部原文另存 *.legacy.md，不删除

为什么记忆流要走「日级聚类」而不是照搬 recent.md：
    recent.md 是 43.7KB 的逐轮纪要（含 User/Assistant 全文截断），
    直接搬过去等于把明细塞进记忆流 —— 违背「记忆流不详细」的定义。
    记忆流只保留「某天干过什么」一行一条，明细留在工作区册。

分类器：关键词打分（不引入模型依赖）。命中数最高者胜，全 0 落 misc。
运行：py nb_migrate.py            # 预演，只打报告
      py nb_migrate.py --apply    # 实际写入
"""

import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_config as C

M = C.MEMORY

# 工作区关键词表：命中数最高者胜
WORKSPACES = {
    "双脑Agent": ["双脑", "CLTRN", "RWKV", "槽位", "状态卡", "神经记忆",
                 "架构设计_v1", "RNN", "递归主干", "readout", "KV Adapter"],
    "记忆系统": ["snowfox", "SnowFox", "DSH", "deepseek-harness", "F0", "组装",
                "五级记忆", "压缩", "注入", "插件", "技能库", "Hermes",
                "assembly", "L1", "L2", "L3", "L4", "最近对话", "长期记忆",
                "计划任务", "memory-continuity"],
    "模型接入": ["模型", "API", "api_key", "key", "Claude", "中转", "Ollama",
                "Codex", "provider", "gpt", "deepseek", "指纹", "路由",
                "cc-switch", "ccswitch", "opus", "sonnet", "代理", "base_url",
                "DeepSeek", "思考链", "token"],
    "Windows维护": ["Windows", "自动更新", "注册表", "wuauserv", "系统还原",
                   "SolidWorks", "DISM", "弹窗", "服务", "gpedit", "更新任务",
                   "守护", "PauseUpdates"],
    "求职": ["求职", "简历", "宣讲会", "汇川", "雷赛", "宇树", "offer", "面试",
            "招聘", "实习", "精慧智", "一跃", "校招"],
    "角色卡": ["游街犬", "人犬", "角色卡", "chara_card", "{{char}}", "{{user}}",
              "世界架构", "叙事身份", "公共性", "牵犬人"],
    "嵌入式": ["CanMV", "sdcard", "libs", "电机", "运动控制", "嵌入式",
              "单片机", "STM32", "K230", "测试代码", "test_classify"],
}
DEFAULT_WS = "misc"

# 记忆流的时间线来源（按沉淀顺序）
STREAM_SOURCES = ["recent.md", "summary.md", "long_term.md", "archive.md"]
# 工作区册的详情来源
WS_SOURCES = ["summary.md", "long_term.md", "archive.md", "recent.md"]

# 记忆流单行长度：用户定义「记忆流不详细」，所以取短摘要。
# 曾用 160 → 记忆流行与工作区册首句几乎相同，检索同时命中两处、一半结果是重复
# （2026-10-05 实测）。压到 70 字后记忆流只承担「何时干过什么」的时间线索引职责。
STREAM_DIGEST_LIMIT = 70
# 工作区册单条上限：180KB 全量入库会让每次检索都冲破 6KB 预算、次次走二次检索，
# 1400 字约等于一个完整技术结论，再长的原文由 *.legacy 兜底。
WS_DETAIL_LIMIT = 1400


# ── 文本工具 ──
def classify(head: str, body: str = "") -> str:
    """按关键词打分分类。

    ⚠️ 标题权重 ×3：标题是内容的浓缩，正文则常夹带无关词。
    实测反例：标题「投资偏好：不限济南，宇树、大疆、众擎等机器人公司均可投」
    正文含「Clash代理致502」→ 单纯按全文计数会被判成「模型接入」，
    而它显然是「求职」。标题加权后归位。
    """
    h, b = head or "", body or ""
    scores = {}
    for ws, kws in WORKSPACES.items():
        scores[ws] = 3 * sum(h.count(k) for k in kws) + sum(b.count(k) for k in kws)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else DEFAULT_WS


def is_meta(s: str) -> bool:
    """判断是否为容器标题（时间戳/序号/重复标记），非真内容。"""
    t = s.strip()
    if not t:
        return True
    if re.fullmatch(r"[\d\-: ]+(\|.*)?", t):
        return True
    if re.fullmatch(r"#+\s*", t):
        return True
    return False


# 旧纪要里区分说话人的标记。用私用区字符而非 '→'：
# ⚠️ 中文技术文本本身就大量使用 '→' 表示数据流（如 「state_t → state_{t+1}」），
# 拿 '→' 当分隔符会把正常内容从箭头处腰斩（2026-10-05 实测）。
# 私用区字符 U+E000/U+E001 在真实语料里不可能出现，作标记安全。
U_MARK = "\ue000"
A_MARK = "\ue001"


def clean(t: str) -> str:
    t = re.sub(r"【重复×\d+】", "", t)
    t = t.replace("**User**:", U_MARK).replace("**Assistant**:", A_MARK)
    t = re.sub(r"^\s*[-*]\s*", "", t, flags=re.M)
    return " ".join(t.split())


# 记忆流是「干过什么」的记录，不是对话摘抄。以下形态不产生有效事件行。
_JUNK_PATTERNS = (
    re.compile(r"^[·→\s]*(OK|好|可以|对|是|嗯|收到|明白)[。！？…\s]*$", re.I),
    re.compile(r"^[·→]?\s*$"),
    re.compile(r"^sk-[A-Za-z0-9]{16,}"),          # 裸 API key，不进时间线
    re.compile(r"^[·→\s]*[（(]?(空|无|略)[)）]?[。\s]*$"),
)


def is_junk(dg: str) -> bool:
    """判断摘要是否为无信息量的碎片。"""
    s = dg.strip()
    if len(s) < 12:
        return True
    return any(p.match(s) for p in _JUNK_PATTERNS)


def answer_part(body: str) -> str:
    """从 '用户问 / 助手答' 形态中取「助手答」部分。

    ⚠️ 记忆流的定义是「何时干过什么」，即**做过的事**，不是用户问了什么。
    实测照搬原文会产生 `你怎么把你自己改挂了？` 这种纯提问行 ——
    既不是动作记录，检索出来也无信息量（2026-10-05）。
    旧纪要格式为 `**User**: X` / `**Assistant**: Y`，clean() 后成为
    `U_MARK X A_MARK Y`，所以取 A_MARK 之后的部分才是「做了什么」。
    只有存在 A_MARK 时才切；否则原样返回，避免误伤普通文本。
    """
    if A_MARK in body:
        tail = body.split(A_MARK)[-1].strip()
        if tail:
            return tail
    return body


def digest(text: str, limit: int = 160, head: str = "") -> str:
    """取一句可读摘要：优先第一句，否则截断。

    ⚠️ 只按中文句读 + '。！？；' 与 'ASCII 句点+空格' 断句。
    曾用 `(?<=[.!?])` 无条件断句 → 'SnowFox五级记忆v4.8.0' 被切成 'v4.'，
    版本号/文件名/域名全被截断（2026-10-05 预演实测）。

    ⚠️ head 用于补充信息，但若正文已以 head 开头（旧标题=首句的常见形态），
    不重复拼接 —— 否则输出 'X。X。' 自我重复（实测出现过）。
    """
    body = clean(text)
    body = re.sub(r"^#+\s*.{0,40}?\s*", "", body)      # 去标题残留
    body = re.sub(r"(session=[\w\-]+|\| 纪要|compressed-from-L1|retired-from-L3)", "", body)
    body = " ".join(body.split())

    h = clean(head)
    if h and body and not body.startswith(h[:20]):
        body = f"{h}。{body}"
    elif not body:
        body = h
    if not body:
        return ""
    parts = re.split(r"(?<=[。！？；])|(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff])", body)
    parts = [p.strip() for p in parts if p and p.strip()]
    first = parts[0] if parts else body
    if 8 <= len(first) <= limit:
        return first
    return (first if len(first) > limit else body)[:limit].rstrip(" ，,、") + ("…" if len(body) > limit else "")


def strip_marks(t: str) -> str:
    """去掉内部标记符，输出前必做。"""
    return t.replace(U_MARK, "").replace(A_MARK, " ").strip()


# ── 密钥/卡号脱敏 ──
# 记忆要被检索并且**每轮注入**，明文密钥/卡号进去等于把它们复制到每一次
# 模型请求里。此处只做模式脱敏，保留可辨识前缀便于回查原文件
# （原文仍在 *.legacy，本地留存，不进检索库）。
#
# ⚠️ 不能用 `\b` 作边界：Python 的 `\w` 含 CJK，中文与数字相邻处**没有**
# 单词边界，所以 `卡号4549 2418 6452 9490` 里的 `\b\d{4}` 匹配不上
# —— 实测漏过整张卡号（2026-10-05）。一律改用 `(?<!\d)` / `(?!\d)`。
_SECRET_PATTERNS = (
    # sk- 开头的各类 key（DeepSeek/OpenAI/中转站），保留前 8 位便于辨认
    (re.compile(r"(?<![A-Za-z0-9])(sk-(?:ant-)?[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]{10,}"),
     r"\1…«已脱敏»"),
    # 长 hex 串（32+），多为 key/token
    (re.compile(r"(?<![0-9a-fA-F])([0-9a-f]{8})[0-9a-f]{24,}(?![0-9a-f])"),
     r"\1…«已脱敏»"),
    # 银行卡号：4 组 4 位（空格/连字符分隔）
    (re.compile(r"(?<!\d)\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{4}(?!\d)"), "«卡号已脱敏»"),
    # 银行卡号：连续 16~19 位
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "«卡号已脱敏»"),
    # 卡片的 BIN / CVV / 有效期：贴着标签的数字才算，避免误伤普通数字
    (re.compile(r"(?i)(BIN\s*[:：]?\s*)\d{6,8}"), r"\1«已脱敏»"),
    (re.compile(r"(?i)(CVV\s*[:：]?\s*)\d{3,4}"), r"\1«已脱敏»"),
    (re.compile(r"(有效期\s*[:：]?\s*)\d{2}\s*/\s*\d{2,4}"), r"\1«已脱敏»"),
    # 常见 bearer / token= 赋值
    (re.compile(r"(?i)\b(token|apikey|api_key|password|passwd|secret)\s*[=:]\s*\S{8,}"),
     r"\1=«已脱敏»"),
)


def scrub_secrets(t: str) -> str:
    """把密钥/卡号从记忆正文里抹掉。"""
    for pat, rep in _SECRET_PATTERNS:
        t = pat.sub(rep, t)
    return t


def sanitize(t: str) -> str:
    """记忆正文出库前的统一清理：标记符 + 密钥。"""
    return scrub_secrets(strip_marks(t))


def heading_of(head: str, body: str) -> str:
    """给一个块定标题。

    ⚠️ 标题是**被索引的关键词来源**，也是检索结果里唯一能说明「这段讲什么」
    的东西。曾对时间戳式标题（recent/summary.md 大量如此）直接填空
    `'条目'`，导致检索结果里满屏 `### [ws/模型接入.md] 条目` ——
    既看不出内容，也让标题的关键词贡献归零（2026-10-05 实测）。
    正确做法：标题缺失/是元信息时，**取正文首句**当标题。
    """
    h = sanitize(clean(head)).strip("：: -|")
    if h and not is_meta(head) and len(h) > 3:
        return h[:100]
    first = sanitize(clean(body)).strip()
    for sep in ("。", "！", "？", "；", "\n", ". ", "; "):
        i = first.find(sep)
        if 0 < i < 90:
            first = first[:i]
            break
    first = first[:80].strip("：: -|")
    return first or (h or "条目")


def parse_blocks(text: str) -> list[tuple[str, str]]:
    """按 '## ' 切块 → [(heading, body)]。

    ⚠️ 不过滤「空正文的标题块」。旧记忆里大量这种结构：
        `## 2026-09-20 16:57:11 | retired-from-L3`   ← 容器标题（无正文）
        `## 真正的内容标题`                            ← 内容
    若在此处按 `if b` 过滤掉空块，日期容器会消失，调用方再也拿不到日期，
    96 条记忆全部记为「未标日期」（2026-10-05 实测：有日期的只剩 1 条）。
    → 这里只做切分，是否跳过由调用方用 is_meta 决定，日期由调用方继承。
    """
    out, cur_h, cur = [], "(前言)", []
    for line in text.split("\n"):
        if line.startswith("## ") and not line.startswith("### "):
            if cur or cur_h != "(前言)":
                out.append((cur_h, "\n".join(cur).strip()))
            cur_h, cur = line[3:].strip(), []
        else:
            cur.append(line)
    if cur or cur_h != "(前言)":
        out.append((cur_h, "\n".join(cur).strip()))
    return out


def date_of(text: str) -> str:
    """提取日期；**无匹配返回空串**。

    ⚠️ 曾返回 '未标日期' 而非 ''，导致调用方 `if h_date:` 恒为真：
    真日期容器读到的 '未标日期' 被当成有效日期写进 last_date，
    再把后续所有块的日期覆盖掉 —— 实测 96 条记忆只剩 1 条有日期。
    哨兵值绝不能和「无值」共用同一个返回通道。
    """
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else ""


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    apply = "--apply" in sys.argv

    backups = sorted((Path(r"D:\AI\双脑Agent\backups")).glob("memories-*"))
    print(f"备份存在: {bool(backups)}  ({backups[-1].name if backups else '无'})")
    if not backups:
        raise SystemExit("先备份再迁移")

    # ── 1. 核心记忆：fixed.md + USER.md ──
    fixed = (M / "fixed.md").read_text(encoding="utf-8", errors="replace")
    user = (M / "USER.md").read_text(encoding="utf-8", errors="replace")

    core_parts = ["# 核心记忆\n_全量注入 | 上限 15KB | 云端模型维护_"]
    # 用户画像（USER.md 以 § 分隔）
    core_parts.append("\n## 用户")
    for seg in [s.strip() for s in user.split("§") if s.strip()]:
        for line in seg.split("\n"):
            line = clean(line)
            if line:
                core_parts.append(f"- {line}")

    # fixed.md 的 ## 段整体保留（路径/配置/踩坑 = 硬事实，正属核心记忆）
    in_comment = False
    for head, body in parse_blocks(fixed):
        if is_meta(head) or head == "(前言)":
            continue
        body_c = body.replace("<!--", "").replace("-->", "").strip()
        core_parts.append(f"\n## {clean(head)}")
        for line in body_c.split("\n"):
            s = line.strip()
            if not s or s.startswith("<!--") or s.endswith("-->"):
                continue
            core_parts.append(s if s.startswith(("-", "*", "|")) else f"- {s}")

    existing = {clean(re.sub(r"\s*\|.*$", "", h)) for h, _ in parse_blocks("\n".join(core_parts))}
    core_text = "\n".join(core_parts).rstrip() + "\n"
    # ⚠️ 按**字节**判超限，不用 KB 浮点近似：15361B → 15.0KB → `15.0 > 15` 为
    # False，刚超限的一刻会静默通过。用 C.over_core_limit 与全项目同一判据。
    core_b = len(core_text.encode("utf-8"))
    over_b = C.over_core_limit(core_b)
    print(f"\n=== core.md  {core_b / 1024:.2f}KB / {C.CORE_MAX_KB}KB  "
          f"（{core_b}B/{C.core_limit_bytes()}B）  "
          f"{f'⚠️ 超限 {over_b} 字节' if over_b else 'OK'} ===")
    for h in [h for h, _ in parse_blocks(core_text)]:
        if h != "(前言)":
            print(f"   ## {h[:70]}")

    # ── 2. 记忆流：时间线，一块一条 ──
    # ⚠️ 不能用「(日期,主题) 聚合后取首条」：实测那样 240KB 只活下来 7 行，
    #    等于把时间线整条丢掉，只剩 7 个孤点。记忆流的定义是「何时干过什么」，
    #    所以每个来源块产出一行，保留完整事件序列，只把单行压短。
    #
    # ⚠️ 日期继承规则（踩过的坑，别改回去）：
    #    只有「**空正文**的纯日期标题」才作为容器供后续块继承。理由是两种文件结构不同：
    #      archive.md  →  `## 2026-09-20 16:57:11 | retired-from-L3`（空正文，纯容器）
    #                     `## 真正的内容标题`                        ← 应继承 09-20 ✓
    #      long_term.md→  `## FANUC 组件重注册（…2026-09-18 实测有效）`（**有正文**）
    #                     它是内容块本身，不是容器 → 不得更新 last_date
    #    曾写成「任何带日期的标题都更新 last_date」→ long_term 里一个 09-18 块
    #    把其后 78 个块全部染成 2026-09-18，整条时间线被压成单日（实测发现）。
    #    另外 date_of 无匹配必须返回 ''：曾返回哨兵 '未标日期'，`if h_date:` 恒真，
    #    真日期被假值覆盖，96 条只剩 1 条有日期。
    timeline: list[tuple[str, str, str, str]] = []
    for name in STREAM_SOURCES:
        p = M / name
        if not p.exists():
            continue
        last_date = ""
        for head, body in parse_blocks(p.read_text(encoding="utf-8", errors="replace")):
            h_date = date_of(head)
            # 容器判定只看「有没有正文」，不看标题长相。
            # ⚠️ 曾用 is_meta(head) 决定跳过，导致 recent.md / summary.md
            # **整份被丢**：它们的标题形如
            #   `## 2026-10-04 15:53:01 | session=… | 纪要`
            #   `## 2026-08-12 09:48:25 | compressed-from-L1`
            # 前导部分是纯时间戳 → is_meta 判为 true → 连同正文一起 continue。
            # 实测后果：43.7KB recent + 96KB summary 全部消失，时间线只剩
            # long_term/archive 的 96 行、且日期几乎全丢（2026-10-05）。
            # 正确规则：**只有空正文的纯日期标题才是容器**；带正文的一律是内容。
            if not body.strip():
                if h_date:
                    last_date = h_date     # 纯日期容器 → 供后续块继承
                continue
            if is_meta(head):
                head = ""                  # 标题只是时间戳/序号 → 不进摘要文本
            d = h_date or date_of(body) or last_date or "未标日期"
            ws = classify(head, body)
            # 取「做了什么」而非「问了什么」，再取短摘要
            act = sanitize(answer_part(body))
            dg = sanitize(digest(act, limit=STREAM_DIGEST_LIMIT, head=head))
            if is_junk(dg):
                continue
            timeline.append((d, ws, dg, name))

    # 去重：同一事件在不同层反复沉淀过（long_term 里大量 【重复×N】）
    seen: set[str] = set()
    uniq: list[tuple[str, str, str, str]] = []
    for d, ws, dg, src in timeline:
        key = re.sub(r"[\s，,。.、:：|]+", "", dg)[:50]
        if key in seen:
            continue
        seen.add(key)
        uniq.append((d, ws, dg, src))

    # 有日期的按时间倒序在前，无日期的垫底（否则 '未标日期' 因 CJK 码位高而排最前）
    uniq.sort(key=lambda x: (x[0] == "未标日期", x[0] if x[0] != "未标日期" else ""), reverse=False)
    dated = [u for u in uniq if u[0] != "未标日期"]
    undated = [u for u in uniq if u[0] == "未标日期"]
    dated.sort(key=lambda x: x[0], reverse=True)
    uniq = dated + undated

    lines = ["# 记忆流\n_只记「何时干过什么」，不写细节 | 本地模型维护 | 仅检索_\n"]
    for d, ws, dg, _src in uniq:
        lines.append(f"- {d} | tag={ws} | {dg}")
    stream_text = "\n".join(lines) + "\n"
    print(f"\n=== stream.md  {len(stream_text.encode()) / 1024:.1f}KB  "
          f"{len(uniq)} 行（原 {len(timeline)} 块去重后；有日期 {len(dated)}） ===")
    for ln in lines[2:12]:
        print("   ", ln[:130])
    print(f"    ... 共 {len(uniq)} 行")

    # ── 3. 工作区册：详细内容按类落位 ──
    # 与记忆流的分工：记忆流存「何时干过什么」的单行时间线；工作区册存**多行详情块**。
    # 曾把同一个多行块同时塞进两处（记忆流取首句、工作区存全文）→ 检索时
    # 同一事实被召回两次，一半注入量是重复（2026-10-05 实测）。
    # 现在按块的自然形态分流：单行短块归时间线，多行结构块归工作区册。
    ws_docs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for name in WS_SOURCES:
        p = M / name
        if not p.exists():
            continue
        for head, body in parse_blocks(p.read_text(encoding="utf-8", errors="replace")):
            if not body.strip():
                continue
            # ⚠️ 与记忆流同样的坑：曾写 `if is_meta(head) or not body.strip(): continue`，
            # 而 recent.md / summary.md 的标题是 `## 2026-10-04 15:53:01 | session=… | 纪要`
            # → is_meta 判 true → 这两份文件的工作区内容也被整份丢弃。
            # 规则同记忆流：只有空正文才是容器；带正文的时间戳标题只当「无标题」。
            if is_meta(head):
                head = ""
            # 短单行块属于时间线粒度，不进工作区册，减少与记忆流的重复召回
            if len(body.strip()) < 120 and body.count("\n") < 2:
                continue
            ws = classify(head, body)
            ws_docs[ws].append((head, body, name))

    print(f"\n=== 工作区册 ===")
    total = 0
    for ws in sorted(ws_docs):
        # 每条只保留摘要长度：工作区册是「详载」，但 180KB 全量入库会让
        # 单次检索轻易冲破 6KB 预算、每次都要走二次检索（2026-10-05 实测）。
        # 1400 字 ≈ 一个完整技术结论的篇幅，再长的原文有 *.legacy 兜底。
        body = "\n\n".join(
            f"## {heading_of(h, b)}\n{sanitize(clean(b))[:WS_DETAIL_LIMIT]}"
            for h, b, _ in ws_docs[ws]
        )
        text = (f"# 工作区 · {ws}\n"
                f"_详载本区工作内容 | 云端模型维护 | 仅检索_\n\n{body}\n")
        total += len(text.encode())
        src = defaultdict(int)
        for _, _, n in ws_docs[ws]:
            src[n] += 1
        print(f"   ws/{ws}.md  {len(text.encode()) / 1024:6.1f}KB  {len(ws_docs[ws]):3d} 条  "
              f"来源 {dict(src)}")
        if apply:
            C.ensure_dirs()
            (C.WS_DIR / f"{ws}.md").write_text(text, encoding="utf-8")
    print(f"   合计 {total / 1024:.1f}KB")

    if not apply:
        print("\n[预演] 未写入。加 --apply 执行。")
        return

    # ── 4. 落盘 + 旧文件转 legacy ──
    C.ensure_dirs()
    C.CORE.write_text(core_text, encoding="utf-8")
    C.STREAM.write_text(stream_text, encoding="utf-8")

    for name in ["fixed.md", "USER.md", "recent.md", "summary.md",
                 "long_term.md", "archive.md"]:
        src = M / name
        if src.exists():
            dst = M / f"{name}.legacy"
            if not dst.exists():
                shutil.copy2(src, dst)

    print("\n[已应用]")
    print(f"  core.md    {C.kb(C.CORE):.1f}KB")
    print(f"  stream.md  {C.kb(C.STREAM):.1f}KB")
    print(f"  ws/        {len(list(C.WS_DIR.glob('*.md')))} 册")
    print("  旧文件已复制为 *.legacy（未删除原文件）")


if __name__ == "__main__":
    main()
