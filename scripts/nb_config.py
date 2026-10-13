"""DBA 三层记忆 + 向量层配置 — 单一真源。

三层职责与写入者（写入者不同是本架构的核心约束）：

    core.md     核心记忆   云端模型写   上限 15KB   每轮全量注入
    stream.md   记忆流     本地模型写   不限长度    仅关键词检索（写「何时干过什么」，不写细节）
    ws/*.md     工作区册   云端模型写   不限长度    仅关键词检索（每项目目录一册，详载工作内容）

外加一层**向量记忆**（memory/.vec/）：与关键词索引**零耦合**，任一方挂了另一方照常工作。

与旧五级记忆（fixed/recent/summary/long_term/USER）的关系：那套已全部迁为 *.legacy，
不再读、不再写、不再注入；本模块是现役架构的唯一配置源。
"""

import os
from pathlib import Path

# ── 数据位置 ──
# 架构约束：**数据跟着代码走**，默认就在本仓库内。
#   为什么不再放 hermes 下（2026-10-06 迁移）：记忆是**可移植资产**，而 hermes 只是
#   某一台机器上某一款客户端程序的安装目录。把记忆埋在别人家目录里，等于要求所有
#   接入的 Agent 都装同一款客户端并遵守它的目录习惯 —— 换客户端就得搬一次家。
#   现在整目录 git clone / 拷走即用。
# 覆盖顺序（先命中先用）：
#   1) FOXNOTE_HOME / DBA_HOME 环境变量
#   2) 仓库内的 memory/（本文件位于 <repo>/scripts/，即上级目录）
ROOT = Path(
    os.environ.get("DBA_HOME") or os.environ.get("FOXNOTE_HOME")
    or Path(__file__).resolve().parent.parent
).resolve()

MEMORY = ROOT / "memory"
CORE = MEMORY / "core.md"
STREAM = MEMORY / "stream.md"
WS_DIR = MEMORY / "ws"
INDEX = MEMORY / ".index" / "inverted.json"

# ── 向量层（第三检索通道，与关键词索引零耦合） ──
VEC_DIR = MEMORY / ".vec"
VEC_INDEX = VEC_DIR / "vectors.npy"     # float32 [N, D]，与 VEC_META 行一一对应
VEC_META = VEC_DIR / "meta.json"        # 每行对应的段来源、行号、文本

# 嵌入走 **Ollama GPU**（用户硬性要求：绝不许退到 CPU）。
VEC_ENABLED = True
VEC_MODEL = "bge-m3"                    # 多语言，中文强，1024 维
VEC_OLLAMA_URL = "http://127.0.0.1:11434"
VEC_TOP_K = 5
# ⚠️ 余弦相似度用**绝对值**，不做「按本次最高分归一」——这点和 BM25 相反，理由：
#   BM25 是量纲浮动的排名分（换语料整体平移），所以那边只能按本次 top 归一；
#   而余弦相似度有物理意义（同语言相关段落 >0.6，不相关 <0.4），绝对值才可比。
#   把 BM25 那套归一逻辑照搬过来会得到「用最高分 0.3 的垃圾当满分」的假结果。
VEC_MIN_SCORE = 0.45

# ── 容量 ──
# 核心记忆：每轮全量注入，必须设限（用户定 15KB）
CORE_MAX_KB = 15
# 记忆流 / 工作区：磁盘不限长度，靠检索取用
STREAM_MAX_KB = 0   # 0 = 不限
WS_MAX_KB = 0       # 0 = 不限

# ── 检索 ──
# 一次检索返回条数与字节上限；超过上限触发二次检索（收窄到命中段落）
RETRIEVAL_TOP_K = 5
RETRIEVAL_MAX_KB = 6      # 命中总量超过此值 → 走二次检索收窄
NARROW_WINDOW = 2          # 二次检索时命中行上下保留的行数

# 相关度阈值 —— **相对本次检索最高分**的比例，不是绝对 BM25 分数。
#   0.0 = 关闭（默认）。设成 0.3 即「丢弃低于最高分 30% 的候选」。
# ⚠️ 这里曾写死 `RETRIEVAL_MIN_SCORE = 0.8`（绝对分），而它是个**空操作**：
#   实测 13 个真实查询（含 `的`/`机`/`表` 这类极泛词），top-5 候选**无一低于 0.8**，
#   阈值真正起作用的查询 = 0/13；低于 0.8 的候选只占候选总数的 0~11%，
#   且全部落在 top-5 之外 —— 既不滤垃圾，也不拦有用结果，白占一行配置。
#   根因：0.8 是拍出来的固定数，而 BM25 是**动态量纲**——随 idf/文档长度浮动，
#   实测同一份语料里中位数在 1.6~5.8、最大值 2.0~23.4，换一份语料整体又平移。
#   写死的绝对分早晚不是变成空操作就是误杀，所以改成按本次 top 分归一。
# ⚠️ 常量改名为 RETRIEVAL_MIN_SCORE_RATIO 是**故意**的：若沿用旧名，某个旧脚本
#   或旧配置里残留的 0.8 会被悄悄解释成「最高分的 80%」= 激进过滤，静默改变检索
#   结果。改名让残留引用直接 attribute error，而不是默默变脸。同理旧的
#   RETRIEVAL_MIN_SCORE 不保留为 0.0 别名——那等于允许旧值继续生效。
RETRIEVAL_MIN_SCORE_RATIO = 0.0

# 收窄后**只剩一条**却仍旧超预算时，要不要把这一条硬截断进预算？
#   True （默认）= 截断到预算内，并在末尾留下 `_TRUNC_MARK` 说明这段不完整。
#   False        = 不截断，但在 note 里如实写「单条命中超预算，未截断」。
# ⚠️ 两个值都合法，唯一**不允许**的是旧行为：什么都没做，却声称「已二次检索收窄」。
#   循环到 len(hits)==1 就必然退出，最后一条**永远丢不掉**——所以「收窄」这句话
#   对单条超预算的情形是假话。实测 max_bytes=1024 时返回 1154B（超 1.1 倍），
#   note 却仍写「已收窄到行窗口」。记忆正文被静默截断是本项目反复踩的坑
#   （见 core.md：哨兵值与真实取值必须分道），所以宁可留标记也不装没发生。
RETRIEVAL_CLIP_OVERBUDGET_HIT = True

# ── 时间权重（**只在二次检索时**施加） ──
# 规则（2026-10-06 用户定）：一次检索纯相关度；命中总量超预算 → 走二次检索，
# 此时叠加时间权重，**越近权重越大**，让新记忆在预算裁剪中活下来。
#
# 为什么一次检索不掺时间：一次检索决定「谁能进入候选集」，纯相关度更可预测。
# 掺进时间后，同一个查询在不同日子会给出不同的候选集，检索结果不可复现 ——
# 排查「为什么这次没召回」时无从下手。
TIME_DECAY_ENABLED = True

# 半衰期：每过这么多天，权重减半（w = 0.5 ** (age/halflife)）。
# 30 天 = 一个月前的记忆权重 50%，两个月前 25%。
TIME_DECAY_HALFLIFE_DAYS = 30.0

# ⚠️ 权重下限 —— 时间只能**倾斜**排序，不许**压倒**相关度。
#   实测（2026-10-06）：语料跨 101 天，最老段落权重 0.0969 → 它要有 **10.3 倍**
#   于最新段落的 BM25 才能排到前面。10 倍意味着「相关度差 10 倍以内，较新的必赢」，
#   于是一条勉强沾边的新记录会压掉一条精确匹配的旧记录 —— 而用户的痛点是
#   「旧事实被丢」，这恰好把痛点做成了机制。
#   截到 0.5 之后时间最多造成 2 倍倾斜：同等相关时新的赢，明显更相关时旧的仍赢。
#   时间权重是**排序的调节项**，不是排序的主导项。
TIME_DECAY_MIN_WEIGHT = 0.5

# ⚠️ 无日期段落取「中性权重」= 有日期段落的**中位数**（建索引时算好存进
#   corpus_dates.neutral_weight）。为什么不能简单给 1.0：
#   1.0 等于「它最新」—— 那是在**编造**日期。当前 523 段里 331 段无日期（63%），
#   给 1.0 就是让三分之二的记忆凭空获得最新优先权，时间权重反而把它们全顶上来。
#   为什么不给最小值（当很旧）：那是往另一个方向编造，把整个 ws 册压到不可见。
#   中位数 = 「不知道，按平均水平算」，既不加分也不减分 —— 这是**唯一诚实**的取值，
#   与项目一贯的「不知道 ≠ 某个具体值」一致。中位数与是否施加时间权重无关地存在，
#   所以它同时保证：没有日期的记忆不会被时间机制系统性排斥。

# 二次检索时把候选池放大到 top_k 的几倍，供时间权重重排。
# 为什么必须放大：只在原 top_k 内部重排，时间权重**什么都改变不了** ——
# 集合一样，只是顺序变了，而顺序在「按内容装不装得下」的裁剪里几乎决定不了
# 谁能活下来。放大池子才能让「相关度略低但更新」的记忆**升上来**。
TIME_POOL_FACTOR = 3

# ── 第三级：云端模型主动再检索 ──
# 规则第三段「云端模型如果觉得不清晰可以主动再次检索」。主观判断若不落成
# 可计算的信号，要么永不触发（形同虚设），要么乱触发（每轮多烧一次）。
# 这两条是「不清晰」的客观判据（详见 nb_search._looks_weak）：
RETRIEVAL_WEAK_TOP_SCORE = 6.0     # 最高分低于此值算「弱命中」
#   6.0 是**实测标定**的：14 个真实查询的最高分落在 4.25~35.12。
# ⚠️ 但它单独用会**误判**：`BOM`（最高分 5.68，命中确实精准——记忆里全是
#   BOM 相关内容）和 `Codex`（4.25，同样精准）都会被判弱。
#   原因是**短查询的通病**：查询词少 → 命中的词少 → BM25 累加项少 → 分数天然低。
#   这不是「不清晰」，只是「词少」。所以这一条从判据升级为**候选信号**，
#   只有配合「无共享词」一起出现才算弱（见 _looks_weak 的组合逻辑）。
RETRIEVAL_WEAK_MIN_TERMS = 3       # 短于这么多词的查询不适用「最高分低」判据
#   实测：BOM(1 词)/Codex(1 词)/向量索引(2 词) 属于短查询；静默失败(3 词)、
#   Claude Code(2 词 ASCII) 等长查询的高分才可比。短查询不参与分数判据 ——
#   **不能拿一把只对长查询标定过的尺子去量短查询**。
RETRIEVAL_MAX_RELAX_ROUNDS = 2     # 最多放宽几级（规格：≤2 次）
#   规格是「≤2 次」，所以默认 2：先试放开 scope，再试条数翻倍。
#   第 3 级（纳入 core）写成 max_rounds>2 才可达，即默认**不**执行 ——
#   保留能力但默认收着，因为「纳入 core」等于绕过「core 已全量注入、
#   不该再召回」的基本设计，只在显式要求时才做。

# ── 子串扩展（数字/ASCII 片段召回）的参数 ──
# 放在 nb_config 而不是 nb_index：nb_index **反过来 import nb_config**，
# 常量放这里两侧才都能读，也才能被 status 打印出来。
# （起初定义在 nb_index，status 想引用它就构成循环导入。）
#
# 阈值取 3：`2023`/`3055` 这类有意义的片段能扩展，而 `si`/`23` 这类两字符
# 会几乎匹配到全表（实测 `si` 精确可召回 SI 班级，不该被扩展冲淡）。
# 上限取 64：要装下 `2023` 这类四位数字的**全部**匹配键（实测 16 个）。
#   旧版截成 8 个，实测把 `20260` 真正相关的文档挤出了候选集
#   （它靠较长的文件名 token 命中，按长度排在第 8 位之后）。
# 词数上限 3：片段查询现实中是 1~2 个词，而每个扩展词要扫全表（16798 个键）。
#   无上限时「几百个查不到的词」的畸形查询会把每轮检索拖到 131ms；
#   加上限后 1.7ms。正常查询远达不到，只为挡住畸形长输入。
MIN_SUBSTR_LEN = 3
MAX_SUBSTR_EXPANSION = 64
MAX_SUBSTR_TERMS = 3

# ── 不进检索的记忆文件 ──
# `ws/角色卡.md` 存的是**角色扮演设定**（虚构世界、人设、NSFW 规则等），
# 它不是关于用户或工作的**事实**。放它进索引有两个坏处：
#   1. 它会被无关查询召回，占用 top_k 名额、还把伪装成事实的虚构设定
#      混进「记忆命中」里 —— 而记忆命中的可信度依赖于「它记的是真事」。
#   2. 语义通道尤其危险：那段文字量大、主题词密集，很容易压过真正相关的记忆。
# 这是**过滤**而不是删除：文件仍在 `memory/ws/` 里、仍可人工阅读，
# 只是不参与检索。用文件名匹配（`<name>.md`），避免误伤同名子串。
EXCLUDED_WS_FILES = ("角色卡.md",)

# ⚠️ 时间权重的**参照点**是「语料内最新日期」，不是 `datetime.now()`：
#   用挂钟意味着系统时间一错、或记忆停更一段时间，所有权重一起塌掉，
#   而此时权重看起来仍在正常工作（都能算出数），属于静默失效。
#   用语料自身最新日期则**自我校准**：最新的记忆恒为权重 1.0，
#   其余按相对新旧衰减，与挂钟无关。
# ⚠️ 但这也意味着：**语料越短，时间权重越不生效**。当前语料只跨 1 天
#   （2026-10-05~06），所有段落权重几乎都是 1.0 —— 此时它就是个空操作。
#   所以二次检索的 note 里必须**如实报告权重跨度**，让人看得出它是否真的起了作用，
#   而不是让人以为「加了时间权重」就一定在按时间排序。

# ── 注入总量预算 ──
# ⚠️ 组装**总量**从前无人过问，而下游是一道会把内容整段丢掉的硬闸门：
#    DSH 侧 `snowfox-inject.mjs` 的 MAX_BYTES = 64KB，超限的处理不是截断，
#    而是 `readSection()` 直接 return '' —— 记忆会在**没有任何报错**的情况下
#    整体消失，人和模型都只会觉得「这轮没想起什么」。
#    所以 64KB 必须在这里有一份镜像，让 Python 侧能在越界前喊出来。
INJECT_HARD_LIMIT_B = 64 * 1024

# 到硬上限的这个比例就告警。不等到 64KB 才喊的理由：二次检索收窄虽然能降量，
# 但它要等下一轮组装才生效，撞线那一轮的记忆已经丢了。
ASSEMBLY_WARN_RATIO = 0.75

# 组装产物里**非正文**的固定开销：头部注释行 + 「## 核心记忆」标题 + 段间空行
# + 检索段标题（不含标签本体）。实测约 400B，与段数相关而与正文长度无关。
# 它由 nominal_assembly_bytes() 消费，用于一眼看出预算还够不够。
ASSEMBLY_OVERHEAD_B = 400


class MissingSize(float):
    """「文件不存在」的专用返回值：-1.0，且带 .missing 标记。

    为什么不能再用 0.0（踩过的坑，别再改回去）：
        0.0 天然是「文件存在但为 0 字节」的**合法**大小，同时又拿它表示
        「文件不存在」，两种含义共用一个值 → 上游无法区分。实测后果：
        core.md 被删后 `kb(CORE) == 0.0`，组装头部平静地显示
        `core=0.0KB/15KB`，与「核心记忆真的为空」完全同形；而且
        `0.0 > CORE_MAX_KB` 恒为 False，**超限告警连带静默失效**。
        这是本项目第三次犯同类错（`date_of` 曾返回 '未标日期'，导致 96 条
        记录只有 1 条带上真日期）——失败与真实取值**必须分道**。

    取 -1.0 而不是 NaN：真实文件大小恒 >= 0，负值不可能是测量结果；
    且 `f"{-1.0:.1f}KB"` → `-1.0KB` 在日志里一眼可疑，而 NaN 会渲染成
    `nanKB` 并让 JSON 产出非法的 `NaN` 字面量。
    """

    __slots__ = ()
    missing = True

    def __format__(self, spec: str) -> str:
        """任何 f-string 插值都渲染成显眼错误态，而不是数值。

        ⚠️ 这一条是**零改动放大**的关键：调用方拿 kb() 的返回值直接拼进
        头部文案（如 nb_assemble 的 `{core_size_kb()}KB/{CORE_MAX_KB}KB`），
        若不做这里，缺失只会显示成 `-1.0KB` —— 可疑但仍像个数字，
        读的人很容易当成测量结果。做成 `__format__` 后，**所有**既有
        插值点自动变成错误态文案，无需逐个去改 nb_assemble.py。
        spec（如 `.1f`）在此被有意忽略：缺失没有精度可谈。
        """
        return "缺失!core.md不存在"


# 模块级单例：便于 `kb(p) is MISSING` 这类判断（但推荐用 is_missing()）
MISSING = MissingSize(-1.0)


def core_bytes() -> int:
    """core.md 的**精确字节数**。缺失时返回 -1（不是 0）。

    ⚠️ 为什么必须有这个函数，不能拿 kb() 去做上限判断：
    `kb()` 是**给人看的**，四舍五入到 0.1KB。于是 15361 字节 → 15.0009KB
    → round(...,1) → 显示 **15.0**，而 `15.0 > CORE_MAX_KB(15)` 是 **False**。
    超限因此被判为合格，**静默通过**，而且文件越大显示得越像没问题
    （16000 字节 → 15.6KB 才终于报警）。

    这与本项目已踩过的同类坑是同一个形状：用「给人看的近似值」去做
    「决定成败的判断」。容量是硬约束，判断必须用原始字节。
    显示仍然用 kb()/kb_label()，两者分工不能混。
    """
    try:
        if not CORE.exists():
            return -1
        return CORE.stat().st_size
    except OSError:
        return -1


def core_bytes_of(text: str) -> int:
    """一段**待写入**文本作为 core 内容时的字节数（用于写盘前判断）。

    编码固定 utf-8 —— 与所有写入路径一致。中文一个字 3 字节，
    所以「字符数」完全不能代表容量，必须按字节算。
    """
    return len(text.encode("utf-8"))


def core_limit_bytes() -> int:
    """core.md 的硬上限（字节）。"""
    return CORE_MAX_KB * 1024


def over_core_limit(nbytes: int) -> int:
    """超出上限多少字节？未超返回 0（而不是负数），便于直接拿去做判断/文案。"""
    return max(0, nbytes - core_limit_bytes())


def kb(path: Path) -> float:
    """文件大小（KB）。文件不存在时返回 MISSING（不是 0.0）。

    ⚠️ exists() 与 stat() 分开调用有 TOCTOU 窗口（junction 目标被拔掉时
    也会抛 OSError），所以 stat 失败一律按「缺失」处理，不向上抛——
    记忆损坏不该连累整轮组装。
    """
    try:
        if not path.exists():
            return MISSING
        return round(path.stat().st_size / 1024, 1)
    except OSError:
        return MISSING


def is_missing(path: Path) -> bool:
    """文件不存在（或读不到）吗？——把「缺失」与「0 字节」分开问。"""
    return getattr(kb(path), "missing", False)


def kb_label(path: Path) -> str:
    """给人看的大小标签。缺失时返回「缺失」而不是 0.0，杜绝两者同形。"""
    return "缺失" if is_missing(path) else f"{kb(path):.1f}"


def nominal_assembly_bytes(core_bytes: int, hits_bytes: int) -> int:
    """按各部分字节数推算组装产物总量（含固定开销）。

    消费 ASSEMBLY_OVERHEAD_B：它此前是**零引用死配置**，导致「总量」这件事
    在 Python 侧根本没有对应物，而下游 64KB 是一道丢内容的硬闸门。
    """
    return core_bytes + hits_bytes + ASSEMBLY_OVERHEAD_B


def check_assembly_budget(total_bytes: int) -> str:
    """检查组装总量是否逼近注入硬上限，返回告警文案（正常时为空串）。

    调用方（nb_assemble.assemble）应在产物里带上这段文案。
    为什么必须提前喊：snowfox-inject.mjs 对超限段的处理不是截断而是
    `return ''` —— 整段记忆凭空消失且**不报错**，事后无从追查。
    所以越线前就要有人说话，且要在头部留下痕迹。
    """
    if total_bytes > INJECT_HARD_LIMIT_B:
        return (f"🚨 组装总量 {total_bytes / 1024:.1f}KB 已超 {INJECT_HARD_LIMIT_B // 1024}KB "
                f"注入硬上限 → DSH 侧会**整段丢弃**（记忆静默消失），必须立刻缩减")
    warn_at = int(INJECT_HARD_LIMIT_B * ASSEMBLY_WARN_RATIO)
    if total_bytes >= warn_at:
        return (f"⚠️ 组装总量 {total_bytes / 1024:.1f}KB 已接近 {INJECT_HARD_LIMIT_B // 1024}KB "
                f"注入硬上限（{warn_at // 1024}KB 起告警），超限即整段丢弃")
    return ""


def ensure_dirs() -> None:
    MEMORY.mkdir(parents=True, exist_ok=True)
    WS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    if VEC_ENABLED:
        VEC_DIR.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    ensure_dirs()
    ws = sorted(p.name for p in WS_DIR.glob("*.md"))
    # core 缺失必须是**显眼错误态**：它是每轮全量注入项，没了等于记忆全哑，
    # 而旧写法会把它显示成平静的 `now 0.0KB`，与「核心记忆真的为空」同形。
    core_show = ("🚨 缺失! core.md 不存在 → 全量注入失效" if is_missing(CORE)
                 else f"{kb(CORE):.1f}KB")
    vec_show = "关闭"
    if VEC_ENABLED:
        try:
            import nb_vec
            vec_show = nb_vec.stats()
        except Exception as e:
            vec_show = f"不可用（{type(e).__name__}）"
    lines = [
        "DBA 三层记忆 + 向量层:",
        f"  root    {ROOT}",
        f"  core    {CORE.name}      <= {CORE_MAX_KB}KB  (云端模型写, 全量注入)   now {core_show}",
        f"  stream  {STREAM.name}    不限          (本地模型写, 检索)         now {kb_label(STREAM)}KB",
        f"  ws/     {len(ws)} 册        不限          (云端模型写, 检索)",
    ]
    for name in ws:
        lines.append(f"          - {name}  {kb_label(WS_DIR / name)}KB")
    lines.append(f"  向量    {vec_show}")
    lines.append(f"  检索: top_k={RETRIEVAL_TOP_K}, max={RETRIEVAL_MAX_KB}KB, "
                 f"min_score_ratio={RETRIEVAL_MIN_SCORE_RATIO}"
                 f"{'（关闭）' if RETRIEVAL_MIN_SCORE_RATIO <= 0 else ''}")
    # ⚠️ 上面那行只报**参数**，报不出**流程** —— 而 2026-10-06 起检索默认走三级阶梯
    # （`mode="ladder"`）。只印参数会让人以为「就是一次 BM25」，
    # 于是「为什么这次答案多了一段无关内容」无从解释（那是第 2/3 级干的）。
    # 阶梯是架构的一部分，状态行必须说清楚。
    lines.append("        流程: 三级阶梯 ①BM25 top_k（纯相关度，候选集可复现）"
                 " ②命中 >6KB → 行窗口收窄 + 按 相关度×时间权重 重排"
                 f" ③不清晰 → 放开 scope / top_k 翻倍（≤{RETRIEVAL_MAX_RELAX_ROUNDS} 次）")
    # 子串扩展也印出来：它是「数字片段能查到完整 ID」的原因，且**有闸门**
    # （只在精确词无候选时才启用）。不说的话，使用者看到 `2023` 命中学号，
    # 会以为索引对数字做了前缀匹配，进而依赖一个并不存在的性质。
    lines.append(f"        数字片段: 子串扩展 ≤{MAX_SUBSTR_TERMS} 词"
                 f"（仅当精确词无候选时启用，避免污染精确排名）")
    # 组装总量的**下界** = core + 固定开销（不含本轮检索命中）。检索命中与 core
    # 共用同一个 64KB 硬上限，所以 core 一侧必须先把开销算进去，否则会误以为
    # 「15KB 的 core 离 64KB 还很远」——实际上命中段随时能再吃掉几 KB。
    core_b = 0 if is_missing(CORE) else CORE.stat().st_size
    floor_b = nominal_assembly_bytes(core_b, 0)
    budget = check_assembly_budget(floor_b)
    lines.append(f"  注入: 组装硬上限 {INJECT_HARD_LIMIT_B // 1024}KB"
                 f"（{int(INJECT_HARD_LIMIT_B * ASSEMBLY_WARN_RATIO) // 1024}KB 起告警）"
                 f" | core+开销下界 {floor_b / 1024:.1f}KB"
                 + (f"\n        {budget}" if budget else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(summary())
