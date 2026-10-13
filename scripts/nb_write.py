"""记事本写入 CLI — 按写入者分口（本地模型 / 云端模型）。

用户定的写入者分工：
    本地模型  → 记忆流 stream.md   只写「什么时间点干过什么」，一句话一条，不写细节
    云端模型  → 核心记忆 core.md   硬事实/约束/偏好；超 15KB 时自行精简，细则下沉到工作区
    云端模型  → 工作区 ws/<区>.md  某项目目录内的工作详情

三个子命令：
    stream   追加一条记忆流         (本地模型用)
    core     覆盖/追加核心记忆       (云端模型用)
    ws       追加工作区册条目        (云端模型用)

命令行：
    py nb_write.py stream "完成了双脑Agent架构评审"
    py nb_write.py stream "..." --at "2026-10-05 22:30" --tag 架构
    py nb_write.py ws "双脑Agent" "定义了三层记忆..." --title "三层记忆定义"
    py nb_write.py core --file new_core.md        # 用文件整体覆盖（云端模型重写核心记忆）
                                                 # ⚠️ 缩水过半会被拒，需再加 --force 明确确认
    py nb_write.py core "新硬事实..." --title "路径"           # 追加一段（默认行为）
    py nb_write.py core --append "新硬事实..." --title "路径"  # 同上，显式写法
    py nb_write.py list                            # 查看三层现状
    py nb_write.py reindex                         # 重建索引

参数约定（踩过的坑，改前先读）：
    --at 不传 = 用当前时间；显式传 "" = 报错。「没传」和「空值」必须是两条通道，
    否则 `at or 当前时间` 会把空串静默变成假时间戳，而排序/检索正是靠时间戳。
    同理空 area、空正文一律报错，不做兜底 —— 兜底会让不同来源的数据糊进同一个
    文件，事后无法分辨（misc.md 已经是一本真实工作区册了）。
    core 的 --file 与 --append 互斥，同时给出直接报错，不替用户猜意图。

退出码：0 成功｜1 参数非法或内容为空，已拒绝（一个字节都没写）｜
        2 内容已写但索引重建失败（索引是检索的唯一来源，此时检索会漏掉刚写的内容）。
        —— 成功与失败必须是能分辨的信号，不能都从 stdout 读一行文本。

写后自动重建索引（索引小，毫秒级），保证检索立即可用。
"""

import argparse
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_config as C


class WriteError(Exception):
    """调用方参数非法或内容为空 —— 已拒绝，一个字节都没写。"""


class IndexRebuildFailed(Exception):
    """索引重建失败。**注意语义**：内容通常已经落盘，只是检索暂时读不到它。"""


def _ensure_files() -> None:
    C.ensure_dirs()
    if not C.CORE.exists():
        C.CORE.write_text(
            "# 核心记忆\n_全量注入 | 上限 15KB | 云端模型维护_\n\n"
            "## 用户\n- 示例用户，学号202301230000，机器人SI 23-2班\n",
            encoding="utf-8",
        )
    if not C.STREAM.exists():
        C.STREAM.write_text(
            "# 记忆流\n_只记「何时干过什么」，不写细节 | 本地模型维护 | 仅检索_\n",
            encoding="utf-8",
        )


# Windows 保留设备名（大小写不敏感）。CON/PRN/AUX/NUL 精确匹配，
# COM0-9 / LPT0-9 用前缀匹配即可。本次建 'CON' 侥幸成功，
# 但保留名的失败方式随 Windows 版本/打补丁变化 —— 不靠运气，直接拦。
# ⚠️ 旧的「COM1.txt 同样打不开」这句注释已被实测推翻：'COM1.md' 之类**带扩展名**
#    的路径在盘上是真文件（见 is_reserved_device 的实测表），真正会被设备吃掉的是
#    **裸设备名**（'CON'/'NUL'/'CON.'），它们 open() 报成功但盘上没有文件 = 静默丢数据。
RESERVED = {"CON", "PRN", "AUX", "NUL"}
RESERVED_PREFIX = ("COM", "LPT")
ILLEGAL_CHARS = '\\/:*?"<>|'


def is_reserved_device(name: str) -> bool:
    """兼容包装：名字是否命中 Windows 保留设备名。判定见 reserved_device_name()。"""
    return reserved_device_name(name) is not None


def reserved_device_name(name: str) -> str | None:
    """命中保留设备名则返回**被判定成设备的那个基名**，否则返回 None。

    返回基名而不是 True/False：报错文案要用它，而它必须与判据同源。
    旧写法在 safe_area 里另用 `raw.rpartition(".")[0]` 重算一遍基名，等于把
    「基名怎么取」这条实测规则抄了两份 —— 将来只改一处就会出现
    「拦住了 CON.txt，却报成基名 'CON.txt'」这种自相矛盾的提示。

    ⚠️ 踩过的坑（2026-10-05 独立复核实测）：旧实现在 safe_area 里直接拿**整个**
    raw 去比 RESERVED / 前缀，结果 `safe_area('CON')` 正确抛错，而
    `safe_area('CON.txt')` / `PRN.txt` / `AUX.txt` / `NUL.txt` **全部静默返回原名**。
    所以判定必须在**剥掉扩展名之后**做。

    为什么保留两条判据、两条都要在剥扩展名后重新适用：
      1) 基名恰好等于 CON/PRN/AUX/NUL/COM1-9/LPT1-9；
      2) 基名是 'COM'/'LPT' + **单个**数字（COM1..COM9 / LPT1..LPT9）。
      旧代码把判据 2 写成「前 3 位是 COM/LPT 且第 4 位是数字」，等价于只要求
      「至少一位数字」，故必须配合 `len(raw) >= 4` 才不会误伤 'com' 本身。

    ⚠️ 不许把这里放宽成 startswith：'console.md' / 'COM.md' / 'com10.md'
    （COM 后跟两位数字）都是**正常名字，必须继续放行**，它们不是设备名。
    判定依据只能是「基名**完全等于**保留名」或「基名是 COM/LPT + **单个**数字」。

    ⚠️ 基名怎么取 —— 实测（%TEMP%/nb_devicename_law.py，本机 Win11 26200；
      判据用**磁盘真实状态**，不看 open() 是否抛错）：
        裸 'CON' / 'NUL' / 'CON.' / 'NUL.' / 'CON..'  → open() 报成功但**盘上没有文件**
                                                        （字节被送进设备 = 静默丢数据）
        裸 'PRN' / 'AUX' / 'COM1' / 'LPT1'            → open() 直接 errno 2（硬失败）
        'CON.md' / 'NUL.txt' / 'CON..md' / 'CON.txt.log' → 都是**盘上的真文件**
      即：结尾的点会被 Windows 规范化掉，'CON.' 与 'CON' 是同一个设备；
      而一旦带了扩展名就正常落盘。所以基名取「**剥掉末尾的点后的首段**」：
      'CON.' → 'CON'（真丢数据，必修）；'CON.md' → 'CON'（保守多拦一档，
      代价可忽略：没人会真的拿设备名当项目名）；'com10.md' → 'com10' 正常放行。
    """
    # 先剥掉末尾的点：Windows 规范化会吃掉结尾的点，'CON.' 与 'CON' 是同一个设备
    # （注意 safe_area 另有 strip(".") 检查会把 'CON.' 拦在更早一步，但本函数
    #  作为纯函数必须自己成立，不能依赖调用方先做过那次检查）
    core_name = name.rstrip(".")
    # 再取「第一个点之前」当基名：多段扩展名（'CON.txt.log'）统一折算到首段
    up = core_name.partition(".")[0].upper()
    if up in RESERVED:
        return up
    # COM/LPT + 恰好一位数字；'com10' / 'comX' / 'console' 落到这里必然为 None
    if len(up) == 4 and up[:3] in RESERVED_PREFIX and up[3].isdigit():
        return up
    return None


def safe_area(area: str) -> str:
    """工作区名 → 合法文件主干名。非法就报错拒绝，**不做静默删除**。

    ⚠️ 旧实现 `"".join(ch for ch in area if ch not in '\\\\/:*?"<>|').strip() or "misc"`
    实测（2026-10-05 审查）：'a/b' 和 'a:b' 都被**静默删掉非法字符变成 'ab'**，
    与本来就合法的 'ab' 撞进同一个 `ab.md` —— 两个不同项目的工作区互相覆盖，
    且覆盖时没有任何提示。注意 'ab' 本身是**合法名字**（测试里确认它能正常写），
    所以这不是「都拒绝掉」的问题，而是「非法输入不该悄悄变成另一个合法名字」。
    这是**静默错误**，比报错危险得多：报错会被立刻看见，坍缩只会让数据悄悄消失。

    另外旧代码用 `or "misc"` 把「非法/空 area」的兜底**也写进 misc.md**，而
    `append_ws("misc")` 是合法调用、写的是同一本册子 —— 真实语料里 misc.md
    已是一本 16.3KB 的真实工作区册（2026-10-05 实测 16749B），两条来源在里面
    再也分不开。和 date_of 的 '未标日期' 是同一个病：**哨兵值绝不能和「无值」
    （这里是「真的叫 misc」）共用同一个返回通道**。空 area 不是「要用 misc」，
    而是「调用方少传了参数」，必须报错。
    """
    raw = (area or "").strip()
    if not raw:
        raise WriteError("ws 的 area 为空：工作区名必填（不会替你兜底到 misc）")
    if raw in (".", ".."):
        raise WriteError(f"ws 的 area 非法: {raw!r}")
    bad = sorted(set(raw) & set(ILLEGAL_CHARS))
    if bad:
        # 带上被拒的字符，调用方一眼能看出自己踩了哪个
        raise WriteError(f"ws 的 area 含非法字符 {bad}: {raw!r}（不做静默删除，请改名）")
    if raw != raw.strip("."):
        raise WriteError(f"ws 的 area 不能以点开头或结尾: {raw!r}")
    if any(ch in raw for ch in ("\r", "\n", "\t")):
        raise WriteError(f"ws 的 area 含控制字符: {raw!r}")
    dev = reserved_device_name(raw)
    if dev is not None:
        # 文案带上被判定成设备的基名：'CON.txt' 的调用方一眼看出拦的是 CON 设备
        # 而不是文件名本身。dev 是已 upper 的基名，故对 raw 也 upper 后再比，
        # 只有「raw 本来就是裸设备名」时才不追加括号说明（'con' 也不追加）。
        raise WriteError(
            f"ws 的 area 是 Windows 保留设备名: {raw!r}"
            + (f"（扩展名无效，基名 {dev} 仍是设备名）" if dev != raw.upper() else "")
        )
    return raw


def _stamp(at: str | None) -> str:
    """时间戳：**None（没传）** 才用当前时间；空串/空白串是「传了个空值」→ 报错。

    ⚠️ 旧实现 `at or time.strftime(...)` 让 `_stamp("")` 和 `_stamp(None)` 一起
    静默返回当前时间：调用方以为写进去了空值，实际写进一个**假的当前时间**，
    而排序与检索正是按时间戳定位的 —— 假时间戳比空值更难查。
    「没传」与「空值」必须是两条通道（同上，项目里这类坑已出现过多次）。
    """
    if at is None:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if not at.strip():
        raise WriteError("--at 给了空值：想用当前时间就不传 --at，不传 ≠ 传空串")
    return at.strip()


# ── 记忆流的内容闸门 ──
# 记忆流只该记「何时干过什么」。迁移脚本那边早就有这套规则（nb_migrate.is_junk），
# 但**写入侧一直没有**，于是实测混进了两类垃圾（2026-10-06 复查 stream.md 发现）：
#   对话片段：`5.5呢？   测完了，结论很干脆 —— **5.5 进不了 DSH**`
#   系统报错：`Hermes hit repeated errors and stopped this turn so it wouldn't...`
# 前者是没头没尾的问答残片，后者是运行时报错文本 —— 两者都不是「干过什么」，
# 却会永久留在检索库里、并占掉命中名额（检索是 top_k 截断的）。
# 判定要窄：宁可漏过，不可错杀真实的工作记录。
_STREAM_JUNK = (
    re.compile(r"^Hermes hit repeated errors", re.I),
    re.compile(r"^(Error|Traceback|Exception|Warning)\b"),
    re.compile(r"hit repeated errors and stopped", re.I),
    re.compile(r"^\s*(ok|done|yes|no|好的|收到|继续|可以|嗯)[。！？!?]?\s*$", re.I),
    re.compile(r"^\s*(https?://|sk-|ghp_)\S+\s*$", re.I),
)
# ⚠️ 2026-10-06 复查又抓到第三类：**助手对用户说的话**（英文/中文自语式残句）。
# 实测进了三条：`Now let me delegate the independent review to sol6.1 while I run
# the full regression:`、`All three suites green. Now let me retry the independent
# review with the correct call mech…`。这类东西**很长、很像正经记录**，所以
# 长度与「空话」规则都拦不住；但它记的是「我打算干什么」而不是「干过什么」，
# 下一个人检索到它等于读到一句没兑现的话。
# ⚠️ 第一版这三条规则**误杀了真实记录**（sol6.1 独立复核实测，我自己的测试样本太窄）：
#   `1. 已完成机械建模`      → 被「编号开头」规则判成垃圾
#   `2026.10.06 完成索引重建` → 同上（`2026.` 也匹配 `\d+\.`）
#   `我们先完成了索引修复并通过测试` → 被「我们先」规则判成自语
#   `本次复核结果如下：`      → 被「结尾冒号」规则判成计划句
# **这条教训比 bug 本身重要**：我为了拦住几句自语，差点把记「干过什么」的路
# 一起堵死。闸门错杀 = **真实记忆静默丢失**，而丢的东西肉眼看不见（不像垃圾
# 那样会在检索时冒出来被删掉）。所以规则一律加「完成标记」豁免，且宁可漏过。
#
# 现行判据（三条，全部带豁免）：
#   1. 明确的计划标志词（let me / i'll / 让我 / 接下来 …）**且**全句无完成标记；
#   2. 加粗编号的「回答体」条目（`1. **X**`）**且**无完成标记；
#   3. `let me know` 不算计划 —— 那是对用户说的请求，不是自语。
# ⚠️ 已删除的两条旧规则及其原因：
#   · 「编号/加粗开头就拒」→ 误杀 `1. 已完成机械建模`、`2026.10.06 完成索引重建`
#   · 「结尾是冒号就拒」  → 误杀 `本次复核结果如下：`、`任务完成，日志标签为：`
#     且它对拦住真垃圾**毫无贡献**：那两条真垃圾都是靠 `Now let me` 拦下的。
_STREAM_PLAN_EN = re.compile(
    r"(?:^|[\s。！？.!?）)】」])\s*(?:now\s+)?"
    r"(?:let\s+me(?!\s+know)|i'?ll|i\s+will|i'?m\s+going\s+to|next,?\s+i)\b", re.I)
_STREAM_PLAN_ZH = re.compile(
    r"(?:^|[\s。！？.!?）)】」])\s*(?:让我|我来|接下来|现在我们来|我们先|首先我)")
# 「回答体」：加粗编号条目（`1. **SEVERITY: BUG** …` / `**FILE:** …`）。
# ⚠️ 这条规则**曾经误杀真实记录**，已收窄（flash 穷举复测 + 我复验）：
#   `**这把 key 是 Study1 的** —— …`、`**结论**：时间权重下限取 0.5`
#   都是货真价实的记录，却因为「开头是 `**`」被拒。
#   而 `**` 开头正是本助手写复核/结论的**固定风格**，实测 stream.md 里
#   已有 2 条真实条目因此写不进去（静默丢失）。
#   现在要求 `**内容**：` 后**紧跟**一条「键: 值」式断言才算回答体碎片 ——
#   真垃圾 `1. **SEVERITY: BUG** **FILE:** …` 依然拦得住，
#   而 `**结论**：X` 这种带实质结论的句子放行。
#   权衡依据：错杀=记忆静默丢失（不可见），漏过=下次检索能看见并删掉。
_STREAM_ANSWER = re.compile(r"^\s*(?:\d+\.\s*)?\*\*[^*]{1,20}\*\*\s*[:：]?\s*"
                            r"(?:\*\*[A-Z][A-Z_ ]{2,}\*\*|FILE|SEVERITY|BUG)", re.I)
# 出现这些 → 这句在讲**已经做完**的事，不管有没有「让我」，都放行。
# ⚠️ 完成标记必须是**完整体**，不能用裸词。两轮独立复核各抓到一个方向的错：
#   · 裸 `已` → `接下来把**已**知的问题整理一下` 被判成「已完成」而放行（漏过计划句）
#   · 英文裸 `done` → 命中了 `abandon**done**d`，`Next, I abandoned the retry` 漏过
#   · 裸 `完成` → `接下来**完成**索引`、`现在我们来**完成**部署` 放行（sol6.1 复测）
#   · 中文裸 `已` 修过头后，`已修复`(6 点) 又被信息量闸门拦掉（flash 复测）
# 根因：**「完成」既是动词也能是完成态**。`接下来完成索引` 是「去做」，
# `我们先完成了索引修复` 是「做完了」—— 区别在**体标记**（了/过/已）。
# 所以完整体只认两种形态：
#   ① 体标记在前：已/均已/全部/都 + 动词   （已完成、已修复）
#   ② 体标记在后：动词 + 了/过             （完成了、修复了、通过了）
# 两者都不满足的裸动词一律不算完成态 —— 它在计划句里就是「打算做」。
_STREAM_DONE = re.compile(
    r"(?:已|均|全部|都)(?:完成|修复|提交|推送|建立|通过|跑通|搞定|生效|上线|"
    r"记录|删除|加上|改好|验证|确认|复现|测试|修好|落库|记下)"
    r"|(?:完成|修好|修复|提交|推送|建立|通过|跑通|搞定|生效|上线|"
    r"记录|删除|改好|验证|确认|复现|测试|落库)(?:了|过)"
    r"|\b(?:verified|passed|completed|fixed|done|built|shipped|committed|created|added|removed)\b",
    re.I)
_PLAN_SCAN_CHARS = 80
# 计划句只看开头这么长：正文中段出现「让我」是正常叙述，不该被牵连。

# ⚠️ 长度闸门按**字符数**数，对中文是错的：一个汉字承载的信息远多于一个
# 拉丁字母。`已修复`、`提交并推送` 都是完整的工作记录，却撞上「<8 字符=碎片」
# 被拒（flash 穷测，实测 11 条中 6 条被误杀）。
# 判据改为按**信息量**折算，且**只有承载信息的字符才计分**：
#   CJK 一字 2 点（一个字就够定位一件事）；拉丁/数字一字 1 点；
#   **空白与标点 0 点** —— 它们不承载任何「干过什么」的信息。
# 早期版本给标点记 1 点，于是 `........`（8 个点）被折算成 8 点、顺利通过
# （sol6.1 复测抓到）。**空话与纯标点靠这层拦，不该靠词表**。
# 门槛取 8 点 ≈ 4 个汉字（或 8 个字母），但见下面的短记录豁免。
def _info_weight(s: str) -> float:
    """折算信息量：CJK 一字 2 点，其余**有内容的**字符 1 点，空白/标点 0 点。"""
    total = 0.0
    for ch in s:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        if unicodedata.category(ch).startswith(("S", "C", "Z")):
            continue          # 符号（emoji/数学符）、控制符、分隔符也不计
        total += 2.0 if "\u4e00" <= ch <= "\u9fff" else 1.0
    return total


_MIN_INFO = 8.0
# ⚠️ 短记录豁免：`已修复`/`已提交`/`跑通了` 是**极短但完整**的工作记录
# （3 个汉字 = 6 点 < 8），实测被误杀两轮。它们没有细节，但「何时干过什么」
# 这件事本身说清楚了 —— 记忆流要的正是这个粒度。
# 豁免条件必须**同时**满足两条，否则会放进碎片：
#   ① 含完成态标记（_STREAM_DONE）—— 说明它在讲已做完的事，不是当场自语；
#   ② 无计划标志词             —— 排除「接下来完成索引」这种（它有完成动词却没完成态）。
_STREAM_SHORT_OK = re.compile(r"^.{0,16}$")


def _stream_junk_reason(flat: str) -> str:
    """这条不该进记忆流的原因；可以写入时返回空串。

    ⚠️ 方向性纪律：**宁可漏过几句自语，也绝不误杀真实记录**。
    漏过的垃圾下次检索读到时就能删掉（可见）；误杀的记忆是静默丢失（不可见）。
    两类错误的代价不对称，所以判据一律取窄 + 加完成标记豁免。
    """
    for pat in _STREAM_JUNK:
        if pat.search(flat):
            return f"匹配垃圾规则 {pat.pattern!r}"
    iw = _info_weight(flat)
    # ⚠️ 顺序很重要：先判「是不是自语」，再判「够不够长」。
    # 反过来会把 `接下来完成索引`（完成动词、无完成态）当成短记录豁免放行。
    head = flat[:_PLAN_SCAN_CHARS]
    done = bool(_STREAM_DONE.search(flat))
    planned = None
    if not done:
        for pat in (_STREAM_PLAN_EN, _STREAM_PLAN_ZH, _STREAM_ANSWER):
            if pat.search(head):
                planned = pat
                break
    if planned is not None:
        return (f"开头像是「打算做某事」的自语（匹配 {planned.pattern[:34]!r}）"
                f"　← 记忆流要记的是已完成的事实")
    if iw < _MIN_INFO:
        # 极短但**已完成**的记录（`已修复`、`跑通了`）放行：它们没有细节，
        # 但「何时干过什么」这件事本身说清楚了。已排除计划句（上面刚返回）。
        if done and _STREAM_SHORT_OK.match(flat):
            return ""
        return (f"信息量不足（{len(flat)} 字 / 折算 {iw:g} 点 < {_MIN_INFO:g}）"
                f"，不足以定位「干过什么」")
    return ""


def append_stream(text: str, at: str | None = None, tag: str = "", session: str = "") -> str:
    """追加一条记忆流。保持单行一条，便于检索收窄。

    ⚠️ 会拒收两类内容（见 _STREAM_JUNK）：对话残片与系统报错。
    要记细节请用 `ws`（工作区册不限长度、就是用来存细节的），
    不要把它们塞进记忆流 —— 记忆流的价值在于「一行一条、可被检索定位」。
    """
    _ensure_files()
    flat = " ".join(text.split())
    if not flat:
        raise WriteError("stream 正文为空：拒绝写入（空行会在记忆流里留下无信息的条目）")
    reason = _stream_junk_reason(flat)
    if reason:
        raise WriteError(
            f"stream 拒收：{reason}\n"
            f"  内容: {flat[:100]}\n"
            f"  细则请写 ws 册：py nb_write.py ws <区> \"...\""
        )
    ts = _stamp(at)
    suffix = f" | tag={tag}" if tag else ""
    suffix += f" | session={session}" if session else ""
    with open(C.STREAM, "a", encoding="utf-8") as f:
        f.write(f"- {ts}{suffix} | {flat}\n")
    return f"stream += {flat[:60]}"


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n{body.strip()}\n"


def _core_size_msg(kind: str, nbytes: int) -> str:
    """core 写入后的尺寸回执。超限时**明确标红**，且用精确字节说话。

    ⚠️ 这里刻意不再用 `C.kb()` 做判断：它四舍五入到 0.1KB，15361 字节会被
    显示成 15.0KB，`15.0 > 15` 是 False —— 超限**静默通过**。判断一律走
    `C.core_bytes()`（原始字节）。显示仍用 KB，但结论基于字节。
    """
    over = C.over_core_limit(nbytes)
    shown = nbytes / 1024
    if over > 0:
        return (f"core = {shown:.2f}KB  ⚠️ 超上限 {over} 字节"
                f"（硬限 {C.CORE_MAX_KB}KB={C.core_limit_bytes()}B）"
                f" → 需云端模型摘要压缩")
    return f"core = {shown:.2f}KB / {C.CORE_MAX_KB}KB（余 {C.core_limit_bytes() - nbytes} 字节）"


def append_core(text: str, title: str = "追加") -> str:
    """向核心记忆追加一段。

    ⚠️ 超限**不阻止写入，但绝不静默** —— 这是权衡后的选择，不是疏忽：
    core 由云端模型维护，模型重写 core 时是「先写全量、再逐步压缩」的，
    若在写入侧硬拒绝，模型就会丢掉它刚整理好的内容（那是**丢记忆**）。
    所以策略是：照写，但把「超了多少字节」明确回执给调用方，
    由下一个回合的云端模型压缩。

    ⚠️ 但**判断**必须用精确字节，不能用心跳式的 KB 近似值（见 _core_size_msg）。
    """
    _ensure_files()
    # ⚠️ 空正文必须拒绝：实测 append_core("") 会在 core.md 尾部留下
    # `\n## 追加\n\n` 的空段 —— parse_sections 会跳过它，但文件已经被污染，
    # 而且每次调用都再加一个，滚起来没人看得出来。
    if not text.strip():
        raise WriteError("core 正文为空：拒绝写入（会在 core.md 里留下空段污染文件）")
    cur = C.CORE.read_text(encoding="utf-8", errors="replace")
    if f"\n## {title}\n" in cur:
        # 同标题追加到该段末尾
        head, _, rest = cur.partition(f"\n## {title}\n")
        nxt = rest.find("\n## ")
        if nxt >= 0:
            cur = head + f"\n## {title}\n" + rest[:nxt].rstrip() + f"\n{text.strip()}\n" + rest[nxt:]
        else:
            cur = head + f"\n## {title}\n" + rest.rstrip() + f"\n{text.strip()}\n"
    else:
        cur = cur.rstrip() + "\n" + _section(title, text)
    C.CORE.write_text(cur, encoding="utf-8")
    # 回执按**实际落盘后的文件字节**算，而不是按我们以为写了多少
    return _core_size_msg("append", C.core_bytes())


# core.md 整体替换的缩水闸门比例：新内容不足旧内容的这个比例 → 判为「疑似误用 --file」。
# 0.5 是权衡后的取值：压缩 core 是 replace 的正当用法，但那通常是 15KB 量级
# 的整篇重写，缩到**不足一半**几乎总是「把 --file 当成追加用」。
_CORE_SHRINK_RATIO = 0.5


def _disk_bytes(text: str) -> int:
    """一段文本落盘后的**真实字节数**，用于跟 `C.core_bytes()` 同尺度比较。

    ⚠️ 不能直接用 `len(text.encode())`：`write_text` 在 Windows 上会把 `\\n`
    转成 `\\r\\n`，每行多一个字节。旧内容是磁盘上的 core.md，新内容还在内存里，
    两者必须换算到同一把尺子上；否则每行 1 字节的系统性偏差会让闸门在临界点上
    判错方向。这与 `kb()` 那个坑同源：**比较用的尺子必须和判断依据一致**。
    """
    return len(text.encode("utf-8")) + text.count("\n")


def replace_core(src: str, force: bool = False) -> str:
    """整体替换核心记忆（云端模型重写时用）。

    ⚠️ **缩水闸门**（2026-10-11 加，实测事故驱动）：`--file` 是**整体覆盖**，
    但它极易被当成「从文件读正文来追加」。真实事故：

        py nb_write.py core --title "DeepSeek V4.1 Pro 状态" --file notes.md
        → core = 0.70KB / 15KB

    本意是追加一段 3 行笔记，结果把 15331B 的 core.md 整个换成 716B，
    **一份每轮全量注入的核心记忆当场没了**；而命令**退出码 0**、回执只有一行
    正常的尺寸 —— 破坏性操作既没有防呆，失败也不报警。这与本项目在
    `load()`/哨兵值上反复踩的坑是同一个形状。

    现在：新内容 < 旧内容的 `_CORE_SHRINK_RATIO`（一半）时**拒绝写入**，
    core.md 一个字节都不动，并给出「要整体重写请加 --force」的出口。
    压缩 core 时若确实要大幅删减，加 `--force` 明确确认即可。

    这里**不拦「写了个更大的进去」**：那唯一合法的解释是模型主动扩充内容，
    拦下来等于丢记忆（超限由 `_core_size_msg` 明确回执，交给下一轮压缩）。
    """
    # ⚠️ 「旧内容」必须在 `_ensure_files()` **之前**取：它会把缺失的 core.md
    #    建成一份 136B 的模板，之后再去量就量到了模板 —— 于是「第一次建库、
    #    写入一份很小的 core」这种正常操作也会被闸门拦住（假阳性）。
    #    闸门要防的是「**已存在的**核心记忆被抹掉」，不是「文件还没建」。
    existed = C.CORE.exists()
    _ensure_files()
    p = Path(src)
    if not p.exists():
        raise SystemExit(f"core source not found: {p}")
    body = p.read_text(encoding="utf-8", errors="replace")
    # ⚠️ 空源文件单独判：`--file empty.md` 会把核心记忆清空。空与「内容很少」
    #    是两件事 —— 前者是「一份记忆都没了」，不归缩水比例管。
    #    与 append_core 的空正文闸门对齐（空值不能从任何一侧溜进去）。
    if not body.strip():
        raise WriteError(
            f"core 整体替换的源文件为空：拒绝写入（{p} 无正文，替换后会清空核心记忆）"
        )
    old = C.core_bytes() if existed else -1
    new = _disk_bytes(body)
    if not force and old > 0 and new < old * _CORE_SHRINK_RATIO:
        raise WriteError(
            f"core 整体替换会让核心记忆从 {old}B 缩到 {new}B"
            f"（{new / old:.1%} < 阈值 {_CORE_SHRINK_RATIO:.0%}）→ 已拒绝，core.md 未改动\n"
            f"  本次调用是 `--file`（整体覆盖），不是追加。\n"
            f"  想追加一段：改用位置参数\n"
            f"      py nb_write.py core \"正文\" --title 段名\n"
            f"  确实要整篇重写（云端模型压缩）：加 --force 明确确认"
        )
    C.CORE.write_text(body, encoding="utf-8")
    return f"{_core_size_msg('replace', C.core_bytes())}（{old}B → {C.core_bytes()}B）"


def append_ws(area: str, text: str, title: str | None = None, at: str | None = None) -> str:
    """向工作区册追加条目。area 即工作区名（通常为项目目录名）。"""
    _ensure_files()
    safe = safe_area(area)
    if not text.strip():
        raise WriteError(f"ws/{safe}.md 正文为空：拒绝写入")
    body = text.strip()
    p = C.WS_DIR / f"{safe}.md"
    if not p.exists():
        p.write_text(
            f"# 工作区 · {safe}\n_详载本区工作内容 | 云端模型维护 | 仅检索_\n",
            encoding="utf-8",
        )
    if title:
        p.write_text(p.read_text(encoding="utf-8").rstrip() + _section(f"{_stamp(at)} | {title}", body), encoding="utf-8")
    else:
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n## {_stamp(at)}\n{body}\n")
    return f"ws/{safe}.md = {C.kb(p)}KB"


def reindex() -> str:
    """重建索引。成功返回一行摘要；失败抛 IndexRebuildFailed。

    ⚠️ 旧实现 `return (r.stdout or r.stderr or "").strip().split("\\n")[-1]`
    有双重坏味道，实测（2026-10-05 审查）：
      1. 成功与崩溃**共用同一个返回通道**：nb_index.py 崩了就把 stderr 最后一行
         当成正常摘要返回，调用方/用户看到一行文本，以为索引建好了 —— 而索引
         其实是旧的，检索静默漏掉刚写入的内容。
      2. 从不检查 `r.returncode`，`or` 短路只看字符串是否为空。
      3. 子进程退出码非 0 但 stdout 有半行输出时，连 stderr 都不会被读到；
         反过来退出码 0 而 stdout 为空时，会把 stderr 的空串当成功返回。
    这与 nb_index.load() 里那条教训是同一个病：**哨兵值绝不能和「无值」共用
    同一个返回通道**（date_of 的 '未标日期' 曾让 96 条记忆只剩 1 条有日期）。
    现在失败走异常，成功走正常返回值，两者不可能混淆。
    """
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "nb_index.py")],
        capture_output=True, text=True, timeout=120,
    )
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    if r.returncode != 0:
        # 带上退出码和 stderr：崩在哪一行比「重建失败」四个字有用得多
        detail = err.split("\n")[-1] if err else "(stderr 为空)"
        raise IndexRebuildFailed(f"nb_index.py 退出码 {r.returncode}: {detail}")
    if not out:
        # 退出码 0 却没有输出 = 约定被破坏，同样是可区分的失败，不许当成功
        raise IndexRebuildFailed("nb_index.py 退出码 0 但无输出（约定被破坏）")
    return out.split("\n")[-1]


def _reindex_or_die() -> None:
    """写命令收尾用：索引建不出来就明确喊出来，不静默当成功。"""
    try:
        print(reindex())
    except IndexRebuildFailed as e:
        # 内容已经写进去了。说清楚「写成功、索引没更新」，别让调用方以为整条命令失败
        # 而重写一遍 —— 重写会在记忆流里留下两条重复记录。
        print(f"⚠️ 内容已写入，但索引重建失败：{e}")
        print("   检索暂时看不到这次写入；修好后跑 `py nb_write.py reindex` 即可，不要重写内容。")
        sys.exit(2)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Notebook 三层记忆写入")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stream", help="追加记忆流 (本地模型)")
    s.add_argument("text")
    s.add_argument("--at")
    s.add_argument("--tag", default="")
    s.add_argument("--session", default="")
    # `--check` 只跑闸门、**不写盘**：给外部写入方（如 DSH 的 snowfox-sync 钩子）
    # 一个「先问再写」的入口。
    # ⚠️ 存在的理由（实测踩到）：那个钩子原本直接 `appendFileSync` 写 stream.md，
    # **完全绕过本模块的闸门**，于是把自己的对话残片当记忆写了进去——
    # 实测进了三条英文残句（"Now let me delegate the independent review to
    # sol6.1 while I run the full regression:"）。真实记录被这种垃圾挤占命中名额，
    # 而闸门还以为自己在生效（它拦的是同一个文件，只是没人从它这儿过）。
    # 闸门只有一条路能绕过去，就不算闸门。给外部写入方留一个 check 入口，
    # 让「绕过」没有借口。
    s.add_argument("--check", action="store_true",
                   help="只校验内容是否可写入（不落盘），退出码 0=可写 1=会被拒")

    c = sub.add_parser("core", help="核心记忆 (云端模型)")
    c.add_argument("text", nargs="?")
    c.add_argument("--title", default="追加")
    # --append 是「显式声明这是追加」的写法，行为与不传 --file 时的默认追加完全一致。
    # 旧版定义了却从不读 a.append，只在文档里宣传 —— 说谎的参数比没有更糟：
    # 用户以为 --append 覆盖了别的默认行为，实际什么都没发生（2026-10-05 审查实测）。
    c.add_argument("--append", action="store_true",
                   help="追加一段（默认行为即追加，此标志用于显式声明）")
    c.add_argument("--file",
                   help="整体覆盖 core.md（⚠️ 不是追加；缩水过半会被拒，需 --force）")
    # ⚠️ --force 存在的唯一理由：缩水闸门必须留一个**明确**的出口。
    # 没有出口的闸门会被绕过（历史上外部写入方就是直接 appendFileSync 绕开
    # stream 闸门的），有出口才能把「我真的要整篇替换」这个意图落到命令行上。
    c.add_argument("--force", action="store_true",
                   help="确认整体覆盖（新内容不足旧内容一半时必需）")

    w = sub.add_parser("ws", help="工作区册 (云端模型)")
    w.add_argument("area")
    w.add_argument("text")
    w.add_argument("--title")
    w.add_argument("--at")

    sub.add_parser("list", help="三层现状")
    sub.add_parser("reindex", help="重建索引")

    a = ap.parse_args()

    # 参数非法/内容为空 → 退出码 1，且**一个字节都没写**（校验都在写盘之前）。
    # 这里统一兜住 WriteError，子命令内部不必各自 try。
    try:
        if a.cmd == "stream":
            if a.check:
                # 只问闸门，不落盘：外部写入方据此决定写不写。
                # 退出码即结论，stdout 给人看原因。
                flat = " ".join(a.text.split())
                if not flat:
                    print("❌ 拒绝写入：正文为空")
                    sys.exit(1)
                reason = _stream_junk_reason(flat)
                if reason:
                    print(f"❌ 拒绝写入：{reason}")
                    sys.exit(1)
                print("✓ 可写入")
                return
            print(append_stream(a.text, a.at, a.tag, a.session))
            _reindex_or_die()
        elif a.cmd == "core":
            if a.file and a.append:
                # 不替用户猜意图：覆盖和追加是相反的操作，同时给出就报错
                raise WriteError("--file（整体覆盖）与 --append（追加）互斥，二选一")
            if a.file:
                print(replace_core(a.file, force=a.force))
            elif a.text is not None:
                # 显式给了位置参数（哪怕是个空串）就走追加，空内容由 append_core 拒绝。
                # ⚠️ `core ""` 与 `core`（不带参数）必须走两条路：前者是「传了个空值」→ 报错；
                # 后者是「只想看看当前核心记忆」→ 打印。
                # 旧代码用 `elif a.text:` 把两者混在一起：当 TEXT 为空时（脚本里
                # `py nb_write.py core "$TEXT"` 很常见）会打印一整页 core.md 并返回退出码 0，
                # 调用方以为写成功了。同 `--at` 那条：**不传 ≠ 传空串**。
                print(append_core(a.text, a.title))
            elif a.append:
                raise WriteError("--append 没有正文：--append 只是声明，正文仍要作为位置参数给出")
            else:
                print(C.CORE.read_text(encoding="utf-8", errors="replace"))
                return
        elif a.cmd == "ws":
            print(append_ws(a.area, a.text, a.title, a.at))
            _reindex_or_die()
        elif a.cmd == "list":
            print(C.summary())
        elif a.cmd == "reindex":
            # 纯 reindex 命令：失败就是这条命令失败，退出码 1 即可，没有「内容已写入」的歧义
            try:
                print(reindex())
            except IndexRebuildFailed as e:
                print(f"⚠️ 索引重建失败：{e}")
                sys.exit(1)
    except WriteError as e:
        print(f"❌ 拒绝写入：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
