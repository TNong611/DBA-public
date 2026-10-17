"""门面字段契约回归测试：每种 mode 都必须返回同一套核心字段。

为什么需要这个测试：字段缺失是**静默**的 —— 调用方直到 KeyError 才发现，
而在此之前它可能已经拿别的 mode 跑通了（实测 mode="kw" 和 mode="both"
各漏过一次 expansion / narrowed）。契约必须由测试钉住，不能靠人记得。
"""
import sys
from pathlib import Path
# 仓库根由本文件位置推导：原为硬编码 D:\AI\DBA，克隆到别处（如 Termux）就跑不了。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'port'))
sys.stdout.reconfigure(encoding='utf-8')
import dba

# 各 mode 都必须有的核心字段
CORE = {"query", "hits", "text", "channels", "degraded", "bytes", "note",
        "expansion", "narrowed"}
MODES = ["kw", "vec", "both", "ladder", "auto"]
QUERIES = ["3055", "时间权重", "", "zzz 不存在的词"]

fails = 0
checks = 0
for q in QUERIES:
    for m in MODES:
        checks += 1
        try:
            r = dba.recall(q, mode=m)
        except Exception as e:
            print(f"  ✗ recall({q!r}, mode={m!r}) 抛异常: {type(e).__name__}: {e}")
            fails += 1
            continue
        missing = CORE - set(r.keys())
        if missing:
            print(f"  ✗ recall({q!r}, mode={m!r}) 缺字段: {sorted(missing)}")
            fails += 1
        if not isinstance(r.get("expansion"), dict):
            print(f"  ✗ recall({q!r}, mode={m!r}) expansion 不是 dict: {r.get('expansion')!r}")
            fails += 1
        if not isinstance(r.get("narrowed"), bool):
            print(f"  ✗ recall({q!r}, mode={m!r}) narrowed 不是 bool: {r.get('narrowed')!r}")
            fails += 1
        if not isinstance(r.get("hits"), list):
            print(f"  ✗ recall({q!r}, mode={m!r}) hits 不是 list")
            fails += 1

print(f"  契约检查: {checks} 项，{fails} 项失败")

print()
print("关键不变量：闸门关着时，检索**完全不碰**扩展逻辑")
# ⚠️ 断言要精确，不能写成「结果完全相同」—— 那是错的：
# `bom ing` 比 `bom` **多了一个真实词** `ing`，即使闸门关着、没有任何扩展，
# BM25 的查询词集合也变了，排名本来就可能变。把「加了词导致排名变化」当成
# 「扩展污染」是误判 —— 会让我去改一个本来正确的闸门。
# 真正该钉死的是：**闸门关着时 expansion 必须为空**。
# 空 expansion 就等价于「扩展词一个都没进查询」，污染在结构上不可能发生。
for q in ["bom ing", "bom con", "索引 重建 dummy", "时间权重 xyz"]:
    r = dba.recall(q, mode="kw")
    empty = not r.get("expansion")
    print(f"  {'✓' if empty else '✗'} {q!r:20} expansion={r.get('expansion')} （闸门应为关）")
    if not empty:
        fails += 1

print()
print("对照：闸门开着时确实扩展了（用合成索引，不依赖当前语料）")
import copy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_index as I
_idx = I.load()
synth = copy.deepcopy(_idx)
for tok in ('2023', '3055'):
    synth['postings'].pop(tok, None)
    synth['df'].pop(tok, None)
for q in ('2023', '3055'):
    t = I.tokenize(q)
    gate = bool(I.missing_terms(t, synth)) and not I.exact_candidates(t, synth)
    ex, mp = I.expand_terms(t, synth)
    ok = gate and bool(mp.get(q))
    print(f"  {'✓' if ok else '✗'} {q!r:8} 闸门开={gate} 扩展={mp.get(q)}")
    if not ok:
        fails += 1

print()
print(f"总失败 = {fails}")
sys.exit(1 if fails else 0)
