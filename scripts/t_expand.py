"""子串扩展的行为与不变量测试。

⚠️ 为什么要用**合成索引**而不是直接用真实索引：
真实索引的语料会**跟着记忆一起变**。实测踩过：把「`2023` 查不到」这件事写进记忆后，
语料里就真的出现了 `2023` 这个 token，于是精确匹配成功、闸门关闭、扩展不再触发 ——
测试开始报假失败。测试若依赖「语料里恰好没有某个词」，就会随记忆内容漂移，
变成不可靠的噪音源（而报假失败的测试会教人忽略 ✗）。
所以这里自己造索引：**明确**删掉目标 token，模拟「语料里确实没有该片段」。
这样测的是**代码逻辑**，不是当前记忆的内容。
"""
import sys, copy, time
from pathlib import Path
# 仓库根由本文件位置推导（原为硬编码 D:\AI\DBA，克隆到别处就跑不了）
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8')
import nb_index as I

fails = 0
real_index = I.load()
if not real_index:
    print('✗ 真实索引不可用，无法构造测试基线')
    sys.exit(1)

# ── 构造一份「没有 2023 / 3055 这两个片段」的索引 ──
synth = copy.deepcopy(real_index)
dropped = []
for tok in ('2023', '3055'):
    if tok in synth['postings']:
        dropped.append(tok)
        del synth['postings'][tok]
        synth['df'].pop(tok, None)
print('══ 合成索引：已删除 token %s（模拟语料里不存在这些片段）══' % dropped)
print('  docs=%d terms=%d' % (len(synth['docs']), len(synth['postings'])))
print()

print('══ A. 闸门：精确有候选时，绝不扩展 ══')
terms = I.tokenize('202301233055')
missing = I.missing_terms(terms, synth)
cands = I.exact_candidates(terms, synth)
gate_open = bool(missing) and not cands
ok = not gate_open
if not ok: fails += 1
print('  %s 精确命中「202301233055」: missing=%s 候选=%d → 闸门%s' % (
    '✓' if ok else '✗', missing, len(cands), '开（不该）' if gate_open else '关（正确）'))
print()

print('══ B. 精确查不到时，闸门打开并扩展 ══')
for q in ('2023', '3055'):
    terms = I.tokenize(q)
    missing = I.missing_terms(terms, synth)
    cands = I.exact_candidates(terms, synth)
    gate_open = bool(missing) and not cands
    ex, mp = I.expand_terms(terms, synth)
    got = mp.get(q, [])
    ok = gate_open and bool(got)
    if not ok: fails += 1
    print('  %s %-6s missing=%s 候选=%d 闸门=%s 扩展=%s' % (
        '✓' if ok else '✗', q, missing, len(cands),
        '开' if gate_open else '关', got[:2]))
print()

print('══ C. 扩展确实能召回 ══')
for q in ('2023', '3055'):
    ex, mp = I.expand_terms(I.tokenize(q), synth)
    scored = I.score(ex, synth)
    ok = bool(scored)
    if not ok: fails += 1
    top = synth['docs'][scored[0][0]]['id'] if scored else '—'
    print('  %s %-6s 扩展后候选文档=%d top=%s' % ('✓' if ok else '✗', q, len(scored), top))
print()

print('══ D. 确定性：等长候选不受插入顺序影响 ══')
a, _ = I.expand_terms(['con'], synth)
rev = copy.deepcopy(synth)
rev['postings'] = {k: synth['postings'][k] for k in reversed(list(synth['postings']))}
b, _ = I.expand_terms(['con'], rev)
ok = (a == b)
if not ok: fails += 1
print('  %s 反转插入顺序后候选一致 = %s' % ('✓' if ok else '✗', ok))
print()

print('══ E. 性能上界 ══')
for label, toks in [('300 个不同未命中词', ['zzz%d' % i for i in range(300)]),
                    ('1000 个重复未命中词', ['zzz'] * 1000)]:
    t0 = time.perf_counter(); I.expand_terms(toks, synth); t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    ok = ms < 50
    if not ok: fails += 1
    print('  %s %-22s %.1f ms（应 <50ms）' % ('✓' if ok else '✗', label, ms))
print()

print('══ F. 正常中文查询不触发扩展 ══')
CH = ['时间权重', '索引 重建', '记忆 三级 检索', '闸门 误杀', '手机 音量',
      '求职 简历', '宣讲会', '代理 地址', '向量 GPU', '脱敏 卡号']
bad = []
for q in CH:
    _, mp = I.expand_terms(I.tokenize(q), synth)
    if mp:
        bad.append((q, mp))
for q, mp in bad:
    print('  ✗ %-14s 意外扩展 %s' % (q, mp))
ok = not bad
if not ok: fails += 1
print('  %s %d 条中文查询，意外扩展 %d 条' % ('✓' if ok else '✗', len(CH), len(bad)))
print()

print('══ G. 异常输入不崩 ══')
for label, t in [('空列表', []), ('空串', ['']), ('单字符', ['a']),
                 ('两字符', ['ab']), ('纯空白', ['   ']),
                 ('重复词', ['zzz', 'zzz', 'zzz']), ('超长', ['x' * 1000])]:
    try:
        ex, mp = I.expand_terms(t, synth)
        print('  ✓ %-8s → 词数 %d，扩展 %d 个' % (label, len(ex), len(mp)))
    except Exception as e:
        fails += 1
        print('  ✗ %-8s 抛 %s: %s' % (label, type(e).__name__, e))
print()
print('总失败 = %d' % fails)
sys.exit(1 if fails else 0)
