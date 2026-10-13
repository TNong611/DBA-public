"""复验 sol6.1 反证指出的「修复引入的新问题」。"""
import sys, json, tempfile, shutil
sys.path.insert(0, r'D:\AI\DBA\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import nb_config as C, nb_index as I, nb_search as S, nb_write as W
from pathlib import Path

print('══ A1. 中文计划句被漏过（完成 被当成完成标记）══')
for t in ['接下来完成索引', '现在我们来完成部署', '我们先计划完成索引',
          '1. 已完成机械建模', '我们先完成了索引修复并通过测试']:
    r = W._stream_junk_reason(' '.join(t.split()))
    should_reject = t.startswith(('接下来', '现在我们来', '我们先计划'))
    verdict = '✓' if (bool(r) == should_reject) else '✗ 错'
    print('  %s %-22s %s' % (verdict, t, (r[:38] if r else '放行')))

print()
print('══ A2. 纯空白/标点必须被拒（信息量折算 = 0 点）══')
for t in ['        ', '........', '!!!!!!!!', '。', ',,,']:
    iw = W._info_weight(t)
    r = W._stream_junk_reason(t)
    print('  %s %-12r 折算=%-5g %s' % ('✓ 拒' if r else '✗ 漏过', t, iw, (r[:30] if r else '放行')))

print()
print('══ A2b. 短的真实记录必须放行（短记录豁免）══')
# ⚠️ 这几条曾被我误列入 A2 的「该拒」名单，导致测试报假失败。
# 它们折算只有 6 点（< 8），但都是**真实记录**，绝不能被拒 ——
# 记忆流误杀是静默丢失，比漏过垃圾严重得多。
# 判据 = 短（<=SHORT_OK 阈值）**且**含完成语标记，两个条件同时成立才豁免。
for t in ['已修复', '已提交', '跑通了', '索引重建完成', '提交并推送']:
    r = W._stream_junk_reason(t)
    print('  %s %-14s 折算=%-5g %s' % ('✗ 误杀' if r else '✓ 放行', t, W._info_weight(t), (r[:34] if r else '')))

print()
print('══ 索引：缺 dated 键时 newest/neutral 全 None 仍可加载 ══')
saved = {k: getattr(C, k) for k in ('ROOT','MEMORY','INDEX')}
tmp = Path(tempfile.mkdtemp())
try:
    (tmp/'memory').mkdir(parents=True)
    C.ROOT = tmp; C.MEMORY = tmp/'memory'; C.INDEX = C.MEMORY/'.index'/'inverted.json'
    C.INDEX.parent.mkdir(parents=True)
    base = {'version':1,'docs':[{'date':'2026-10-06'}],'postings':{'a':[0]},'tf':{},'df':{},
            'len':{},'avg_len':1.0,
            'corpus_dates':{'dated':1,'newest':'2026-10-06','neutral_weight':0.74}}
    for cd, desc, want_corrupt in [
            ({'newest': None, 'neutral_weight': None}, '无 dated 键、全 None', True),
            ({}, '空', True),
            ({'dated': 0, 'newest': None, 'neutral_weight': None}, 'dated=0 全 None', True),
            # ⚠️ 这条**必须被接受**：它是健康的。健康判定是「整体」做的 ——
            # 有一份自洽的 dated/newest/neutral_weight，时间权重就能正常生效。
            # 旧测试把它列进「都该拒」的名单，于是每次都报一个 ✗ 假失败。
            # 测试里的期望值写错，比没有测试更糟：它会教人忽略 ✗。
            ({'dated': 1, 'newest': '2026-10-06', 'neutral_weight': 0.74}, '健康（应接受）', False)]:
        C.INDEX.write_text(json.dumps({**base, 'corpus_dates': cd}), encoding='utf-8')
        try:
            I.load()
            got_corrupt = False
        except I.IndexCorrupt as e:
            got_corrupt = True
            err = str(e)[:44]
        ok = (got_corrupt == want_corrupt)
        if got_corrupt:
            print('  %s %-24s 拒: %s' % ('✓' if ok else '✗ 误拒', desc, err))
        else:
            print('  %s %-24s 接受' % ('✓' if ok else '✗ 该拒不拒', desc))
finally:
    for k, v in saved.items(): setattr(C, k, v)
    shutil.rmtree(tmp, ignore_errors=True)

print()
print('══ stream.md 第 230 行到底是什么（判断是否真误杀）══')
lines = Path(r'D:\AI\DBA\memory\stream.md').read_text(encoding='utf-8').splitlines()
for i in (229, 230, 231):
    if i < len(lines):
        print('  %d: %r' % (i+1, lines[i][:96]))
