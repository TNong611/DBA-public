"""精确验证：变了但变短 → 不采纳，但必须记进 relax_notes。"""
import sys
sys.path.insert(0, r'D:\AI\DBA\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import nb_search as S

orig = S.search

def trial(fn):
    calls = {'n': 0}
    def fake(query, top_k=S.C.RETRIEVAL_TOP_K, scope=None, max_bytes=None):
        calls['n'] += 1
        r = orig(query, top_k=top_k, scope=scope, max_bytes=max_bytes)
        return fn(dict(r)) if calls['n'] >= 2 else r
    S.search = fake
    try:
        return S.search_ladder('量子纠缠光谱仪', top_k=3, scope=None, max_rounds=2)
    finally:
        S.search = orig

print('=== A. 变了但变短 → 不采纳，但记 relax_notes ===')
def shorter(r):
    r['hits'] = [dict(h, id='DIFF%s' % i, text='不同内容') for i, h in enumerate(r['hits'][:1])]
    return r
r = trial(shorter)
adopted = any(str(h['id']).startswith('DIFF') for h in r['hits'])
print('  relaxed     =', r['relaxed'])
print('  relax_notes =', r.get('relax_notes'))
print('  采纳了吗    =', adopted)
print('  note        =', r.get('note', '')[:90])
ok = (not adopted) and bool(r.get('relax_notes'))
print('  %s 未采纳 + 如实记录了「试过但更少」' % ('✓' if ok else '✗'))

print()
print('=== B. 变了且更多 → 采纳并记 relaxed ===')
def more(r):
    r['hits'] = [dict(h, id='MORE%s' % i) for i, h in enumerate(r['hits'])] + \
                [dict(r['hits'][0], id='EXTRA')] if r['hits'] else r['hits']
    return r
r = trial(more)
print('  relaxed     =', r['relaxed'])
print('  relax_notes =', r.get('relax_notes'))
print('  %s 采纳并记录' % ('✓' if r['relaxed'] and not r.get('relax_notes') else '✗'))

print()
print('=== C. 条数相同但换了 → 采纳并记录（上一轮修的）===')
def same_diff(r):
    r['hits'] = [dict(h, id='SAME%s' % i) for i, h in enumerate(r['hits'])]
    return r
r = trial(same_diff)
print('  relaxed     =', r['relaxed'])
print('  %s 采纳并记录' % ('✓' if r['relaxed'] else '✗'))
