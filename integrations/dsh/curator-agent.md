# 记忆整理子代理（Curator Agent）

> 给「专职整理记忆」的子代理用的**固定任务书**。任何会话都可以按这份规格派一个子代理，
> 也可以直接对着它干活。**目标只有一个：让 `core.md` 长期保持在 15KB 上限内，且一条硬事实都不丢。**

## 何时派它

| 触发条件 | 动作 |
|---|---|
| `core.md` 余量 < 200 字节 | 派子代理，按本文件流程压缩 |
| `memory/_pending_core.md` 有内容 | 一并并入（这是每日任务暂存的待办条目） |
| 用户说「整理记忆 / 记忆满了 / 压一下 core」 | 同上 |

## 铁律（违反即失败）

1. **只做提炼，不做搬家**：把细节压成一句结论并在句末标「细节见 `ws/<区>.md`」；
   被删掉的细节如果还没进 `ws`，**先写进对应 `ws/<区>.md`**（`nb_write.py ws <区>`），再从 core 里删。
2. **硬事实零丢失**：路径、端口、`DSH_*`/`nb_*` 任务名、反引号标识符、容量阈值 ——
   一个字都不能少（可以换位置、合并句式）。
3. **容量按原始字节算**：`len(text.encode('utf-8')) <= 15360`。
   不要用 `kb()` 那种四舍五入到 0.1KB 的口径（15361 会显示成 15.0KB 而 `15.0 > 15` 为假 → 刚超限会静默通过）。
4. **不许编造**：不得新增原文没有的事实；不确定就保留原句。
5. 改完**必须**跑校验器，不许手工判断「应该没丢」：

```
py -3.14 D:\AI\DBA\tools\curate_core.py --engine ollama      # 本地 qwen3:8b 出提案
py -3.14 D:\AI\DBA\tools\curate_core.py --engine deepseek    # 云端出提案（本地模型压不动时）
```

## 标准流程

```
① 读现状
   py -3.14 D:\AI\DBA\tools\curate_core.py --self-test        # 先确认校验器本身可信
   看 memory/_curate_report.md（上一轮的大小、缺失项）

② 出提案（两个引擎选一个；本地免费、云端更强）
   py -3.14 D:\AI\DBA\tools\curate_core.py --engine ollama
   → 产物 memory/_pending_core.curated.md + memory/_curate_report.md

③ 读报告，按失败原因决定
   - 校验通过 → 人工/云端扫一眼 diff，确认没丢语义 → --apply
   - 校验失败（缺 token / 超上限 / 分区丢失）→ **不要强推**：
     换成 deepseek 引擎重跑，或手工改 _pending_core.curated.md 补齐缺失项再 --apply

④ 落地
   py -3.14 D:\AI\DBA\tools\curate_core.py --engine ollama --apply
   （--apply 会自动备份 core.md.bak-<时间戳> 并清空 _pending_core.md）

⑤ 收尾
   py -3.14 D:\AI\DBA\scripts\nb_rebuild.py     # 记忆改了要重建索引
   git -C D:\AI\DBA add -A && git commit -m "memory: curate core.md"
```

## 禁止事项

- ❌ **不要手工把 `_pending_core.curated.md` 搬成 `core.md`** —— 必须走 `--apply`。
  手工 `shutil.move` 会**绕过校验闸门**，把没验过的提案直接变成记忆。
  （2026-10-13 就犯过一次：提案缺一个代码片段，靠 `core.md.bak-*` 才救回来。）
- ❌ **不要**用「删一段看着不重要的内容」来凑字节 —— 先确认它已在 `ws/` 里，没有就先补写 `ws/`。
- ❌ **不要**把 `core.md` 里的事实挪进 `_pending_core.md` 或临时文件当"暂存"（那是静默丢失）。
- ❌ **不要**把个人信息（学号/真名/公司名/key）写进 `scripts/`、`docs/`、`tools/` —— 会被 `DBA-public` 镜像公开。
- ❌ **不要**跳过 `--self-test`：校验器自己坏掉时，"校验通过" 毫无意义。

## 相关文件

| 文件 | 作用 |
|---|---|
| `memory/core.md` | 被整理的对象（上限 15360 字节） |
| `memory/_pending_core.md` | 每日任务暂存的待并入条目（并入后清空） |
| `memory/_pending_core.curated.md` | 整理器产出的提案 |
| `memory/_curate_report.md` | 校验明细 / 失败原因 / 并入记录 |
| `tools/curate_core.py` | 整理器本体（模型提议 + 确定性校验 + 可选落盘） |
| `memory/core.md.bak-*` | 每次 `--apply` 前的自动备份 |

---

_规格版本：2026-10-13（由雪狐建立；配合 `tools/curate_core.py`）_
