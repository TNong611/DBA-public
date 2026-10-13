# integrations/ — 各客户端的适配器

这里的文件是**已在实际运行的**集成代码的仓库副本。它们的作用是：换一台机器、
换一个客户端时，不必从零写接线。

> ⚠️ **这里的是副本，不是运行位置。** 真正被客户端加载的是下面「安装到」那一列。
> 改代码时**两边都要改**，否则仓库里的副本会与运行中的版本漂移 —— 而漂移的
> 集成代码比没有更危险：它看起来是对的。

| 客户端 | 仓库副本 | 实际安装位置 |
|---|---|---|
| DSH（DeepSeek Harness） | `dsh/snowfox-inject.mjs`<br>`dsh/snowfox-sync.mjs` | `~/.dsh/profiles/web/` |
| DSH 插件注册 | `dsh/cordis.patch.yml` | `~/.dsh/profiles/web/cordis.patch.yml` |
| DSH 计划任务入口 | `dsh/memory-daily.ps1` | `~/.dsh/scripts/memory-daily.ps1` |
| Hermes | `hermes/snowfox-memory/__init__.py`<br>`hermes/snowfox-memory/plugin.yaml` | `~/AppData/Local/hermes/plugins/snowfox-memory/` |

一致性自查（两边哈希应相同）：
```powershell
Get-FileHash ~\.dsh\profiles\web\snowfox-inject.mjs, D:\AI\DBA\integrations\dsh\snowfox-inject.mjs
```

---

## DSH 接入

DSH 侧分两个插件，职责不同，**两个都要装**：

### 1. `snowfox-sync.mjs` —— 写入 + 每轮组装

- 监听会话事件，每个 turn 结束把助手侧动作压成一行写进 `memory/stream.md`
  （`- YYYY-MM-DD HH:MM:SS | tag=DSH | …`）。
- 用户输入到达时**同步**调用 `nb_bridge.py` 重建组装产物。
  **同步而非异步**：`snowfox-inject` 紧接着就要读那两个文件，异步会有竞态
  （读到的可能是上一轮的命中）。实测组装约 100ms，相对 LLM 往返可忽略。

写入过滤（都是踩过的坑）：只记人类输入、丢弃空话回复、把 `{{` 打断为 `{\u200b{`
（角色卡文本曾带 `{{char}}` 进记忆 → 被插值器当未注册变量 → **整轮抛错**）、
剥掉代码围栏保证一行一条。

### 2. `snowfox-inject.mjs` —— 注入系统提示

把组装产物拆成 **两个 section** 注入，这是前缀缓存的要求：

| section | 文件 | order | 为什么在这个位置 |
|---|---|---|---|
| `snowfox:memory` | `_core_context.md` | **-50** | 核心记忆几天才变 → 排前面，稳定内容在前才能被缓存 |
| `snowfox:retrieval` | `_hits_context.md` | **200** | 每轮都变 → 排最末尾（工具定义之后），变化不破坏前缀 |

合成一段的后果（实测）：检索一命中，system prompt 从该段起整段变化，
其后所有段落缓存全失效 —— 命中率 92.7% → **0%**。

**路径解析**：两个插件都从 `DBA_HOME` 环境变量取仓库根，默认 `D:\AI\DBA`。
不硬编码路径，仓库可以整体搬走。

### 安装

1. 把 `dsh/*.mjs` 复制到 `~/.dsh/profiles/web/`
2. 把 `dsh/cordis.patch.yml` 复制到 `~/.dsh/profiles/web/cordis.patch.yml`
   （若已有该文件，把 `insert:` 那两条合并进去，**别覆盖**你原有的条目）
3. 重启 DSH

> ⚠️ **`snowfox-assemble` 必须保持停用**。它在 DSH 内复刻旧的五级预算组装，
> 读 `fixed/long_term/summary/recent.md` 并覆写 `_assembled_context.md`。
> 那几个文件已迁为 `*.legacy`，启用它会**用过期内容覆盖新产物**，
> 注入的记忆变成旧的，而且不报错。组装权归 `nb_assemble.py`。

---

## Hermes 接入

`hermes/snowfox-memory/` 是一个 standalone 插件，挂三个钩子：

| 钩子 | 做什么 |
|---|---|
| `pre_llm_call` | 以本轮用户输入为 query 检索记忆 → 组装 → `override_messages` |
| `post_llm_call` | 把助手侧回填进 pending 区，等待配对 |
| `on_session_end` | 把 pending 提炼成记忆流条目，并重建索引 |

**安装**：整个目录复制到 `~/AppData/Local/hermes/plugins/snowfox-memory/`，
然后重启 Hermes 网关（`hermes gateway restart`）。

> ⚠️ **必须重启网关**。Hermes 在启动时把插件代码读进内存，只改文件不重启的话
> 跑的还是旧代码 —— 本项目实际踩过：迁移后旧插件仍在写 `recent.md`（那是已废弃
> 的文件），而日志看起来一切正常。

> ⚠️ 插件内部**不再自己拼路径**，而是 `sys.path` 挂上 `DBA_HOME/scripts` 后
> 导入 `nb_config`，由它决定数据在哪。所以记忆和插件可以分别放在不同盘。

---

## 计划任务

`dsh/memory-daily.ps1` 是每日维护入口（计划任务 `DSH_Memory_Daily`，每日 00:05）。

它**现在只做转发**：把工作目录切到 DBA 仓库，再跑那边的脚本。
保留在 `~/.dsh/scripts/` 只是因为计划任务指向这个路径。

步骤：
1. `mem_tasklog.py compress` —— 昨天的任务日志 → `memory/_pending_core.md`（素材，不是 core）
2. `mem_tasklog.py init` —— 清理昨日 workspace，建今日目录
3. `memory_maintenance.py` —— 组装注入产物 + 向量索引按需重建

**旧的 STEP 3（`mem_compress.py`）已删除**：DBA 的 stream/ws 不限长度，
没有 L1→L2 分层压缩这一步，该脚本在新架构下只会打印 skip。留一个必然
失败/空转的步骤在日志里，会训练人忽略日志里的 FAIL。

注册（需管理员）：
```powershell
schtasks /Create /TN DSH_Memory_Daily /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"$env:USERPROFILE\.dsh\scripts\memory-daily.ps1\"" /SC DAILY /ST 00:05 /F
```

---

## 接入别的东西

不想碰上面这些客户端代码的话，直接用门面 —— 它不依赖任何客户端：

```python
import sys; sys.path.insert(0, r"D:\AI\DBA\port")
import dba
print(dba.recall("关键词")["text"])
```

见 [../docs/接入指南.md](../docs/接入指南.md)。

_DBA · 双脑架构 · 2026-10-06_
