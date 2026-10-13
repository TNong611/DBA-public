"""Curated event -> four-line memory text pairs for the RWKV state decoder pilot.

This module is DATA ONLY. It owns the cases, the serialization used for both
RWKV observation and decoder targets, and the quality gates that keep the
labels honest (unverified work must stay explicitly unverified, a request must
not be recorded as a completed result).

Two disjoint groups:
  TRAIN   : the pilot trains on these states.
  HELDOUT : deliberately distinct topics, never observed by the frozen RWKV
            backbone either, so a decoder that can reconstruct heldout memory
            text from a heldout recurrent state is evidence of state->text
            transfer rather than of memorising the backbone's own context.

Cases are loaded from a JSONL file (`load_events(cases_path, template_path)`),
so the pilot is not limited to the examples embedded below.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Scope of the pilot, written into every checkpoint and status file. It is a
# decoder-only experiment on top of a FROZEN backbone: not RWKV fine-tuning.
LABEL = "decoder-only pilot (frozen RWKV-7 recurrent state, no backbone LoRA)"
BACKBONE_TRAINING = False

SLOT_KEYS = ("topic", "constraint", "pending", "result")
SLOT_LABELS = {"topic": "主题", "constraint": "约束", "pending": "未完成", "result": "结果"}
LABEL_TO_SLOT = {label: key for key, label in SLOT_LABELS.items()}
SLOT_PREFIX = {"topic": "主题：", "constraint": "约束：", "pending": "未完成：", "result": "结果："}
EMPTY_SLOT = "无"
FORMAT = 1

BOS = "<|bos|>"
EOS = "<|eos|>"
# id 0 is the RWKV tokenizer's own end-of-text token and must never be emitted
# by the decoder as a visible label token.
RWKV_EOT_ID = 0

REQUIRED_CASE_FIELDS = {"id", "group", "user", "assistant", "unverified", "summary"}
SUMMARY_LABEL_ORDER = ("主题", "约束", "未完成", "结果")

# Completion words that must never appear inside 已确认 verification claims.
_COMPLETION_WORDS = (
    "已经完成", "已完成", "已完成", "已经解决", "已解决", "已经验证", "已验证",
    "已经通过", "已通过", "已经修复", "已修复", "已经上线", "已部署", "已经部署",
    "已经确认", "已确认成功", "修复成功", "验证成功", "调试完成", "已调试",
)
# Hedges that mark a statement as explicitly unverified / pending.
_UNVERIFIED_HEDGES = ("未验证", "尚未", "未能", "未完成", "未通过", "未修复", "未确认", "待")
# Words that must not be asserted as done when the source says they were not done.
_ACTION_WORDS = ("验证", "测试", "部署", "完成", "修复", "通过", "解决")

# Group names. Anything not in GROUPS is rejected by validation.
TRAIN = "train"
HELDOUT = "heldout"
GROUPS = (TRAIN, HELDOUT)

# The pilot's training cases. Each is one closed event (user + assistant) with
# a four-line labelled memory text as the decoder target.
TRAIN_CASES = [
    {
        "id": "T01", "group": TRAIN,
        "user": "电机控制器的闭环调试还没做完，先检查编码器极性，再跑低速空载测试。",
        "assistant": "收到。闭环调试尚未完成，我先核对编码器极性并准备低速空载测试，完成前不会汇报调试通过。",
        "unverified": True,
        "summary": "主题：电机控制器闭环调试\n约束：完成前不得汇报调试通过\n未完成：核对编码器极性并跑低速空载测试\n结果：尚未确认闭环调试结果",
    },
    {
        "id": "T02", "group": TRAIN,
        "user": "这个项目只能用GPU，CPU推理不接受。",
        "assistant": "明白。推理强制使用CUDA BF16，GPU不可用时直接失败，不回退CPU。",
        "unverified": False,
        "summary": "主题：推理硬件约束\n约束：只允许GPU，禁止CPU回退\n未完成：无\n结果：已确认GPU约束写入推理配置",
    },
    {
        "id": "T03", "group": TRAIN,
        "user": "索引恢复失败了，先修复重建流程，不要动原始事件。",
        "assistant": "我会保留原始事件，检查损坏索引并从正文重建；完成前不报告修复成功。",
        "unverified": True,
        "summary": "主题：索引恢复流程\n约束：原始事件不得修改\n未完成：修复重建流程\n结果：尚未修复索引恢复",
    },
    {
        "id": "T04", "group": TRAIN,
        "user": "索引重建昨晚跑完了，正文校验也通过了。",
        "assistant": "确认：重建流程已经跑完，重建结果与正文校验全部通过，未发现残留损坏。",
        "unverified": False,
        "summary": "主题：索引重建已完成\n约束：无\n未完成：无\n结果：已确认重建完成且正文校验通过",
    },
    {
        "id": "T05", "group": TRAIN,
        "user": "先把训练方案记下来，训练本身还没开始。",
        "assistant": "已记录方案。训练尚未开始，方案里的超参数都还没有实测数据。",
        "unverified": True,
        "summary": "主题：训练方案记录\n约束：方案不得写成已完成实测\n未完成：开始训练并验证方案\n结果：尚未开始训练",
    },
    {
        "id": "T06", "group": TRAIN,
        "user": "训练跑完了吗？显存峰值多少？",
        "assistant": "训练已完成，实测显存峰值2989MiB，速度约3.0秒每条。",
        "unverified": False,
        "summary": "主题：训练实测结果\n约束：无\n未完成：无\n结果：已确认训练完成，显存峰值2989MiB，约3.0秒每条",
    },
    {
        "id": "T07", "group": TRAIN,
        "user": "断电时保存checkpoint，重新插电后继续训练。",
        "assistant": "已实现：拔电时在step边界原子保存checkpoint并暂停，重新插电后自动续训。",
        "unverified": False,
        "summary": "主题：断电续训机制\n约束：只在交流供电且插电时允许训练\n未完成：无\n结果：已确认checkpoint原子保存与自动续训可用",
    },
    {
        "id": "T08", "group": TRAIN,
        "user": "断电续训实测过了吗？",
        "assistant": "还没有实测。机制写好了，但没有真的拔过电验证续训。",
        "unverified": True,
        "summary": "主题：断电续训未实测\n约束：未实测前不得写成可用\n未完成：拔电实测续训\n结果：尚未实测断电续训",
    },
    {
        "id": "T09", "group": TRAIN,
        "user": "那个检索路径有注入风险，先别上线。",
        "assistant": "已停在上线之前：检索路径存在注入风险，未经安全测试不会部署到生产。",
        "unverified": True,
        "summary": "主题：检索路径注入风险\n约束：安全测试通过前不得上线\n未完成：无\n结果：尚未确认检索路径可以上线",
    },
    {
        "id": "T10", "group": TRAIN,
        "user": "安全测试做了吗？",
        "assistant": "还没做。注入风险仍然存在，测试计划还没开始。",
        "unverified": True,
        "summary": "主题：检索注入风险未处置\n约束：不得把未测试的检索路径当作安全\n未完成：执行注入安全测试\n结果：尚未确认注入风险已消除",
    },
    {
        "id": "T11", "group": TRAIN,
        "user": "验证一下定时任务还活着没有。",
        "assistant": "查过了：每日记忆维护任务上次运行成功，返回码0。",
        "unverified": False,
        "summary": "主题：每日记忆维护任务\n约束：无\n未完成：无\n结果：已确认任务上次运行成功且返回码为0",
    },
    {
        "id": "T12", "group": TRAIN,
        "user": "定时任务报错了吗？",
        "assistant": "报了。任务返回码1，日志里是编码问题，还没定位到具体脚本。",
        "unverified": True,
        "summary": "主题：定时任务报错\n约束：无\n未完成：定位编码问题所在脚本\n结果：尚未定位报错原因",
    },
    {
        "id": "T13", "group": TRAIN,
        "user": "换行符要统一成LF，不要CRLF。",
        "assistant": "已统一：仓库属性设置为不做行尾转换，脚本字节保持一致。",
        "unverified": False,
        "summary": "主题：仓库行尾规范\n约束：文件行尾保持LF\n未完成：无\n结果：已确认行尾转换关闭",
    },
    {
        "id": "T14", "group": TRAIN,
        "user": "把旧的五级记忆文件都迁移掉。",
        "assistant": "旧文件已全部改名为legacy，不再读写。迁移动作做完了，但检索结果还没抽查。",
        "unverified": True,
        "summary": "主题：旧记忆文件迁移\n约束：旧文件不再读取与写入\n未完成：抽查迁移后的检索结果\n结果：尚未抽查检索结果是否正常",
    },
    {
        "id": "T15", "group": TRAIN,
        "user": "向量索引能从正文重建出来吗？",
        "assistant": "可以从正文重建。我实测重建成功，且与关键词索引互不依赖。",
        "unverified": False,
        "summary": "主题：向量索引重建\n约束：无\n未完成：无\n结果：已确认向量索引可从正文重建",
    },
    {
        "id": "T16", "group": TRAIN,
        "user": "GPU推理慢怎么办？",
        "assistant": "先测真实速度再谈优化：现在还没有测速数据，谈不上优化结论。",
        "unverified": True,
        "summary": "主题：推理速度优化\n约束：没有实测数据不得下优化结论\n未完成：先测真实推理速度\n结果：尚未取得测速数据",
    },
]

# Heldout: distinct topics, deliberately overlapping with TRAIN only in the
# slot semantics (what belongs in which slot), never in subject matter.
HELDOUT_CASES = [
    {
        "id": "H01", "group": HELDOUT,
        "user": "飞控固件烧写失败了，先查串口连接，不要动已经标定的参数。",
        "assistant": "会保留标定参数，先检查串口连接再重试烧写；完成前不报告烧写成功。",
        "unverified": True,
        "summary": "主题：飞控固件烧写\n约束：标定参数不得改动\n未完成：检查串口连接后重试烧写\n结果：尚未烧写成功",
    },
    {
        "id": "H02", "group": HELDOUT,
        "user": "电池组均衡测试做完了吗？",
        "assistant": "做完了：均衡测试跑完，单体温差2度，记录已归档。",
        "unverified": False,
        "summary": "主题：电池组均衡测试\n约束：无\n未完成：无\n结果：已确认均衡测试完成，单体温差2度",
    },
    {
        "id": "H03", "group": HELDOUT,
        "user": "相机标定还没做完，先把内参文件备份。",
        "assistant": "内参文件已备份；相机标定仍未完成，标定流程还没跑完。",
        "unverified": True,
        "summary": "主题：相机标定\n约束：内参文件保持备份\n未完成：完成相机标定流程\n结果：尚未确认相机标定完成",
    },
    {
        "id": "H04", "group": HELDOUT,
        "user": "这次翻译质量评估的结论是什么？",
        "assistant": "评估已确认：抽样500句，人工评分平均4.6，无严重误译。",
        "unverified": False,
        "summary": "主题：翻译质量评估\n约束：无\n未完成：无\n结果：已确认抽样500句平均4.6且无严重误译",
    },
    {
        "id": "H05", "group": HELDOUT,
        "user": "把日志脱敏规则写下来，别把规则当成已经生效。",
        "assistant": "规则已记录。脱敏尚未接入写入流程，还没有生效。",
        "unverified": True,
        "summary": "主题：日志脱敏规则\n约束：规则未接入不得当作生效\n未完成：把脱敏接入写入流程\n结果：尚未确认脱敏生效",
    },
    {
        "id": "H06", "group": HELDOUT,
        "user": "老显卡的驱动回滚完了吗？",
        "assistant": "回滚完成：驱动已回到上一个稳定版本，重启后画面正常。",
        "unverified": False,
        "summary": "主题：显卡驱动回滚\n约束：无\n未完成：无\n结果：已确认驱动回滚完成且重启后正常",
    },
]


def cases(group=None):
    both = list(TRAIN_CASES) + list(HELDOUT_CASES)
    if group is None:
        return [dict(case) for case in both]
    if group not in GROUPS:
        raise ValueError("unknown group: " + str(group))
    return [dict(case) for case in both if case["group"] == group]


def load_events(cases_path=None, template_path=None):
    """Read extra cases from JSONL. Missing files are simply no extra data."""
    path = Path(cases_path or os.environ.get("DBA_TRAIN_CASES", "cases.jsonl"))
    explicit_template = template_path or (os.environ.get("DBA_TRAIN_TEMPLATE") if not cases_path else None)
    template = Path(explicit_template) if explicit_template else None
    if cases_path and template_path:
        raise ValueError("choose cases_path or template_path, not both")
    if template is not None and template.is_file():
        payload = json.loads(template.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list):
            raise ValueError("case template must be a JSON array of cases")
        return [_normalized(case) for case in payload]
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    out = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(_normalized(json.loads(line)))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("cases line %d invalid: %s" % (number, exc)) from exc
    return out


def _normalized(raw):
    if not isinstance(raw, dict):
        raise ValueError("case must be an object")
    missing = REQUIRED_CASE_FIELDS - set(raw)
    if missing:
        raise ValueError("case missing fields: " + ", ".join(sorted(missing)))
    if raw["group"] not in GROUPS:
        raise ValueError("case group must be one of " + ", ".join(GROUPS))
    out = {k: raw[k] for k in ("id", "group", "user", "assistant", "summary")}
    out["unverified"] = bool(raw["unverified"])
    for key in ("id", "user", "assistant", "summary"):
        if not isinstance(out[key], str) or not out[key].strip():
            raise ValueError("case %s must be a nonempty string" % key)
    return out


def all_cases(cases_path=None, template_path=None):
    """Embedded cases first, then any JSONL/template extras. Validated."""
    items = validate(cases() + load_events(cases_path, template_path))
    # Keep the feature matrix layout aligned with train_count/heldout_count.
    return sorted(items, key=lambda case: (0 if case["group"] == TRAIN else 1, case["id"]))


def validate(items):
    """Reject label sets that would teach the decoder to lie or to copy input."""
    ids = set()
    summaries = set()
    for case in items:
        if case["id"] in ids:
            raise ValueError("duplicate case id: " + case["id"])
        ids.add(case["id"])
        summary = case["summary"]
        pairs = _slot_pairs(summary)
        labels = dict(pairs)
        if [name for name, _ in pairs] != list(SLOT_KEYS):
            raise ValueError("case %s must have exactly the four slots in order %s"
                             % (case["id"], "/".join(SLOT_KEYS)))
        for name, value in pairs:
            if not value:
                raise ValueError("case %s slot %s is empty" % (case["id"], name))
        if len(summary) > 200:
            raise ValueError("case %s summary is longer than 200 characters" % case["id"])
        if case["summary"] in summaries:
            raise ValueError("duplicate memory text across cases: " + case["id"])
        summaries.add(summary)
        result = labels["result"]
        source = case["user"] + "\n" + case["assistant"]
        if case["unverified"]:
            if not has_hedge(result):
                raise ValueError("case %s is marked unverified but its result reads as done: %s"
                                 % (case["id"], result))
        else:
            if not has_hedge(result):
                for action in _ACTION_WORDS:
                    if asserts_unperformed(source, action) and action in result:
                        raise ValueError("case %s records an explicitly undone action as done: %s"
                                         % (case["id"], result))
        for name, value in pairs:
            if name == "result" and not case["unverified"]:
                if not any(marker in value for marker in ("已确认", "已完成") + _COMPLETION_WORDS):
                    raise ValueError("case %s verified result states no confirmation: %s"
                                     % (case["id"], value))
        # The decoder must not be able to copy the answer out of its input.
        # 主题 is a noun phrase and legitimately shares wording with the event;
        # the three assertion slots must not be liftable verbatim.
        for name, value in pairs:
            if name == "topic":
                continue
            if value != EMPTY_SLOT and len(value) >= 8 and value in source:
                raise ValueError("case %s copies its source verbatim into %s: %s"
                                 % (case["id"], name, value))
        if len(labels["topic"]) > 14:
            raise ValueError("case %s topic is a sentence, not a noun phrase" % case["id"])
    return items


def has_hedge(text):
    """True when a statement is explicitly marked unverified / unfinished."""
    return any(hedge in text for hedge in _UNVERIFIED_HEDGES)


def asserts_unperformed(source, action):
    """True when the source says this action was not done / not started."""
    return any(marker + action in source
               for marker in ("尚未", "没有", "还没", "未开始", "未", "没"))


def _slot_pairs(summary):
    pairs = []
    for line in summary.split("\n"):
        if "：" not in line:
            raise ValueError("slot line has no label: " + line)
        label, value = line.split("：", 1)
        key = LABEL_TO_SLOT.get(label)
        if key is None:
            raise ValueError("unknown slot label: " + label)
        pairs.append((key, value.strip()))
    return pairs


def observation_text(case, kind="train"):
    """The exact byte stream the frozen backbone observes (train and eval agree)."""
    if kind not in ("train", "source"):
        raise ValueError("observation kind must be train or source")
    payload = {"user": case["user"], "assistant": case["assistant"]}
    return "<observed_turn>\n" + json.dumps(payload, ensure_ascii=False) + "\n</observed_turn>\n"


def source_text(case):
    """Ablation input: the raw event, not the memory text. No label leakage."""
    return observation_text(case, "source")


def memory_text(case):
    return case["summary"]


def token_counts(pipeline, items):
    """Token counts of everything that must fit the backbone context window."""
    counts = []
    for case in items:
        counts.append(len(pipeline.encode(observation_text(case))))
        counts.append(len(pipeline.encode(case["summary"])))
    return counts


def data_version(items):
    """Stable identity of the exact case set a checkpoint was trained on."""
    payload = json.dumps(
        [{k: c[k] for k in ("id", "group", "user", "assistant", "unverified", "summary")} for c in items],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_counts(items):
    train = sum(1 for c in items if c["group"] == TRAIN)
    return {"train": train, "heldout": len(items) - train, "total": len(items)}


def slot_match(generated, expected):
    """Slot-level generation metric. Empty slots count only when both are empty."""
    got = dict(_tolerant_slots(generated))
    want = dict(_tolerant_slots(expected))
    hits = {k: 0 for k in SLOT_KEYS}
    exact = 0
    for key in SLOT_KEYS:
        if key in got and key in want and got[key] == want[key]:
            hits[key] = 1
            exact += 1
    return {"per_slot": hits, "slots_exact": exact, "slots_total": len(SUMMARY_LABEL_ORDER),
            "all_slots_exact": exact == len(SUMMARY_LABEL_ORDER)}


def content_slot_match(generated, expected):
    """The same comparison restricted to slots that were NOT empty in the reference.

    `slot_match` gives an empty slot a hit whenever the model also writes `无`, so a
    decoder that only ever emits `无` already scores about 1.0/4 while knowing nothing.
    That floor is what the v3 run sat at. This variant cannot be earned that way: an
    empty or missing reference slot is simply not counted, so the denominator is the
    number of slots that actually carried content.
    """
    got = dict(_tolerant_slots(generated))
    want = dict(_tolerant_slots(expected))
    hits = {}
    for key in SLOT_KEYS:
        value = want.get(key)
        if value is None or value.strip() in ("", EMPTY_SLOT):
            continue
        hits[key] = 1 if got.get(key) == value else 0
    scored = len(hits)
    return {"per_slot": hits, "hits": sum(hits.values()), "scored": scored,
            "accuracy": (sum(hits.values()) / scored) if scored else None}


def slot_values(text):
    """Tolerant parse as {slot key -> value}, e.g. {"topic": "飞控固件烧写"}.

    Public face of the parser slot_match uses. Callers that need to know WHICH
    slots carried a real value (rather than just how many matched) read this
    instead of re-implementing the parse, so the two cannot drift apart.
    """
    return dict(_tolerant_slots(text))


def _tolerant_slots(text):
    out = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        name = None
        for label, key in LABEL_TO_SLOT.items():
            if line.startswith(label + "：") or line.startswith(label + ":"):
                name, line = key, line[len(label) + 1:]
                break
        if name is not None:
            out.append((name, line.strip()))
    return out


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
