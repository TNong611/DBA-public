"""Convert RiSAWOZ (GEM) dialogue annotations into four-slot memory cases.

WHAT THIS DOES
--------------
RiSAWOZ is a Chinese task-oriented dialogue corpus (hotel / restaurant /
attraction / train / flight / weather / ...). Every turn carries a *dialogue
state* (`belief_state`) and *system dialogue acts* (`system_actions`). That is
exactly the shape the RWKV-state decoder pilot needs: a closed event (user turn
+ system turn) that can be summarised into the four memory slots the pilot
trains on.

This converter emits cases that satisfy `rwkv_slots_data.py`'s contract:

    id, group, user, assistant, unverified, summary

with `summary` being four Chinese lines in the fixed order

    主题 / 约束 / 未完成 / 结果

derived from the dialogue state, never invented:

    topic      domain + the slot the turn is about
    constraint  slot=value pairs the user informed (this turn, else carried)
    pending     slots requested this turn and not answered in this turn
    result      system acts with a value  -> verified   ("已确认…")
                system acts without a value -> unverified ("尚未取得…")

THE OBSERVATION IS THE WHOLE PREFIX OF THE DIALOGUE
---------------------------------------------------
`user` is NOT a single turn. It is every earlier turn of the dialogue followed
by the current user utterance:

    用户：<turn 0 user>
    系统：<turn 0 system>
    用户：<turn 1 user>
    ...                     (turns strictly before the current one)
    用户：<current user utterance>

`assistant` is the current system utterance. Rationale: the 约束 slot records
*carried* state ("沿用已告知条件：…"), and a label that refers to information the
model cannot see is unsupported. With the full prefix in the observation every
slot is checkable against the input. There is no future text: a turn's system
reply is the only text from that turn, and later turns never appear.

Labels and inputs are BOTH sanitised by the same redaction pass, history
included, so no prior turn can smuggle a telephone number or address in.

TRUTHFULNESS RULES (enforced, not aspirational)
-----------------------------------------------
* A turn is only marked `unverified=False` when a system act actually carries a
  non-empty value. A bare Request/Greeting/Bye/General act never counts as a
  completed inform, so it stays `unverified=True` with a hedging 结果.
* `pending` says "本轮未答" for a non-final turn and "待查" for the dialogue's
  final turn: on a non-final turn the requested information may still be
  delivered later, and the label must not claim otherwise.
* 结果 keeps the turn's factual value when that is safe ("已确认告知票价：120元"),
  so the task is value extraction where possible. It falls back to the generic
  slot form ("已确认告知票价信息") when the value is an identifier, is itself PII,
  is too long, or contains a hedge/action word. Both rates are reported.
* Every produced case is run through `rwkv_slots_data.validate([case])`. A case
  that violates a label gate is SKIPPED and the exact reason is counted, never
  silently rewritten.
* Two cases may not share the same observation (user + assistant) with different
  labels: supervised contradiction is dropped, counted as `conflicting_label`.
* Observation pairs are unique across splits with TRAIN precedence: a dev/test
  case whose observation already exists in train is dropped, and dev/test are
  isolated from each other, so the test file shares no input with training.

PII HANDLING
------------
* `db_results` is never used as case content. It holds telephone numbers and
  street addresses; those fields are excluded by construction.
* System-act *values* are never copied into the summary (they are entity names:
  周庄古镇, 某某酒店). Only act type + slot name are used, and the act's target
  slot name is the generic schema slot (价格/评分/房型/...). Telephone/address
  slot names are aliased away (电话号码 -> 联系方式, 地址 -> 位置).
* Belief-state values are only used for attribute slots. Slots that carry
  identifiers (名称/品牌/地址/电话/邮箱/身份证/号码/编号/车次/航班/...) are
  dropped from the summary entirely.
* User and system utterances are redacted in place: e-mail, mobile, landline,
  400-numbers, ID-card, long digit runs and URLs become [已脱敏].
* After writing, the output is re-scanned for phone/e-mail/ID/digit-run shapes
  and every written case is checked against the PII values extracted from its
  own turn's `db_results`; both checks must come back empty.

WHAT IS STILL TRUE ABOUT THE RESULT (stated so it is not over-claimed)
---------------------------------------------------------------------
* The utterances themselves are raw corpus text. A user or system turn may
  name a shop/hotel/attraction because that is what the dialogue says. The
  converter does not copy those names into memory slots, but it does not
  rewrite the utterances either; the overlap is counted and reported.
* `user` is the full dialogue prefix (see above), and `assistant` is the current
  system turn; together they are one memory-forming event, observed with its
  history. This is still not a recurrent-memory test: the backbone sees one
  concatenated observation, and the 未完成 slot is deliberately labelled from the
  current turn only.
* Splits: `memory_cases.jsonl` = train.json (group `train`) + dev.json (group
  `heldout`), so tuning has a validation split in the same file; and
  `memory_test.jsonl` = test.json (group `heldout`) for final evaluation only.
  Selection, dedup precedence and every label policy are derived from
  train.json and dev.json alone -- test.json is never read for a decision, is
  never used to set a threshold, and shares no observation and no memory text
  with the training file.

Run:
    py -3.14 convert_risawoz_memory.py                     # write both files
    py -3.14 convert_risawoz_memory.py --check-only        # re-validate output
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import rwkv_slots_data as DATA  # noqa: E402

CONVERTER = "convert_risawoz_memory.py"
VERSION = 1

DEFAULT_SOURCE_DIR = Path(r"D:\AI\DBA\training\web-sources\risawoz")
DEFAULT_OUT = DEFAULT_SOURCE_DIR / "memory_cases.jsonl"
DEFAULT_TEST_OUT = DEFAULT_SOURCE_DIR / "memory_test.jsonl"
DEFAULT_PROVENANCE = DEFAULT_SOURCE_DIR / "provenance.json"

SOURCE_URLS = {
    "dataset": "RiSAWOZ (GEM release of the RiSAWOZ corpus)",
    "homepage": "https://terryqj0107.github.io/RiSAWOZ_webpage",
    "repository": "https://github.com/terryqj0107/RiSAWOZ",
    "paper": "https://aclanthology.org/2020.emnlp-main.67",
    "gem_data_card": "https://gem-benchmark.com/data_cards/RiSAWOZ",
    "hf_loader": "https://huggingface.co/datasets/GEM/RiSAWOZ",
}

# source file -> (split tag, id tag, rwkv_slots_data group).
# dev.json is group HELDOUT on purpose: it is the validation split used while
# tuning, kept in the same file as train.json so a training run can evaluate
# without ever touching test.json.
SPLIT_PLAN = (
    ("train", "train.json", "tr", DATA.TRAIN),
    ("dev", "dev.json", "dv", DATA.HELDOUT),
    ("test", "test.json", "te", DATA.HELDOUT),
)
# Which output file each split lands in. Iterated in SPLIT_PLAN order, so train
# and dev claim their observation pairs before test is considered.
TRAIN_FILE_SPLITS = ("train", "dev")
TEST_FILE_SPLITS = ("test",)

DEFAULT_TRAIN_CAP = 10000
DEFAULT_DEV_CAP = 2000
DEFAULT_TEST_CAP = 2000
DEFAULT_SEED = 20261007

# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------
# Order matters: the specific shapes run before the generic digit-run sweep.
REDACTION_RULES = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("url", re.compile(r"https?://\S+")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("mobile", re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)")),
    ("service_400", re.compile(r"(?<!\d)400[-\s]?\d{3}[-\s]?\d{4}(?!\d)")),
    ("landline", re.compile(r"(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)")),
    ("long_digits", re.compile(r"(?<!\d)\d{7,}(?!\d)")),
)
REDACTION_PLACEHOLDER = "[已脱敏]"

# Shapes that must NEVER survive into the output. Checked after writing.
PII_SCAN = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("url", re.compile(r"https?://")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("mobile", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("service_400", re.compile(r"(?<!\d)400[-\s]?\d{3}[-\s]?\d{4}(?!\d)")),
    ("landline", re.compile(r"(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)")),
    ("long_digits", re.compile(r"(?<!\d)\d{7,}(?!\d)")),
)

# Slot names whose *values* identify a person, a business or a contact route.
# Values for these are never written into the summary.
BLOCKED_VALUE_SLOT_PARTS = (
    "名称", "地址", "电话", "手机", "邮箱", "邮件", "网址", "链接", "账号", "卡号",
    "身份证", "姓名", "联系人", "品牌", "片名", "主演", "车次", "航班", "车牌",
    "编号", "订单", "密码", "联系人", "单位名称", "店名", "公司",
)

# Slot names written into the summary get a neutral alias so that a contact
# route is described without echoing the corpus' own telephone/address wording.
SLOT_ALIAS = {
    "电话号码": "联系方式",
    "电话": "联系方式",
    "联系电话": "联系方式",
    "地址": "位置",
    "手机号": "联系方式",
    "邮箱": "联系邮箱",
}

# System act types that may claim a completed inform *only with a value*.
VALUED_ACT_TYPES = ("Inform", "Recommend", "No-offer")
# A reply ending in one of these asks the user something instead of telling them
# something, so a value it carries is an option being quoted, not a delivered fact.
QUESTION_TAIL = ("？", "?")

# Words rwkv_slots_data treats as "actions that must not be claimed done".
ACTION_WORDS = DATA._ACTION_WORDS  # ("验证","测试","部署","完成","修复","通过","解决")

CJK = re.compile(r"[\u4e00-\u9fff]")
CJK_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")
# Speaker prefixes used when the dialogue prefix is folded into `user`.
HISTORY_USER_PREFIX = "用户："
HISTORY_SYSTEM_PREFIX = "系统："
# Guard on the whole rendered prefix. Measured over the corpus the longest turn
# prefix is 1067 characters, so the default never truncates; a prefix above the
# limit is SKIPPED rather than truncated, because a truncated prefix can drop the
# very turn a carried condition came from.
DEFAULT_MAX_CONTEXT_CHARS = 2000
MAX_ASSISTANT_CHARS = 400
MAX_TOPIC_CHARS = 14
MAX_CONSTRAINT_CHARS = 78
MAX_EMBEDDED_VALUE_CHARS = 24


class ConvertError(Exception):
    """A failure that must stop the conversion instead of producing bad data."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def redact(text: str, values=()):
    """Replace PII shapes and known db-derived values.

    `values` is a sequence of (string, kind) pairs taken from the turn's own
    `db_results` -- exact strings we already know are telephone numbers, street
    addresses or names. Those run first and longest-first, so a whole address is
    removed before any shorter substring of it can match. Returns the text, the
    replacement count and the kinds touched.
    """
    count = 0
    kinds = []
    for value, kind in sorted(set(values), key=lambda item: -len(item[0])):
        if value and value in text:
            count += text.count(value)
            text = text.replace(value, REDACTION_PLACEHOLDER)
            kinds.append(kind)
    for name, pattern in REDACTION_RULES:
        text, hits = pattern.subn(REDACTION_PLACEHOLDER, text)
        if hits:
            count += hits
            kinds.append(name)
    return text, count, tuple(dict.fromkeys(kinds))


def scan_pii(text: str):
    """Report surviving PII shapes; used as a post-write gate."""
    found = {}
    for name, pattern in PII_SCAN:
        hits = pattern.findall(text)
        if hits:
            found[name] = len(hits)
    return found


def clean(text) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def split_slot(key: str):
    """`旅游景点-景点类型` -> ('旅游景点', '景点类型'). Unknown shapes pass through."""
    if not isinstance(key, str) or not key:
        return "", ""
    if "-" in key:
        domain, slot = key.split("-", 1)
        return domain.strip(), slot.strip()
    return "", key.strip()


def slot_alias(slot: str) -> str:
    slot = (slot or "").strip()
    return SLOT_ALIAS.get(slot, slot)


def slot_is_blocked(slot: str) -> bool:
    return any(part in (slot or "") for part in BLOCKED_VALUE_SLOT_PARTS)


def safe_condition(key: str, value):
    """A slot=value pair safe to write into the summary, or None.

    Rejected: identifying slots, values carrying PII shapes, values too long to
    be an attribute, values that are themselves sentences.
    """
    domain, slot = split_slot(key)
    slot = slot_alias(slot)
    if not slot or slot_is_blocked(slot):
        return None
    value = clean(value)
    if not value or len(value) > 16:
        return None
    redacted, count, _kinds = redact(value)
    if count:
        return None                      # the value itself was PII
    if value.startswith(REDACTION_PLACEHOLDER):
        return None
    return slot, value


def safe_action_slot(action) -> str:
    """Generic slot name of a system act; entity values are never taken."""
    if not isinstance(action, (list, tuple)) or len(action) < 3:
        return ""
    slot = clean(action[2])
    if slot and not CJK_ONLY.match(slot) and len(slot) > 6:
        return ""
    return slot_alias(slot)


def action_type(action) -> str:
    if not isinstance(action, (list, tuple)) or not action:
        return ""
    return clean(action[0])


def action_has_value(action) -> bool:
    if not isinstance(action, (list, tuple)) or len(action) < 4:
        return False
    return bool(clean(action[3]))


def action_value(action) -> str:
    if not isinstance(action, (list, tuple)) or len(action) < 4:
        return ""
    return clean(action[3])


def is_question_reply(utterance) -> bool:
    """True when the system's reply is interrogative rather than declarative.

    RiSAWOZ act types are noisy in both directions: a turn that plainly answers
    ("推荐您去山崎面包，这家店不错。") can be annotated `Request` and still carry
    the value it delivered, while a genuine question can also carry a value (the
    options it quotes). The act type alone therefore cannot decide whether the
    slot was obtained, and the reply's own shape is the one signal available
    without reading a later turn.
    """
    text = clean(utterance)
    return bool(text) and text.endswith(QUESTION_TAIL)


def embeddable_value(slot, value, unsafe_values=()):
    """The factual value of a completed act, if it is safe to write verbatim.

    Returns "" when the value must not be written: identifying slots (a venue or
    person name), values that are themselves PII, values containing a hedge (the
    result slot would stop reading as a confirmation) or an action word (the
    contract forbids claiming an undone action), over-long values, and any value
    the turn's own `db_results` identified as an entity or contact string.
    """
    slot = (slot or "").strip()
    if not slot or slot_is_blocked(slot):
        return ""
    value = clean(value)
    if not value or len(value) > MAX_EMBEDDED_VALUE_CHARS:
        return ""
    if REDACTION_PLACEHOLDER in value:
        return ""
    if DATA.has_hedge(value) or any(word in value for word in ACTION_WORDS):
        return ""
    for unsafe in unsafe_values:
        if unsafe and (unsafe in value or value in unsafe):
            return ""
    if redact(value)[1]:
        return ""                       # the value itself carries a PII shape
    return value


DB_VALUE_KEYS = {
    "pii": ("电话", "地址", "手机", "邮箱", "邮件", "姓名", "身份证", "账号", "卡号"),
    # Name-ish fields only. A bare "景点"/"酒店"/"医院" marker would also match
    # attribute fields such as 景点类型 or 医院等级 and erase a *type* value from
    # the utterance, which is over-redaction: the type is dialogue content.
    "entity": ("名称", "品牌", "主演", "片名", "店名", "公司名", "演出团体"),
}


def extract_db_values(db_results):
    """Values taken from a turn's `db_results`, used ONLY for redaction/checks.

    Returns {"pii": [...], "entity": [...]} where entity values are venue/title
    names and pii values are contact routes, addresses and person names. Nothing
    here is ever copied into a memory slot; the strings are used to erase
    themselves from the utterances and to prove no leak afterwards.
    """
    found = {"pii": [], "entity": []}
    if not isinstance(db_results, list):
        return found
    pattern = re.compile(r"'([^']*)':\s*'([^']*)'")
    for entry in db_results:
        if not isinstance(entry, str):
            continue
        for key, value in pattern.findall(entry):
            value = clean(value)
            if len(value) < 4:
                continue
            for kind, markers in DB_VALUE_KEYS.items():
                if kind == "entity" and any(m in key for m in DB_VALUE_KEYS["pii"]):
                    continue        # an address field is not an entity name
                if any(marker in key for marker in markers):
                    if value not in found[kind]:
                        found[kind].append(value)
                    break
    return found


def dpii_from_db_results(db_results):
    """Backwards-compatible helper: the contact/address values only."""
    return extract_db_values(db_results)["pii"]


# ---------------------------------------------------------------------------
# dialogue -> case
# ---------------------------------------------------------------------------
def summary_from_turn(domain, informs, carried, requests, actions, is_final_turn,
                      unsafe_values=(), question_reply=False):
    """Build the four-slot memory text.

    Returns (summary, unverified, parts), or None when the turn carries no
    information of its own (nothing informed, requested or delivered), in which
    case the caller skips it instead of teaching a content-free label.
    `unsafe_values` are strings the turn's `db_results` identified as entity
    names or contact routes: they must never be written into a slot.
    `question_reply` says the system's own turn is interrogative: a value-bearing
    act then asked about the slot rather than delivering it, so the turn stays
    unverified and the slot is reported as not yet obtained.
    """
    domain = domain or "通用"

    conditions = []
    for key, value in informs:
        pair = safe_condition(key, value)
        if pair:
            conditions.append(pair)
    carried_conditions = []
    for key, value in carried:
        pair = safe_condition(key, value)
        if pair and pair not in conditions:
            carried_conditions.append(pair)

    slots = []
    for action in actions:
        slot = safe_action_slot(action)
        if slot and slot not in slots:
            slots.append(slot)

    # The topic names the SLOT under discussion (出发地), never one of its values
    # (厦门): the value already lives in the constraint slot, and a value in the
    # topic would both duplicate it and mislabel what the memory is about.
    focus = ""
    focus_candidates = [slot for slot, _value in conditions]
    focus_candidates += [slot_alias(clean(request)) for request in requests]
    focus_candidates += slots
    for candidate in focus_candidates:
        candidate = clean(candidate)
        if candidate and CJK_ONLY.match(candidate) and len(candidate) <= 8:
            focus = candidate
            break

    topic = ""
    if focus and focus == domain:
        focus = ""                                  # the slot name IS the domain
    if focus and focus.startswith(domain):
        topic = focus          # the schema slot already carries the domain
    elif focus:
        # 旅游景点 + 景点类型 would read as 旅游景点景点类型; drop the overlap
        # (two characters or more, so 火车 + 车次信息 is left alone).
        for size in range(min(len(domain), len(focus)), 1, -1):
            if domain.endswith(focus[:size]):
                focus = focus[size:]
                break
        topic = (domain + focus) if focus else (domain + "咨询")
    else:
        topic = domain + "咨询"
    if len(topic) > MAX_TOPIC_CHARS:
        topic = domain + "咨询"
    if len(topic) > MAX_TOPIC_CHARS:
        topic = "通用咨询"

    if conditions and carried_conditions:
        constraint = ("本轮条件：" + "、".join("%s=%s" % pair for pair in conditions[:2])
                      + "；沿用：" + "、".join("%s=%s" % pair for pair in carried_conditions[:2]))
    elif conditions:
        constraint = "本轮用户条件：" + "、".join("%s=%s" % pair for pair in conditions[:2])
    elif carried_conditions:
        constraint = "沿用已告知条件：" + "、".join("%s=%s" % pair for pair in carried_conditions[:2])
    else:
        constraint = "无"
    if len(constraint) > MAX_CONSTRAINT_CHARS:
        constraint = constraint[:MAX_CONSTRAINT_CHARS].rstrip("、；：") + "…"

    answered = set()
    for action in actions:
        if action_has_value(action):
            slot = safe_action_slot(action)
            if slot:
                answered.add(slot)

    pending_names = []
    for request in requests:
        slot = slot_alias(clean(request))
        if not slot or slot in answered or slot in pending_names:
            continue
        if slot_is_blocked(slot) and slot not in ("联系方式", "位置"):
            continue
        pending_names.append(slot)

    if pending_names:
        marker = "待查：" if is_final_turn else "本轮未答："
        pending = marker + "、".join(pending_names[:3])
    else:
        pending = "无"

    value_acts = [a for a in actions if action_has_value(a)]
    valued = [a for a in value_acts if action_type(a) in VALUED_ACT_TYPES]
    act_type_noise = False
    if not valued and value_acts and not question_reply:
        # The corpus annotates some clear answers as Request/Greeting/..., and
        # those acts still carry the value the reply delivered. A declarative
        # reply did deliver it, so the turn counts as verified; an interrogative
        # one did not, and stays unverified below.
        valued = value_acts[:1]
        act_type_noise = True
    question_value_act = bool(value_acts) and question_reply and not valued
    informative = bool(conditions or pending_names or valued)

    # A turn that informed nothing, requested nothing and delivered nothing is
    # still worth keeping when the state carries standing conditions: the label
    # then records those conditions and states plainly that this turn confirmed
    # nothing new. Only a turn with no state at all is dropped as content-free.
    if not informative and not carried_conditions:
        return None

    fact = ""
    if valued:
        unverified = False
        act = action_type(valued[0])
        act_slot = safe_action_slot(valued[0])
        # The result may only name the slot the valued act actually delivered.
        # Falling back to an unrelated slot of the same turn would claim delivery
        # of something the turn never delivered.
        target = act_slot
        fact = embeddable_value(act_slot, action_value(valued[0]), unsafe_values)
        if act == "Recommend" and target and fact:
            result = "已确认给出%s推荐：%s" % (target, fact)
        elif act == "Recommend" and target:
            result = "已确认给出%s推荐" % target
        elif act == "No-offer":
            result = ("已确认无匹配%s的结果" % target) if target else "已确认无匹配结果"
            fact = ""          # "no matching option" carries no value to quote
        elif target and fact:
            result = "已确认告知%s：%s" % (target, fact)
        elif target:
            result = "已确认告知%s信息" % target
        else:
            result = "已确认回应本轮咨询"
        if any(word in result for word in ACTION_WORDS):
            result = "已确认回应本轮咨询"
            fact = ""
    elif not informative:
        unverified = True
        result = "本轮未确认新增信息"
    else:
        unverified = True
        types = [t for t in (action_type(a) for a in actions) if t]
        target = slots[0] if slots else ""
        if target:
            result = "尚未取得%s信息" % target
        elif types:
            result = "尚未取得本轮答复"
        else:
            result = "尚未取得本轮结论"
        if not DATA.has_hedge(result):
            raise ConvertError("internal error: unverified result lost its hedge")

    summary = "\n".join((
        DATA.SLOT_PREFIX["topic"] + topic,
        DATA.SLOT_PREFIX["constraint"] + constraint,
        DATA.SLOT_PREFIX["pending"] + pending,
        DATA.SLOT_PREFIX["result"] + result,
    ))
    parts = {"topic": topic, "constraint": constraint, "pending": pending,
             "result": result, "value": fact, "act_type_noise": act_type_noise,
             "question_value_act": question_value_act}
    return summary, unverified, parts


def sanitize_dialogue(turns, redact_entities=True):
    """Redact every turn once, so the whole prefix is sanitised identically.

    Returns a list aligned with `turns`, each entry carrying the redacted user
    and system text, the db-derived values of that turn, and how many
    replacements the redaction made.
    """
    cleaned = []
    for turn in turns:
        db_values = extract_db_values(turn.get("db_results")) if isinstance(turn, dict) else {"pii": [], "entity": []}
        pairs = [(value, "db_contact_or_address") for value in db_values["pii"]]
        if redact_entities:
            pairs += [(value, "db_entity_name") for value in db_values["entity"]]
        entry = {"pii": db_values["pii"], "entity": db_values["entity"],
                 "hits": 0, "kinds": ()}
        for key in ("user_utterance", "system_utterance"):
            text, hits, kinds = redact(clean(turn.get(key) if isinstance(turn, dict) else ""), pairs)
            entry[key] = text
            entry["hits"] += hits
            entry["kinds"] = tuple(dict.fromkeys(entry["kinds"] + tuple(kinds)))
        cleaned.append(entry)
    return cleaned


def render_prefix(sanitized, position):
    """The observation prefix: every earlier turn, then the current user turn.

    Only turns strictly before `position` contribute, and from those turns both
    speakers appear, so the text is exactly the dialogue as it stood when the
    current user spoke -- no future text can leak in.
    """
    lines = []
    for entry in sanitized[:position]:
        if entry["user_utterance"]:
            lines.append(HISTORY_USER_PREFIX + entry["user_utterance"])
        if entry["system_utterance"]:
            lines.append(HISTORY_SYSTEM_PREFIX + entry["system_utterance"])
    current = sanitized[position]["user_utterance"] if position < len(sanitized) else ""
    lines.append(HISTORY_USER_PREFIX + current)
    return "\n".join(lines), current


def turn_candidates(dialogue, split_tag, index, redact_entities=True,
                    max_context_chars=DEFAULT_MAX_CONTEXT_CHARS):
    """Yield case drafts for one dialogue, final turn first."""
    turns = [t for t in (dialogue.get("dialogue") or []) if isinstance(t, dict)]
    sanitized = sanitize_dialogue(turns, redact_entities)
    usable = []
    for position, turn in enumerate(turns):
        belief = turn.get("belief_state") or {}
        actions = [a for a in (turn.get("system_actions") or []) if action_type(a)]
        if not isinstance(belief, dict) or not beliefs_usable(belief) or not actions:
            continue
        usable.append((position, turn, belief, actions))

    ordered = []
    if usable:
        ordered.append((usable[-1], True))
        for item in reversed(usable[:-1]):
            ordered.append((item, False))

    drafts = []
    for (position, turn, belief, actions), is_final in ordered:
        draft = build_draft(dialogue, turn, belief, actions, is_final, split_tag, index,
                            position, sanitized, redact_entities, max_context_chars)
        if draft is not None:
            drafts.append(draft)
    return drafts


def beliefs_usable(belief):
    for key in ("inform slot-values", "turn_inform", "turn request"):
        if key in belief:
            return True
    return False


def build_draft(dialogue, turn, belief, actions, is_final, split_tag, index, position,
                sanitized, redact_entities=True, max_context_chars=DEFAULT_MAX_CONTEXT_CHARS):
    context, user_raw = render_prefix(sanitized, position)
    assistant_raw = sanitized[position]["system_utterance"] if position < len(sanitized) else ""
    if not user_raw.strip() or not assistant_raw.strip():
        return {"skip": "empty_utterance"}
    if len(context) > max_context_chars:
        return {"skip": "context_too_long"}
    if len(assistant_raw) > MAX_ASSISTANT_CHARS:
        return {"skip": "assistant_too_long"}

    # Redaction already ran over the whole dialogue; the counters below describe
    # the text this case actually ships (prefix + current system turn).
    user, assistant = context, assistant_raw
    user_hits = sum(entry["hits"] for entry in sanitized[:position])
    user_kinds = tuple(kind for entry in sanitized[:position] for kind in entry["kinds"])
    current = sanitized[position]
    assistant_hits = current["hits"]
    assistant_kinds = current["kinds"]
    unsafe_values = list(current["pii"]) + list(current["entity"])

    informs = pairs_of(belief.get("turn_inform"))
    carried = pairs_of(belief.get("inform slot-values"))
    requests = [clean(r) for r in as_list(belief.get("turn request")) if clean(r)]

    domain = ""
    turn_domain = as_list(turn.get("turn_domain"))
    if turn_domain:
        domain = clean(turn_domain[0])
    if not domain:
        for action in actions:
            if isinstance(action, (list, tuple)) and len(action) > 1 and clean(action[1]):
                domain = clean(action[1])
                break
    if not domain:
        domains = as_list(dialogue.get("domains"))
        domain = clean(domains[0]) if domains else "通用"

    try:
        built = summary_from_turn(domain, informs, carried, requests, actions, is_final,
                                  unsafe_values,
                                  question_reply=is_question_reply(
                                      current["system_utterance"] if current else ""))
    except ConvertError as exc:
        return {"skip": "summary_build_error", "detail": str(exc)}
    if built is None:
        return {"skip": "no_turn_information"}
    summary, unverified, parts = built

    turn_id = turn.get("turn_id", position)
    case = {
        "id": "RS-%s-%05d-%02d" % (split_tag, index, int(turn_id) if str(turn_id).isdigit() else position),
        "group": None,      # filled by the caller (assigned to a split group)
        "user": user,
        "assistant": assistant,
        "unverified": unverified,
        "summary": summary,
    }
    return {
        "case": case,
        "parts": parts,
        "stratum": (domain, action_type(actions[0])),
        "final_turn": is_final,
        "redaction": {
            "user": user_hits, "assistant": assistant_hits,
            "kinds": tuple(dict.fromkeys(user_kinds + assistant_kinds)),
        },
        "turn": turn,
        "position": position,
        "dialogue_index": index,
        "context_chars": len(context),
        "dialogue_id": clean(dialogue.get("dialogue_id")),
    }


def pairs_of(mapping):
    out = []
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            out.append((key, value))
    return out


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def select(drafts, cap, seen_summaries, seen_pairs, reason_counts, group):
    """Stratified round-robin over (domain, act type); one pass per turn tier.

    `seen_pairs` maps an observation (user, assistant) to the summary that was
    already accepted for it. It lives across splits: train and dev are selected
    before test, so a repeated observation is always resolved in favour of the
    earlier split (train precedence) and test can never share an input with the
    training file. A pair seen with a *different* label is a supervision
    contradiction and is dropped.
    """
    by_tier = {True: collections.defaultdict(list), False: collections.defaultdict(list)}
    for draft in drafts:
        if "skip" in draft:
            reason_counts[draft["skip"]] += 1
            continue
        by_tier[draft["final_turn"]][draft["stratum"]].append(draft)

    chosen = []
    for tier in (True, False):
        strata = by_tier[tier]
        if not strata:
            continue
        keys = sorted(strata)
        queues = [list(strata[key]) for key in keys]
        cursor = [0] * len(queues)
        while len(chosen) < cap:
            progressed = False
            for position, queue in enumerate(queues):
                if cursor[position] >= len(queue):
                    continue
                progressed = True
                draft = queue[cursor[position]]
                cursor[position] += 1
                case = draft["case"]
                summary = case["summary"]
                key = (case["user"], case["assistant"])
                previous = seen_pairs.get(key)
                if previous is not None:
                    reason_counts["conflicting_label" if previous != summary
                                  else "duplicate_observation"] += 1
                    continue
                if summary in seen_summaries:
                    reason_counts["duplicate_summary"] += 1
                    continue
                case["group"] = group
                try:
                    DATA.validate([case])
                except ValueError as exc:
                    reason_counts[classify_skip(str(exc))] += 1
                    continue
                seen_summaries.add(summary)
                seen_pairs[key] = summary
                chosen.append(draft)
                if len(chosen) >= cap:
                    break
            if not progressed:
                break
    return chosen


def classify_skip(message: str) -> str:
    """Normalise a rwkv_slots_data validation message into a reason key."""
    text = message
    for marker, key in (
        ("is marked unverified but its result reads as done", "reject_result_says_done"),
        ("records an explicitly undone action as done", "reject_claims_undone_action"),
        ("states no confirmation", "reject_verified_without_confirmation"),
        ("copies its source verbatim", "reject_copies_source"),
        ("is a sentence, not a noun phrase", "reject_topic_too_long"),
        ("longer than 200 characters", "reject_summary_too_long"),
        ("must have exactly the four slots", "reject_slot_order"),
        ("slot line has no label", "reject_slot_label"),
        ("unknown slot label", "reject_slot_label"),
        ("is empty", "reject_empty_slot"),
        ("duplicate memory text", "duplicate_summary"),
        ("duplicate case id", "duplicate_id"),
    ):
        if marker in text:
            return key
    return "reject_other"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def split_caps(source_dir: Path, train_cap: int, dev_cap: int, test_cap: int):
    """Per-source caps; each split has its own budget (train / dev / test)."""
    for _tag, filename, _id_tag, _group in SPLIT_PLAN:
        path = source_dir / filename
        if not path.is_file():
            raise ConvertError("missing source file: " + str(path))
    return {"train": train_cap, "dev": dev_cap, "test": test_cap}


def build(source_dir: Path, train_cap: int, dev_cap: int, test_cap: int, seed: int,
          max_dialogues=None, redact_entities=True,
          max_context_chars=DEFAULT_MAX_CONTEXT_CHARS):
    reason_counts = collections.Counter()
    seen_summaries = set()
    # Observation -> accepted summary. Shared by every split on purpose: it is
    # what gives train precedence and keeps the test file disjoint from training.
    seen_pairs = {}
    cases_by_file = {"train_file": [], "test_file": []}
    source_stats = []
    redaction_stats = {"cases": 0, "replacements": 0,
                       "kinds": collections.Counter(), "by_split": collections.Counter()}
    strata_counts = collections.Counter()
    value_stats = {"result_carries_value": 0, "result_generic": 0,
                   "unverified_result": 0, "value_chars": 0,
                   "act_type_noise_credited": 0, "interrogative_value_act": 0}
    context_stats = {"max_chars": 0, "sum_chars": 0, "with_history": 0}
    entity_stats = {"cases_with_entity_value": 0, "cases_entity_redacted": 0,
                    "partial_mentions": 0, "utterance_entity_names_left": 0}

    caps = split_caps(source_dir, train_cap, dev_cap, test_cap)
    progress = []

    for tag, filename, id_tag, group in SPLIT_PLAN:
        path = source_dir / filename
        if not path.is_file():
            raise ConvertError("missing source file: " + str(path))
        digest = sha256_file(path)
        started = time.perf_counter()
        with path.open("r", encoding="utf-8") as handle:
            dialogues = json.load(handle)
        if not isinstance(dialogues, list):
            raise ConvertError("source file is not a list of dialogues: " + str(path))
        turns_total = sum(len(d.get("dialogue") or []) for d in dialogues if isinstance(d, dict))

        cap = caps[tag]
        drafts = []
        for index, dialogue in enumerate(dialogues):
            if max_dialogues is not None and index >= max_dialogues:
                break
            if not isinstance(dialogue, dict):
                reason_counts["dialogue_not_object"] += 1
                continue
            drafts.extend(turn_candidates(dialogue, id_tag, index, redact_entities,
                                          max_context_chars))

        merged = collections.defaultdict(list)
        split_skips = collections.Counter()
        for draft in drafts:
            if "skip" in draft:
                reason_counts[draft["skip"]] += 1
                split_skips[draft["skip"]] += 1
                continue
            merged[draft["stratum"]].append(draft)
        ordered_drafts = []
        for key in sorted(merged):
            ordered_drafts.extend(merged[key])
        chosen = select(ordered_drafts, cap, seen_summaries, seen_pairs, reason_counts, group)

        # PII leak gate: compare every written case against its own turn's db_results.
        kept = []
        for draft in chosen:
            values = extract_db_values(draft["turn"].get("db_results"))
            text = draft["case"]["user"] + "\n" + draft["case"]["assistant"] + "\n" + draft["case"]["summary"]
            leaking = [value for value in values["pii"] if value in text]
            if leaking:
                reason_counts["reject_db_pii_leak"] += 1
                continue
            entity_hits = [value for value in values["entity"] if value in text]
            if values["entity"]:
                entity_stats["cases_with_entity_value"] += 1
                if redact_entities:
                    entity_stats["cases_entity_redacted"] += 1
            entity_stats["utterance_entity_names_left"] += len(entity_hits)
            # Residual that redaction cannot remove: a partial mention such as
            # "周庄" when the database name is "周庄古镇".
            for value in values["entity"]:
                stem = value[:2]
                if len(value) >= 3 and stem in draft["case"]["user"] + draft["case"]["assistant"]:
                    entity_stats["partial_mentions"] += 1
                    break
            kept.append(draft)

        for draft in kept:
            redaction = draft["redaction"]
            if redaction["user"] or redaction["assistant"]:
                redaction_stats["cases"] += 1
            redaction_stats["replacements"] += redaction["user"] + redaction["assistant"]
            for kind in redaction["kinds"]:
                redaction_stats["kinds"][kind] += 1
            if redaction["user"] or redaction["assistant"]:
                redaction_stats["by_split"][tag] += 1
            strata_counts["%s/%s" % draft["stratum"]] += 1
            if draft["parts"]["value"]:
                value_stats["result_carries_value"] += 1
                value_stats["value_chars"] += len(draft["parts"]["value"])
            elif draft["case"]["unverified"]:
                value_stats["unverified_result"] += 1
            else:
                value_stats["result_generic"] += 1
            if draft["parts"]["act_type_noise"]:
                value_stats["act_type_noise_credited"] += 1
            if draft["parts"]["question_value_act"]:
                value_stats["interrogative_value_act"] += 1
            context_stats["max_chars"] = max(context_stats["max_chars"], draft["context_chars"])
            context_stats["sum_chars"] += draft["context_chars"]
            if draft["position"] > 0:
                context_stats["with_history"] += 1
            draft.pop("turn", None)

        # `kept` holds drafts (they still carry provenance for the accounting
        # below); only the six contract fields go into the case list.
        target = "test_file" if tag in TEST_FILE_SPLITS else "train_file"
        cases_by_file[target].extend(draft["case"] for draft in kept)
        source_stats.append({
            "split": tag,
            "file": filename,
            "group": group,
            "out_file": target,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "dialogues": len(dialogues),
            "turns": turns_total,
            "candidate_turns": len(drafts),
            "skipped_in_split": dict(sorted(split_skips.items())),
            "cases_written": len(kept),
            "cap": cap,
            "seconds": round(time.perf_counter() - started, 2),
        })
        progress.append("%s: %d dialogues, %d turns -> %d cases (%s)"
                        % (tag, len(dialogues), turns_total, len(kept), target))
        del dialogues, drafts, merged, ordered_drafts

    return {
        "cases_by_file": cases_by_file,
        "sources": source_stats,
        "reasons": reason_counts,
        "redaction": redaction_stats,
        "strata": strata_counts,
        "entity": entity_stats,
        "values": value_stats,
        "context": context_stats,
        "progress": progress,
    }


def final_validation(cases, severity_counter):
    """Validate the whole list the way the trainer loads it; drop offenders."""
    while True:
        try:
            DATA.validate(list(cases))
            return cases
        except ValueError as exc:
            message = str(exc)
            # Every case this converter emits carries a `RS-<split>-<dlg>-<turn>`
            # id, so the offender can be pulled straight out of the message.
            offender = None
            for target in re.findall(r"RS-[A-Za-z0-9]+-[0-9]+-[0-9]+", message):
                offender = next((c for c in cases if c["id"] == target), None)
                if offender is not None:
                    break
            if offender is None:
                summary_match = re.search(r"duplicate memory text across cases: (.+)$",
                                          message)
                if summary_match:
                    wanted = summary_match.group(1).strip()
                    offender = next((c for c in cases if c["summary"] == wanted), None)
            severity_counter[classify_skip(message)] += 1
            if offender is None:
                raise ConvertError("final validation failed and the offender could not "
                                   "be identified: " + message)
            cases = [c for c in cases if c is not offender]


def write_cases(path: Path, cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = ("id", "group", "user", "assistant", "unverified", "summary")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            payload = {key: case[key] for key in fields}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
    temporary.replace(path)
    return path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="training + validation cases: train.json (group train) + "
                             "dev.json (group heldout, used for tuning)")
    parser.add_argument("--test-out", default=str(DEFAULT_TEST_OUT),
                        help="final-only evaluation cases from test.json (group heldout)")
    parser.add_argument("--provenance", default=str(DEFAULT_PROVENANCE))
    parser.add_argument("--train-cap", type=int, default=DEFAULT_TRAIN_CAP)
    parser.add_argument("--dev-cap", type=int, default=DEFAULT_DEV_CAP)
    parser.add_argument("--test-cap", type=int, default=DEFAULT_TEST_CAP)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS,
                        help="skip a turn whose full dialogue prefix exceeds this; never truncates")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-dialogues", type=int, default=None,
                        help="smoke test only: stop after N dialogues per split")
    parser.add_argument("--redact-entity-names", dest="redact_entities", action="store_true",
                        default=True,
                        help="erase venue/title names taken from db_results in the "
                             "utterances as well (default: on)")
    parser.add_argument("--no-redact-entity-names", dest="redact_entities", action="store_false",
                        help="keep venue/title names in the utterances; they still never "
                             "enter a memory slot")
    parser.add_argument("--check-only", action="store_true",
                        help="re-validate existing output files and exit")
    args = parser.parse_args(argv)

    source_dir = Path(args.source_dir)
    out_path = Path(args.out)
    test_out_path = Path(args.test_out)
    provenance_path = Path(args.provenance)

    if args.check_only:
        report = {"ok": True}
        loaded = {}
        for key, path in (("train_file", out_path), ("test_file", test_out_path)):
            if not path.is_file():
                raise SystemExit(json.dumps({"ok": False, "error": "no output file: " + str(path)},
                                            ensure_ascii=False))
            cases = DATA.load_events(str(path))
            DATA.validate(cases)
            combined = DATA.all_cases(str(path))
            loaded[key] = cases
            report[key] = {"path": str(path), "cases": len(cases),
                           "groups": dict(collections.Counter(c["group"] for c in cases)),
                           "integration_cases": len(combined),
                           "sha256": sha256_file(path),
                           "data_version": DATA.data_version(cases)}
        shared_pairs = ({(c["user"], c["assistant"]) for c in loaded["train_file"]}
                        & {(c["user"], c["assistant"]) for c in loaded["test_file"]})
        shared_summaries = ({c["summary"] for c in loaded["train_file"]}
                            & {c["summary"] for c in loaded["test_file"]})
        report["cross_file"] = {"shared_observation_pairs": len(shared_pairs),
                                "shared_summaries": len(shared_summaries)}
        if shared_pairs or shared_summaries:
            report["ok"] = False
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1

    # The loader no longer falls back to a cwd template.json when an explicit
    # cases path is given, but the trainer's own default path still can, so a
    # stray template.json in cwd is recorded as a warning signal.
    stray_template = Path.cwd() / "template.json"
    result = build(source_dir, args.train_cap, args.dev_cap, args.test_cap, args.seed,
                   max_dialogues=args.max_dialogues,
                   redact_entities=args.redact_entities,
                   max_context_chars=args.max_context_chars)

    dropped_reasons = collections.Counter()
    paths = {"train_file": out_path, "test_file": test_out_path}
    cases_by_file, integration_by_file = {}, {}
    for key in ("train_file", "test_file"):
        cases = final_validation(result["cases_by_file"][key], dropped_reasons)
        write_cases(paths[key], cases)
        cases_by_file[key] = cases
        entry = {"check": "rwkv_slots_data.all_cases(cases_path=...)",
                 "note": "embedded cases + converter output, validated together"}
        try:
            combined = DATA.all_cases(str(paths[key]))
            entry["ok"] = True
            entry["cases"] = len(combined)
            entry["train"] = sum(1 for c in combined if c["group"] == DATA.TRAIN)
            entry["heldout"] = sum(1 for c in combined if c["group"] == DATA.HELDOUT)
        except ValueError as exc:
            entry["ok"] = False
            entry["error"] = str(exc)
            raise ConvertError("integration validation failed for %s: %s" % (key, exc))
        integration_by_file[key] = entry

    # Post-write PII scan over the real file contents, and cross-file isolation.
    scan_counts = collections.Counter()
    scan_examples = []
    per_file = {}
    for key in ("train_file", "test_file"):
        cases = cases_by_file[key]
        groups = collections.Counter()
        unverified_counts = collections.Counter()
        domains = collections.Counter()
        lengths = {"max_user_chars": 0, "max_assistant_chars": 0, "max_summary_chars": 0}
        for case in cases:
            groups[case["group"]] += 1
            unverified_counts["unverified" if case["unverified"] else "verified"] += 1
            domains[case["summary"].split("\n")[0].replace(DATA.SLOT_PREFIX["topic"], "")] += 1
            lengths["max_user_chars"] = max(lengths["max_user_chars"], len(case["user"]))
            lengths["max_assistant_chars"] = max(lengths["max_assistant_chars"], len(case["assistant"]))
            lengths["max_summary_chars"] = max(lengths["max_summary_chars"], len(case["summary"]))
            for field in ("user", "assistant", "summary"):
                for name, hits in scan_pii(case[field]).items():
                    scan_counts[name] += hits
                    if len(scan_examples) < 5:
                        scan_examples.append({"id": case["id"], "field": field, "kind": name})
        summaries_per_pair = collections.defaultdict(set)
        for case in cases:
            summaries_per_pair[(case["user"], case["assistant"])].add(case["summary"])
        per_file[key] = {
            "path": str(paths[key]),
            "bytes": paths[key].stat().st_size,
            "sha256": sha256_file(paths[key]),
            "cases": len(cases),
            "by_group": dict(groups),
            "by_source_split": {entry["split"]: entry["cases_written"] for entry in result["sources"]
                                if entry["out_file"] == key},
            "verified": unverified_counts["verified"],
            "unverified": unverified_counts["unverified"],
            "distinct_topics": len(domains),
            "distinct_ids": len({case["id"] for case in cases}),
            "distinct_summaries": len({case["summary"] for case in cases}),
            "distinct_observation_pairs": len(summaries_per_pair),
            "observation_pairs_with_several_labels": sum(
                1 for values in summaries_per_pair.values() if len(values) > 1),
            "longest": lengths,
            "data_version": DATA.data_version(cases),
            "top_topics": dict(domains.most_common(10)),
        }

    if scan_counts:
        raise ConvertError("PII scan found surviving shapes in the output: "
                           + json.dumps(dict(scan_counts), ensure_ascii=False))

    train_cases, test_cases = cases_by_file["train_file"], cases_by_file["test_file"]
    shared_pairs = ({(c["user"], c["assistant"]) for c in train_cases}
                    & {(c["user"], c["assistant"]) for c in test_cases})
    shared_summaries = ({c["summary"] for c in train_cases} & {c["summary"] for c in test_cases})
    if shared_pairs or shared_summaries:
        raise ConvertError("train/test isolation broken: %d shared observations, %d shared summaries"
                           % (len(shared_pairs), len(shared_summaries)))

    value_stats = result["values"]
    labelled = value_stats["result_carries_value"] + value_stats["result_generic"]
    provenance = {
        "converter": CONVERTER,
        "version": VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dataset": SOURCE_URLS,
        "sources": result["sources"],
        "outputs": per_file,
        "counts": {
            "cases_total": len(train_cases) + len(test_cases),
            "by_file": {"train_file": len(train_cases), "test_file": len(test_cases)},
            "by_group": dict(collections.Counter(c["group"] for c in train_cases + test_cases)),
            "verified": per_file["train_file"]["verified"] + per_file["test_file"]["verified"],
            "unverified": per_file["train_file"]["unverified"] + per_file["test_file"]["unverified"],
            "strata_used": len(result["strata"]),
            "label_balance_note": "RiSAWOZ delivers information in roughly 82% of turns, so the "
                                  "hedged (unverified) class is naturally the smaller one. Cases "
                                  "whose state carries standing conditions are labelled "
                                  "'本轮未确认新增信息' rather than being dropped.",
        },
        "context_policy": {
            "observation": "user = every earlier turn of the dialogue (用户/系统 lines) followed "
                           "by the current user utterance; assistant = the current system turn",
            "no_future_text": "only turns strictly before the current one contribute, and a turn's "
                              "own system reply appears only as `assistant`",
            "padding": "no padding, no ellipsis, no truncation: a prefix above "
                       "--max-context-chars is skipped as context_too_long",
            "max_context_chars": args.max_context_chars,
            "longest_prefix_chars": result["context"]["max_chars"],
            "mean_prefix_chars": (round(result["context"]["sum_chars"]
                                        / max(1, sum(entry["cases_written"] for entry in result["sources"])), 1)),
            "cases_with_history": result["context"]["with_history"],
            "skipped_context_too_long": result["reasons"].get("context_too_long", 0),
            "history_sanitised": "prior turns go through the same redaction pass as the current "
                                 "turn (db-derived contact/entity strings + PII shapes)",
        },
        "contradiction_policy": {
            "rule": "an observation (user + assistant) may carry exactly one label; a repeated "
                    "observation with a different summary is dropped as conflicting_label",
            "conflicting_label_dropped": result["reasons"].get("conflicting_label", 0),
            "duplicate_observation_dropped": result["reasons"].get("duplicate_observation", 0),
            "cross_split_precedence": "train.json, then dev.json, then test.json claim their "
                                      "observation pairs in that order; test never wins over a "
                                      "training case",
            "train_test_shared_observations": len(shared_pairs),
            "train_test_shared_summaries": len(shared_summaries),
        },
        "split_policy": {
            "train_file": {"path": str(out_path), "sources": list(TRAIN_FILE_SPLITS),
                           "groups": {"train.json": DATA.TRAIN, "dev.json": DATA.HELDOUT}},
            "test_file": {"path": str(test_out_path), "sources": list(TEST_FILE_SPLITS),
                          "groups": {"test.json": DATA.HELDOUT}},
            "tuning_split": "dev.json is the validation split for tuning; it lives in the "
                            "training file as group heldout so a training run can evaluate "
                            "checkpoints without opening the test file",
            "test_is_final_only": "test.json is written to its own file and is never used for "
                                  "selection, dedup precedence, thresholds or any tuning "
                                  "decision; the converter's own checks are limited to "
                                  "contract validation of the file it wrote",
            "caps": {"train.json": args.train_cap, "dev.json": args.dev_cap,
                     "test.json": args.test_cap},
            "seed": args.seed,
            "sampling": "stratified round-robin over (domain, first system act type), "
                        "final turn of each dialogue first, then earlier turns",
            "determinism": "file order + sorted strata keys; no RNG used in selection",
        },
        "label_policy": {
            "topic": "domain + the turn's slot (CJK only, <= %d chars)" % MAX_TOPIC_CHARS,
            "constraint": "slot=value pairs the user informed (this turn, else carried); 无 if none",
            "pending": "slots requested this turn and unanswered this turn; "
                       "'本轮未答' on non-final turns, '待查' on the dialogue's final turn",
            "result": "system act with a non-empty value -> verified (已确认…); "
                      "otherwise unverified (尚未取得…)",
            "verified_rule": "an Inform/Recommend/No-offer act with a value counts as a delivery; "
                             "a Request/Greeting/General/Bye act does not, unless it carries a "
                             "value AND the reply is declarative (the corpus mislabels answers as "
                             "Request/Greeting) -- an interrogative reply then asked about the "
                             "slot and delivered nothing",
            "value_policy": "结果 carries the act's factual value when it is safe "
                            "(已确认告知%s：值), else the generic slot form (已确认告知%s信息)" % ("<槽>", "<槽>"),
            "value_blocked_when": "slot is an identifier slot, the value is itself PII, is longer "
                                  "than %d chars, or contains a hedge/action word"
                                  % MAX_EMBEDDED_VALUE_CHARS,
            "reduced_task_note": "where no value is embedded the label is slot-level only "
                                 "(reduced extraction): it says which slot was handled, not what "
                                 "the value was. Counts below state how often that happens.",
        },
        "value_stats": {
            "verified_result_with_value": value_stats["result_carries_value"],
            "verified_result_generic": value_stats["result_generic"],
            "unverified_result": value_stats["unverified_result"],
            "value_rate_in_verified": (round(value_stats["result_carries_value"] / labelled, 4)
                                       if labelled else None),
            "total_value_chars": value_stats["value_chars"],
            "act_type_noise_credited": value_stats["act_type_noise_credited"],
            "interrogative_value_act": value_stats["interrogative_value_act"],
            "act_type_note": "RiSAWOZ act types are noisy: a reply that plainly answers can be "
                             "annotated Request/Greeting/... and still carry the delivered value. "
                             "A value-bearing act of such a type counts as a delivery only when "
                             "the reply is declarative; an interrogative reply keeps the hedged "
                             "(not yet obtained) label, because there the value is an option "
                             "being quoted. act_type_noise_credited / interrogative_value_act "
                             "give the two counts.",
        },
        "skipped": {
            "total": sum(result["reasons"].values()) + sum(dropped_reasons.values()),
            "by_reason": dict(sorted((result["reasons"] + dropped_reasons).items(),
                                     key=lambda item: (-item[1], item[0]))),
        },
        "redaction": {
            "rules": [name for name, _pattern in REDACTION_RULES],
            "placeholder": REDACTION_PLACEHOLDER,
            "cases_touched": result["redaction"]["cases"],
            "replacements": result["redaction"]["replacements"],
            "by_kind": dict(result["redaction"]["kinds"]),
            "by_source_split": dict(result["redaction"]["by_split"]),
            "blocked_value_slots": list(BLOCKED_VALUE_SLOT_PARTS),
            "slot_alias": SLOT_ALIAS,
            "utterance_entity_redaction_enabled": bool(args.redact_entities),
            "applies_to_history": True,
            "cases_with_db_entity_value": result["entity"]["cases_with_entity_value"],
            "cases_entity_names_erased": result["entity"]["cases_entity_redacted"],
            "entity_names_surviving_in_utterances": result["entity"]["utterance_entity_names_left"],
            "cases_with_partial_entity_mention": result["entity"]["partial_mentions"],
        },
        "pii_assertions": {
            "db_results_used_as_case_content": False,
            "db_results_read_only_for_redaction_and_checks": True,
            "entity_values_from_system_acts_in_summary": False,
            "output_pii_scan": {"shapes_found": dict(scan_counts), "examples": scan_examples},
            "per_case_db_leak_check": "every written case is compared against the telephone/"
                                      "address/contact values parsed out of its own turn's "
                                      "db_results; a surviving match is dropped, not published",
            "history_included_in_scan": "the whole `user` prefix is scanned, not just the last turn",
            "cases_dropped_for_db_leak": result["reasons"].get("reject_db_pii_leak", 0),
            "entity_names_surviving_in_utterances": result["entity"]["utterance_entity_names_left"],
            "residual_limitation": "utterances are raw corpus text. Exact venue/person names from "
                                   "db_results are erased when entity redaction is on, but a "
                                   "partial mention (e.g. 周庄 for 周庄古镇) can remain; those "
                                   "cases are counted as cases_with_partial_entity_mention. No "
                                   "entity value is ever copied into a memory slot.",
        },
        "validation": {
            "per_case": "rwkv_slots_data.validate([case]) before a case is accepted",
            "final": "rwkv_slots_data.validate(all cases) per output file",
            "integration": integration_by_file,
            "cases_dropped_in_final_validation": sum(dropped_reasons.values()),
            "stray_template_json_in_cwd": str(stray_template) if stray_template.is_file() else None,
            "loader_note": "all_cases(cases_path) is called with a single argument; "
                           "rwkv_slots_data rejects cases_path + template_path together",
        },
        "progress": result["progress"],
    }
    write_json(provenance_path, provenance)

    print(json.dumps({
        "ok": True,
        "train_file": {"path": str(out_path), "cases": per_file["train_file"]["cases"],
                       "by_group": per_file["train_file"]["by_group"],
                       "sha256": per_file["train_file"]["sha256"],
                       "data_version": per_file["train_file"]["data_version"]},
        "test_file": {"path": str(test_out_path), "cases": per_file["test_file"]["cases"],
                      "by_group": per_file["test_file"]["by_group"],
                      "sha256": per_file["test_file"]["sha256"],
                      "data_version": per_file["test_file"]["data_version"]},
        "verified": provenance["counts"]["verified"],
        "unverified": provenance["counts"]["unverified"],
        "value_stats": provenance["value_stats"],
        "skipped": provenance["skipped"],
        "contradiction_policy": provenance["contradiction_policy"],
        "integration": integration_by_file,
        "provenance_path": str(provenance_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvertError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
