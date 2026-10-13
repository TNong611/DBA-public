"""Independent verification of the RiSAWOZ memory datasets.

Deliberately does NOT import convert_risawoz_memory: it re-reads the source
corpus and the published artifacts, re-deriving expectations with its own code,
so a bug in the converter cannot hide behind the converter's helper functions.

Checked here
  1. shape: exactly the six contract fields, correct types, id <-> group agreement
  2. rwkv_slots_data.validate() accepts each file, and all_cases() loads it
  3. four slots, fixed order, topic <= 14 chars, summary <= 200 chars
  4. unverified results hedge; verified results carry a confirmation marker
  5. HISTORY SUPPORT: `user` is re-derived from the source dialogue as
     "every earlier turn (用户/系统) + the current user utterance" and compared
     line by line with the shipped text under the same redaction normalisation
  6. NO FUTURE TEXT: the last user line is the current turn's own utterance, and
     no line of the prefix can come from a later turn
  7. SOURCE SUPPORT OF 结果: an embedded factual value must be a value of that
     turn's own system act; a generic slot label must still name a slot the turn
     delivered with a value; unverified results must name a slot the turn only
     requested / did not deliver
  8. SOURCE SUPPORT OF 约束: every "沿用…" pair must be carried by a turn
     strictly earlier than the labelled turn, and every "本轮…" pair must come
     from the current turn -- the check that a carried claim is not invented
  9. observed-input uniqueness: no observation carries two labels in a file, no
     observation or memory text is shared between the training and test files
 10. independent PII scan over the whole `user` prefix, not just the last turn
 11. split contract: the training file holds train.json (group train) + dev.json
     (group heldout, the tuning/validation split) and the test file holds only
     test.json (group heldout, final evaluation); no case crosses those lines and
     no observation or memory text is shared between the two files
 12. provenance.json agrees with both files (sizes, hashes, counts, value stats)

Run: py -3.14 verify_risawoz_conversion.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The contract module lives beside this checker in the DBA repo; the fallback
# keeps the script runnable when it is copied out for a one-off audit.
for candidate in (HERE, Path(r"D:\AI\DBA\scripts")):
    if (candidate / "rwkv_slots_data.py").is_file():
        sys.path.insert(0, str(candidate))
        break
import rwkv_slots_data as DATA  # noqa: E402

BASE = Path(r"D:\AI\DBA\training\web-sources\risawoz")
TRAIN_CASES = BASE / "memory_cases.jsonl"
TEST_CASES = BASE / "memory_test.jsonl"
PROVENANCE = BASE / "provenance.json"

FIELDS = {"id", "group", "user", "assistant", "unverified", "summary"}
PLACEHOLDER = "[已脱敏]"

PII = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("url", re.compile(r"https?://")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("mobile", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("service_400", re.compile(r"(?<!\d)400[-\s]?\d{3}[-\s]?\d{4}(?!\d)")),
    ("landline", re.compile(r"(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)")),
    ("long_digits", re.compile(r"(?<!\d)\d{7,}(?!\d)")),
)
# Same shapes the converter redacts, rebuilt here from the documented list.
REDACTION_SHAPES = (
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"https?://\S+"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)400[-\s]?\d{3}[-\s]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)"),
    re.compile(r"(?<!\d)\d{7,}(?!\d)"),
)
DB_FIELD = re.compile(r"'([^']*)':\s*'([^']*)'")
DB_PII_MARKERS = ("电话", "地址", "手机", "邮箱", "邮件", "姓名", "身份证", "账号", "卡号")
DB_ENTITY_MARKERS = ("名称", "品牌", "主演", "片名", "店名", "公司名", "演出团体")

# Act types whose value counts as a delivery on their own (stated here
# independently of the converter, so a rule change shows up as a mismatch).
VALUED_ACT_TYPES = ("Inform", "Recommend", "No-offer")
QUESTION_TAIL = ("？", "?")

VALUE_RESULT = re.compile(r"^已确认(?:告知(?P<slot_i>.+?)|给出(?P<slot_r>.+?)推荐)：(?P<value>.+)$")
GENERIC_RESULT = re.compile(r"^已确认(?:告知(?P<slot_i>.+?)信息|给出(?P<slot_r>.+?)推荐)$")
NO_OFFER_RESULT = re.compile(r"^已确认无匹配(?:(?P<slot>.+?)的结果|结果)$")
SLOTLESS_RESULT = re.compile(r"^已确认回应本轮咨询$")
HEDGE_RESULT = re.compile(r"^(?:尚未取得(?P<slot>.+?)信息|尚未取得本轮(?:答复|结论)|本轮未确认新增信息)$")


def problem_kind(message: str) -> str:
    """Coarse bucket for a problem message, so the report can show a breakdown."""
    for marker, name in (
        ("history line mismatch", "history_line_mismatch"),
        ("history line count mismatch", "history_line_count_mismatch"),
        ("is not the current system turn", "assistant_mismatch"),
        ("is not a value of that turn's", "value_unsupported"),
        ("that the turn never delivered", "generic_slot_unsupported"),
        ("neither requested nor delivered", "hedge_slot_unsupported"),
        ("unrecognised result shape", "result_shape_unknown"),
        ("is not present in any earlier turn", "carried_constraint_unsupported"),
        ("is not in this turn's", "current_constraint_unsupported"),
        ("does not point at a source", "source_lookup_failed"),
        ("survived in", "pii_survived"),
        ("conflicting labels", "conflicting_labels"),
        ("isolation broken", "cross_file_leak"),
        ("is not domain+slot", "topic_not_domain_slot"),
        ("sha256 mismatch", "provenance_mismatch"),
        ("count mismatch", "provenance_mismatch"),
        ("data_version mismatch", "provenance_mismatch"),
        ("byte size mismatch", "provenance_mismatch"),
        ("value count", "provenance_value_mismatch"),
        ("validate rejected", "contract_rejected"),
        ("all_cases failed", "loader_failed"),
        ("speaker prefix", "missing_speaker_prefix"),
        ("field set mismatch", "shape"),
        ("summary is not four lines", "shape"),
    ):
        if marker in message:
            return name
    return "other"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(text) -> str:
    return " ".join(str(text or "").split())


def db_values(db_results):
    """Contact/address values and entity names of one turn, re-derived here."""
    pii, entity = [], []
    for entry in db_results or []:
        if not isinstance(entry, str):
            continue
        for key, value in DB_FIELD.findall(entry):
            value = clean(value)
            if len(value) < 4:
                continue
            if any(marker in key for marker in DB_PII_MARKERS):
                if value not in pii:
                    pii.append(value)
            elif any(marker in key for marker in DB_ENTITY_MARKERS):
                if value not in entity:
                    entity.append(value)
    return pii, entity


def normalize(text: str, values) -> str:
    """Reduce text to a comparable form, ignoring redaction and whitespace.

    Values are removed longest-first and PII shapes plus the placeholder are
    erased, so source text and shipped text become comparable without either
    side's redaction implementation being trusted. Whitespace is then removed
    entirely rather than collapsed: substituting a placeholder for a value and
    deleting the value leave different space runs around the gap, and the corpus
    itself carries irregular spaces inside slot values. Only content, order and
    speaker assignment are compared here.
    """
    for value in sorted({v for v in values if v}, key=len, reverse=True):
        text = text.replace(value, "")
    for pattern in REDACTION_SHAPES:
        text = pattern.sub("", text)
    text = text.replace(PLACEHOLDER, "")
    return re.sub(r"\s+", "", text)


def load_cases(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def turn_positions(dialogue):
    """Same view of a dialogue the converter works from: dict turns, in order."""
    return [t for t in (dialogue.get("dialogue") or []) if isinstance(t, dict)]


def pairs_of(text):
    out = []
    for chunk in text.replace("；", "、").split("、"):
        chunk = chunk.strip()
        if "=" in chunk:
            slot, _eq, value = chunk.partition("=")
            out.append((slot.strip(), value.strip()))
    return out


def main():  # noqa: C901 - a checker is allowed to be explicit
    problems = []
    stats = collections.Counter()

    train_cases = load_cases(TRAIN_CASES)
    test_cases = load_cases(TEST_CASES)
    files = {"train_file": (TRAIN_CASES, train_cases), "test_file": (TEST_CASES, test_cases)}
    stats["train_cases"] = len(train_cases)
    stats["test_cases"] = len(test_cases)

    # 1. shape
    for cases in (train_cases, test_cases):
        for case in cases:
            if set(case) != FIELDS:
                problems.append("field set mismatch in %s: %s" % (case.get("id"), sorted(case)))
            if not isinstance(case["unverified"], bool):
                problems.append("unverified is not a bool in " + str(case.get("id")))
            for key in ("id", "user", "assistant", "summary"):
                if not isinstance(case[key], str) or not case[key].strip():
                    problems.append("empty/!str %s in %s" % (key, case.get("id")))
            if case["group"] not in DATA.GROUPS:
                problems.append("bad group in %s" % case.get("id"))

    # 2. contract validation per file
    for key, (path, cases) in files.items():
        try:
            DATA.validate(cases)
        except ValueError as exc:
            problems.append("rwkv_slots_data.validate rejected %s: %s" % (key, exc))
        try:
            combined = DATA.all_cases(str(path))
            stats[key + "_integration"] = len(combined)
        except ValueError as exc:
            problems.append("all_cases failed for %s: %s" % (key, exc))

    # 3/4. slot semantics + split contract (source vocabulary rebuilt from corpus)
    vocabulary = {"domains": set(), "slots": set()}
    dialogues_by_split = {}
    for split, name in (("train", "train.json"), ("dev", "dev.json"), ("test", "test.json")):
        dialogues = json.loads((BASE / name).read_text(encoding="utf-8"))
        dialogues_by_split[split] = dialogues
        for dialogue in dialogues:
            if not isinstance(dialogue, dict):
                continue
            for domain in dialogue.get("domains") or []:
                vocabulary["domains"].add(str(domain))
            for turn in dialogue.get("dialogue") or []:
                for domain in turn.get("turn_domain") or []:
                    vocabulary["domains"].add(str(domain))
                for action in turn.get("system_actions") or []:
                    if isinstance(action, list) and len(action) > 2:
                        if action[1]:
                            vocabulary["domains"].add(str(action[1]))
                        if action[2]:
                            vocabulary["slots"].add(str(action[2]))
                belief = turn.get("belief_state") or {}
                for key in belief.get("turn request") or []:
                    vocabulary["slots"].add(str(key))
                for field in ("inform slot-values", "turn_inform"):
                    for key in (belief.get(field) or {}):
                        domain, _dash, slot = str(key).partition("-")
                        vocabulary["domains"].add(domain)
                        if slot:
                            vocabulary["slots"].add(slot)

    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    slot_alias = dict(provenance["redaction"]["slot_alias"])
    vocabulary["slots"].update(slot_alias.values())

    unmatched_topics = collections.Counter()
    # Group expected for each source tag, and the file each tag may appear in.
    group_of_tag = {"tr": (DATA.TRAIN, "train_file"),
                    "dv": (DATA.HELDOUT, "train_file"),
                    "te": (DATA.HELDOUT, "test_file")}
    for key, (_path, cases) in files.items():
        for case in cases:
            lines = case["summary"].split("\n")
            if len(lines) != 4:
                problems.append("summary is not four lines: " + case["id"])
                continue
            if [line.split("：", 1)[0] for line in lines] != list(DATA.SUMMARY_LABEL_ORDER):
                problems.append("slot order wrong in " + case["id"])
            topic = lines[0].split("：", 1)[1]
            if len(topic) > 14:
                problems.append("topic longer than 14 chars in " + case["id"])
            if len(case["summary"]) > 200:
                problems.append("summary longer than 200 chars in " + case["id"])
            tail = None
            for domain in sorted(vocabulary["domains"], key=len, reverse=True):
                if topic.startswith(domain):
                    tail = topic[len(domain):]
                    break
            if tail is None or not (tail == "咨询" or tail in vocabulary["slots"]):
                if topic not in vocabulary["slots"]:
                    unmatched_topics[topic] += 1
            result = lines[3].split("：", 1)[1]
            if case["unverified"]:
                stats["unverified"] += 1
                if not DATA.has_hedge(result):
                    problems.append("unverified result without hedge: " + case["id"])
            else:
                stats["verified"] += 1
                if not any(m in result for m in ("已确认", "已完成") + DATA._COMPLETION_WORDS):
                    problems.append("verified result without confirmation: " + case["id"])
            tag = case["id"].split("-")[1]
            expected = group_of_tag.get(tag)
            if expected is None:
                problems.append("case id has an unknown source tag: " + case["id"])
                continue
            want_group, want_file = expected
            if case["group"] != want_group:
                problems.append("%s (%s.json) must be group %s, found %s"
                                % (case["id"], tag, want_group, case["group"]))
            if key != want_file:
                problems.append("%s (%s.json) belongs in %s, found in %s"
                                % (case["id"], tag, want_file, key))
    if unmatched_topics:
        problems.append("%d topics are not domain+slot: %s"
                        % (sum(unmatched_topics.values()), unmatched_topics.most_common(5)))

    # 11. the training file must carry a real tuning split, test must stand alone
    train_groups = collections.Counter(c["group"] for c in train_cases)
    test_groups = collections.Counter(c["group"] for c in test_cases)
    if train_groups.get(DATA.TRAIN, 0) == 0:
        problems.append("training file has no group train cases")
    if train_groups.get(DATA.HELDOUT, 0) == 0:
        problems.append("training file has no validation (heldout/dev) cases: "
                        "tuning would have to reach for the test file")
    if set(test_groups) != {DATA.HELDOUT}:
        problems.append("test file must hold only heldout cases, found %s" % dict(test_groups))
    if any(c["id"].split("-")[1] == "te" for c in train_cases):
        problems.append("a test.json case leaked into the training file")
    if any(c["id"].split("-")[1] in ("tr", "dv") for c in test_cases):
        problems.append("a train/dev case leaked into the test file")

    # 5/6/7/8. history, no future text, source support
    split_of_tag = {"tr": "train", "dv": "dev", "te": "test"}
    for key, (_path, cases) in files.items():
        locale = {"checked": 0, "history_lines": 0, "value_supported": 0,
                  "generic_supported": 0, "hedge_supported": 0,
                  "carried_supported": 0, "act_type_noise_credited": 0,
                  "interrogative_value_act": 0}
        for case in cases:
            match = re.match(r"^RS-([a-z]{2})-(\d+)-(\d+)$", case["id"])
            if not match:
                problems.append("unexpected case id: " + case["id"])
                continue
            tag, index, turn_id = match.group(1), int(match.group(2)), int(match.group(3))
            split = split_of_tag.get(tag)
            if split is None or index >= len(dialogues_by_split[split]):
                problems.append("case %s does not point at a source dialogue" % case["id"])
                continue
            dialogue = dialogues_by_split[split][index]
            turns = turn_positions(dialogue)
            found = None
            for position, turn in enumerate(turns):
                if str(turn.get("turn_id")) == str(turn_id):
                    found = (position, turn)
                    break
            if found is None:
                problems.append("case %s does not point at a source turn" % case["id"])
                continue
            position, turn = found
            locale["checked"] += 1

            # 6. expected prefix: strictly earlier turns, then the current user turn.
            # Both sides are normalised line by line with the same value set -- every
            # db-derived value of the whole dialogue, since a redaction may draw on
            # any turn's values -- and the LINE LISTS are compared, so a content,
            # order or speaker difference cannot hide behind a joined blob.
            dialogue_values = []
            for other in turns:
                other_pii, other_entity = db_values(other.get("db_results"))
                dialogue_values.extend(other_pii + other_entity)
            expected_lines = []
            for earlier in turns[:position]:
                user_text = clean(earlier.get("user_utterance"))
                system_text = clean(earlier.get("system_utterance"))
                if user_text:
                    expected_lines.append("用户：" + user_text)
                if system_text:
                    expected_lines.append("系统：" + system_text)
            current_user = clean(turn.get("user_utterance"))
            expected_lines.append("用户：" + current_user)
            locale["history_lines"] += len(expected_lines)

            shipped_lines = []
            bad_prefix = False
            for line in case["user"].split("\n"):
                if line.startswith("用户：") or line.startswith("系统："):
                    shipped_lines.append(line)
                else:
                    problems.append("user line has no speaker prefix in " + case["id"])
                    bad_prefix = True
                    break
            if not bad_prefix:
                got = [normalize(line, dialogue_values) for line in shipped_lines]
                want = [normalize(line, dialogue_values) for line in expected_lines]
                if len(got) != len(want):
                    problems.append("history line count mismatch for %s: shipped %d, expected %d"
                                    % (case["id"], len(got), len(want)))
                else:
                    for line_index, (shipped_line, expected_line) in enumerate(zip(got, want)):
                        if shipped_line != expected_line:
                            problems.append("history line %d mismatch for %s:\n  shipped : %s\n"
                                            "  expected: %s"
                                            % (line_index, case["id"], shipped_line[:300],
                                               expected_line[:300]))
                            break
                # the reply must be this turn's own system utterance, not a later one
                if normalize(case["assistant"], dialogue_values) != normalize(
                        clean(turn.get("system_utterance")), dialogue_values):
                    problems.append("assistant is not the current system turn: " + case["id"])

            # 7. result support
            result = case["summary"].split("\n")[3].split("：", 1)[1]
            # A reply ending in ? asks the user for the slot instead of delivering
            # it, so a value it carries is an option, not a fact.
            question_reply = clean(turn.get("system_utterance")).endswith(QUESTION_TAIL)
            act_values = collections.defaultdict(list)   # slots delivered with a value
            act_slots = set()                            # every slot the turn engages
            valued_acts = 0
            value_acts_present = False
            valued_type_present = False
            for action in turn.get("system_actions") or []:
                if not isinstance(action, list) or len(action) < 3:
                    continue
                if clean(action[2]):
                    act_slots.add(slot_alias.get(clean(action[2]), clean(action[2])))
                if len(action) < 4 or not clean(action[3]):
                    continue
                value_acts_present = True
                act_type = clean(action[0])
                if act_type in VALUED_ACT_TYPES:
                    valued_type_present = True
                if act_type in VALUED_ACT_TYPES or not question_reply:
                    # A valued act counts even when it names no schema slot: the
                    # contract only requires the act type and a non-empty value.
                    valued_acts += 1
                    if clean(action[2]):
                        act_values[slot_alias.get(clean(action[2]), clean(action[2]))].append(
                            clean(action[3]))
            # The noisy-act rule is decisive only when no valued-type act exists,
            # which is exactly when the converter has to fall back to it.
            if value_acts_present and not valued_type_present:
                if question_reply:
                    locale["interrogative_value_act"] += 1
                else:
                    locale["act_type_noise_credited"] += 1
            request_slots = {slot_alias.get(clean(r), clean(r))
                             for r in (turn.get("belief_state") or {}).get("turn request") or []}
            value_match = VALUE_RESULT.match(result)
            generic_match = GENERIC_RESULT.match(result)
            no_offer_match = NO_OFFER_RESULT.match(result)
            hedge_match = HEDGE_RESULT.match(result)
            if value_match:
                slot = value_match.group("slot_i") or value_match.group("slot_r")
                value = value_match.group("value")
                if value not in act_values.get(slot, []):
                    problems.append("value %r in %s is not a value of that turn's %r act (has %r)"
                                    % (value, case["id"], slot, act_values.get(slot)))
                else:
                    locale["value_supported"] += 1
            elif no_offer_match:
                slot = no_offer_match.group("slot")
                if slot and slot not in act_slots:
                    problems.append("no-offer result names slot %r the turn has no act for: %s"
                                    % (slot, case["id"]))
                else:
                    locale["generic_supported"] += 1
            elif generic_match:
                slot = generic_match.group("slot_i") or generic_match.group("slot_r")
                if not act_values.get(slot):
                    problems.append("generic result names slot %r that the turn never delivered: %s"
                                    % (slot, case["id"]))
                else:
                    locale["generic_supported"] += 1
            elif SLOTLESS_RESULT.match(result):
                # The turn delivered something with a value but the act names no
                # usable schema slot; the label then claims only that this turn's
                # consult was answered, which requires a valued act to exist.
                if not valued_acts:
                    problems.append("slot-less result in %s claims an answer but the turn "
                                    "delivered no value" % case["id"])
            elif hedge_match:
                # "尚未取得X信息" means the turn did not obtain a value for slot X,
                # so X must be a slot this turn actually engaged: one of its acts
                # (a Request for X, or an act naming X without a value) or an entry
                # of its request state. Naming a slot the turn never engages would
                # be inventing a subject, and naming one it DID deliver a value for
                # would contradict the hedge itself.
                slot = hedge_match.group("slot")
                if slot and slot not in request_slots and slot not in act_slots:
                    problems.append("hedged result names slot %r the turn never touches: %s"
                                    % (slot, case["id"]))
                elif slot and slot in act_values:
                    problems.append("hedged result names slot %r the turn delivered a value for: %s"
                                    % (slot, case["id"]))
                else:
                    locale["hedge_supported"] += 1
            else:
                problems.append("unrecognised result shape in %s: %s" % (case["id"], result))

            # 8. constraint support: carried pairs must come from a strictly earlier turn
            constraint = case["summary"].split("\n")[1].split("：", 1)[1]
            earlier_beliefs = []
            for earlier in turns[:position]:
                belief = earlier.get("belief_state") or {}
                for field in ("inform slot-values", "turn_inform"):
                    for pair_key, pair_value in (belief.get(field) or {}).items():
                        _domain, _dash, slot = str(pair_key).partition("-")
                        earlier_beliefs.append((slot_alias.get(slot or str(pair_key),
                                                               slot or str(pair_key)),
                                                clean(pair_value)))
            current_belief = turn.get("belief_state") or {}
            current_pairs = []
            for pair_key, pair_value in (current_belief.get("turn_inform") or {}).items():
                _domain, _dash, slot = str(pair_key).partition("-")
                current_pairs.append((slot_alias.get(slot or str(pair_key),
                                                     slot or str(pair_key)), clean(pair_value)))

            carried_text = ""
            if constraint.startswith("沿用已告知条件："):
                carried_text = constraint[len("沿用已告知条件："):]
            elif "沿用：" in constraint:
                carried_text = constraint.split("沿用：", 1)[1]
            for pair in pairs_of(carried_text):
                if pair in earlier_beliefs:
                    locale["carried_supported"] += 1
                else:
                    problems.append("carried constraint %r in %s is not present in any earlier turn"
                                    % (pair, case["id"]))
            local_text = ""
            if constraint.startswith("本轮用户条件："):
                local_text = constraint[len("本轮用户条件："):]
            elif constraint.startswith("本轮条件："):
                local_text = constraint[len("本轮条件："):].split("；", 1)[0]
            for pair in pairs_of(local_text):
                if pair not in current_pairs:
                    problems.append("current-turn constraint %r in %s is not in this turn's "
                                    "turn_inform" % (pair, case["id"]))
        for name, value in locale.items():
            stats[key + "_" + name] = value

    # 9. uniqueness and cross-file isolation
    for key, (_path, cases) in files.items():
        by_pair = collections.defaultdict(set)
        for case in cases:
            by_pair[(case["user"], case["assistant"])].add(case["summary"])
        conflicting = [pair for pair, summaries in by_pair.items() if len(summaries) > 1]
        if conflicting:
            problems.append("%s has %d observations with conflicting labels"
                            % (key, len(conflicting)))
        stats[key + "_distinct_observations"] = len(by_pair)
    train_pairs = {(c["user"], c["assistant"]) for c in train_cases}
    test_pairs = {(c["user"], c["assistant"]) for c in test_cases}
    shared_pairs = train_pairs & test_pairs
    shared_summaries = ({c["summary"] for c in train_cases} & {c["summary"] for c in test_cases})
    if shared_pairs or shared_summaries:
        problems.append("train/test isolation broken: %d shared inputs, %d shared labels"
                        % (len(shared_pairs), len(shared_summaries)))

    # 10. independent PII scan over both files, whole prefix included
    found = collections.Counter()
    for cases in (train_cases, test_cases):
        for case in cases:
            for field in ("user", "assistant", "summary"):
                for name, pattern in PII:
                    hits = len(pattern.findall(case[field]))
                    if hits:
                        found[name] += hits
                        if len(problems) < 25:
                            problems.append("PII %s survived in %s/%s" % (name, case["id"], field))
    stats["pii_shapes_found"] = sum(found.values())

    # 10b. no case may carry a db contact value of its own turn
    cross_checked = 0
    for key in ("train_file", "test_file"):
        for case in (train_cases if key == "train_file" else test_cases):
            match = re.match(r"^RS-([a-z]{2})-(\d+)-(\d+)$", case["id"])
            if not match:
                continue
            split = split_of_tag.get(match.group(1))
            dialogue = dialogues_by_split[split][int(match.group(2))]
            for turn in turn_positions(dialogue):
                if str(turn.get("turn_id")) != str(int(match.group(3))):
                    continue
                pii, _entity = db_values(turn.get("db_results"))
                text = case["user"] + "\n" + case["assistant"] + "\n" + case["summary"]
                for value in pii:
                    if value in text:
                        problems.append("db contact value survived in " + case["id"])
                cross_checked += 1
                break
    stats["db_leak_checked"] = cross_checked

    # 11. provenance agreement
    for key, (path, cases) in files.items():
        entry = provenance["outputs"][key]
        if entry["sha256"] != sha256_file(path):
            problems.append("provenance sha256 mismatch for " + key)
        if entry["cases"] != len(cases):
            problems.append("provenance case count mismatch for " + key)
        if entry["data_version"] != DATA.data_version(cases):
            problems.append("provenance data_version mismatch for " + key)
        if entry["bytes"] != path.stat().st_size:
            problems.append("provenance byte size mismatch for " + key)
        if provenance["validation"]["integration"][key].get("ok") is not True:
            problems.append("provenance records a failed integration check for " + key)
    declared = provenance["value_stats"]
    counted_values = stats["train_file_value_supported"] + stats["test_file_value_supported"]
    if declared["verified_result_with_value"] != counted_values:
        problems.append("provenance value count %d != independently counted %d"
                        % (declared["verified_result_with_value"], counted_values))
    counted_noise = (stats["train_file_act_type_noise_credited"]
                     + stats["test_file_act_type_noise_credited"])
    if declared.get("act_type_noise_credited") != counted_noise:
        problems.append("provenance act_type_noise_credited %r != independently counted %d "
                        "(the delivery rule and the data disagree)"
                        % (declared.get("act_type_noise_credited"), counted_noise))
    counted_question_acts = (stats["train_file_interrogative_value_act"]
                             + stats["test_file_interrogative_value_act"])
    if declared.get("interrogative_value_act") != counted_question_acts:
        problems.append("provenance interrogative_value_act %r != independently counted %d"
                        % (declared.get("interrogative_value_act"), counted_question_acts))
    if "act_type_note" not in declared:
        problems.append("provenance does not document the noisy-act-type rule")
    if declared["unverified_result"] != stats["unverified"]:
        problems.append("provenance unverified count disagrees")
    if provenance["counts"]["verified"] + provenance["counts"]["unverified"] != len(train_cases) + len(test_cases):
        problems.append("provenance verified+unverified does not match the file sizes")

    # 11b. the declared split policy must match the data (doc <-> data agreement)
    policy = provenance["split_policy"]
    declared_train_groups = policy["train_file"]["groups"]
    declared_test_groups = policy["test_file"]["groups"]
    if declared_train_groups.get("dev.json") != DATA.HELDOUT:
        problems.append("provenance says dev.json is group %r; it must be %r (tuning validation)"
                        % (declared_train_groups.get("dev.json"), DATA.HELDOUT))
    if declared_train_groups.get("train.json") != DATA.TRAIN:
        problems.append("provenance says train.json is group %r; it must be %r"
                        % (declared_train_groups.get("train.json"), DATA.TRAIN))
    if declared_test_groups.get("test.json") != DATA.HELDOUT:
        problems.append("provenance says test.json is group %r; it must be %r"
                        % (declared_test_groups.get("test.json"), DATA.HELDOUT))
    if policy["train_file"]["sources"] != ["train", "dev"]:
        problems.append("provenance lists the wrong sources for the training file")
    if policy["test_file"]["sources"] != ["test"]:
        problems.append("provenance lists the wrong sources for the test file")
    if "tuning_split" not in policy or "test_is_final_only" not in policy:
        problems.append("provenance does not document the tuning/test separation")
    for name in ("train_file", "test_file"):
        if provenance["outputs"][name]["by_group"] != dict(
                collections.Counter(c["group"] for c in (train_cases if name == "train_file" else test_cases))):
            problems.append("provenance by_group disagrees for " + name)

    report = {
        "ok": not problems,
        "problem_kinds": dict(collections.Counter(problem_kind(p) for p in problems).most_common()),
        "train_file": {"cases": len(train_cases), "sha256": sha256_file(TRAIN_CASES),
                       "data_version": DATA.data_version(train_cases),
                       "by_group": dict(collections.Counter(c["group"] for c in train_cases)),
                       "by_source_split": dict(collections.Counter(c["id"].split("-")[1] for c in train_cases))},
        "test_file": {"cases": len(test_cases), "sha256": sha256_file(TEST_CASES),
                      "data_version": DATA.data_version(test_cases),
                      "by_group": dict(collections.Counter(c["group"] for c in test_cases)),
                      "by_source_split": dict(collections.Counter(c["id"].split("-")[1] for c in test_cases))},
        "verified": stats["verified"],
        "unverified": stats["unverified"],
        "value_supported": counted_values,
        "generic_slot_only": stats["train_file_generic_supported"] + stats["test_file_generic_supported"],
        "hedged_result_slots_supported": (stats["train_file_hedge_supported"]
                                          + stats["test_file_hedge_supported"]),
        "act_type_noise_credited": (stats["train_file_act_type_noise_credited"]
                                    + stats["test_file_act_type_noise_credited"]),
        "interrogative_value_act": (stats["train_file_interrogative_value_act"]
                                    + stats["test_file_interrogative_value_act"]),
        "carried_constraint_pairs_supported": (stats["train_file_carried_supported"]
                                               + stats["test_file_carried_supported"]),
        "history_checked": stats["train_file_checked"] + stats["test_file_checked"],
        "history_lines_checked": stats["train_file_history_lines"] + stats["test_file_history_lines"],
        "db_leak_checked": stats["db_leak_checked"],
        "pii_shapes_found": stats["pii_shapes_found"],
        "shared_train_test_inputs": len(shared_pairs),
        "shared_train_test_labels": len(shared_summaries),
        "topics_not_matching_domain_plus_slot": unmatched_topics.most_common(5),
        "problems": problems[:25],
        "problem_count": len(problems),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
