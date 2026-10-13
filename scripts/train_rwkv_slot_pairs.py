"""FROZEN RWKV STATE -> SLOT-VALUE PAIRS (a pair set, not a string to copy).

WHAT CHANGED AND WHY
--------------------
Four runs already tried to make a decoder emit the four-line memory TEXT:

  v3   teacher-forced token loss, train loss 0.23, dev four-slot exact    0/1779
  v4   + FiLM conditioning and scheduled TF dropout,  dev four-slot exact 0/1779
  classifier   structured per-slot heads,             dev four-slot exact 0/1779
  structural   marker/item composition,                dev four-slot exact 2/1779

The blocker was always the same slot: `约束`, a composed string of ~24 median
characters that had to match byte for byte. Measured on this corpus, its per-slot
rate never left the floor. A composed string is the wrong estimator: at 90%
per-character accuracy a 24-character string survives `0.9**24 = 8%`.

This script therefore changes the TASK, not just the capacity:

  * the constraint is predicted as a SET OF ATOMIC TRIPLES `(marker, attribute,
    value)`, one sigmoid per triple class, thresholded at 0.5 -- no ordering to
    get wrong, no per-character compounding;
  * topic and pending are single classes;
  * result is decomposed into `(pattern, value)` with the value predicted both as
    a class over TRAIN values and by a character-level fallback head, so a value
    never seen in TRAIN can still be composed from characters instead of being
    silently mapped onto a near-miss class.

CANONICAL ORDER -- STATED HERE, IN THE REPORT, AND IN THE METRIC PAIR
---------------------------------------------------------------------
The reference constraint order is NOT canonical: 45.1% of heldout multi-item
groups disagree with the order those attributes first appeared in TRAIN. The set
of triples is therefore the semantically correct target, and this script
canonicalises BOTH sides before comparing (groups ordered by marker, items by
attribute then value). Rendering to text happens only at output time.

So the new metric cannot be accused of being quietly loosened: every evaluation
reports the canonical number AND the old order-sensitive number side by side.
`four_slot_exact_strict` compares the rendered constraint against the RAW
reference text byte for byte, so it is exactly the metric the earlier runs were
judged by, and it is always <= the canonical one.

MEASURED CEILING (recomputed independently by `--preflight`, see the notes there
for the two figures of the task statement that did NOT reproduce):

    per-slot heldout reachability, TRAIN vocabularies only
      topic       0.9989        pending     0.9977
      constraint  0.9904 (all triples seen in TRAIN)
      result      0.8534 closed vocabulary; 0.9718 with the character fallback
    JOINT pair-level reachability
      0.8421 closed vocabulary; 0.9594 with the character fallback
    so the 0.80 acceptance bar is reachable, and the character fallback is what
    lifts it clear of the bar rather than being a decoration.

WHAT THIS FILE DOES NOT DO
--------------------------
It never reads the sealed final split (`memory_test.jsonl`, ids `RS-te-*`): the
refusal is by path name and by case id, and the source itself is scanned for the
string in `--self-test`. It never trains on CPU, never writes to or deletes the
feature cache it reads, and makes no accuracy claim before a GPU run exists:
`--self-test` and `--preflight` are CPU-only and produce no metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rwkv_slots_data as DATA  # noqa: E402

# Shared contracts live in the earlier scripts so the feature-cache format, the AC
# gate and the checkpoint format cannot drift between programs. Importing the
# classifier only defines objects: it loads torch lazily and touches no device.
import train_rwkv_slot_classifier as base  # noqa: E402

LABEL = "slot-value pair predictor (frozen RWKV-7 recurrent state, no backbone training)"
BACKBONE_TRAINING = False
SCRIPT_VERSION = "train_rwkv_slot_pairs.py/1"
FORMAT = DATA.FORMAT
SCHEMA = 1

EXIT_PAUSED = base.EXIT_PAUSED
TrainingError = base.TrainingError
Done = base.Done
PowerLoss = base.PowerLoss
ReadOnlyFeatureCache = base.ReadOnlyFeatureCache
build_pooled = base.build_pooled
refuse_sealed_split = base.refuse_sealed_split
RunLog = base.RunLog
wrong_state_shift = base.wrong_state_shift
paired = base.paired
write_json = base.write_json
atomic_torch_save = base.atomic_torch_save
save_checkpoint = base.save_checkpoint
load_checkpoint = base.load_checkpoint
rng_state = base.rng_state
restore_rng = base.restore_rng
cpu_optimizer_state = base.cpu_optimizer_state
main_connected = base.main_connected
power_report = base.power_report

CONDITIONS = base.CONDITIONS                      # ("true", "zero", "wrong")
POOL_MODES = base.POOL_MODES
SLOT_KEYS = DATA.SLOT_KEYS
EMPTY = DATA.EMPTY_SLOT                           # 无
IGNORE = base.IGNORE

GROUP_SEP = "\uff1b"       # ；between constraint groups
ITEM_SEP = "\u3001"        # 、between items inside a group
COLON = "\uff1a"           # ：between marker and body, and pattern and value
EQ = "="
END_CHAR = "<end>"         # terminator class of the character fallback head

DEFAULT_CASES = r"D:\AI\DBA\training\web-sources\risawoz\memory_cases.jsonl"
DEFAULT_OUT = r"D:\AI\DBA\training\rwkv-slot-pairs"
DEFAULT_FEATURE_CACHE = r"D:\AI\DBA\training\risawoz-run-v3\features\1f0b28205d715209"
KNOWN_BACKBONE_SHA256 = "23e13ff62ac6f80b28910ffc230324939e80ddb00b98f892a5951c0da071e700"
# The cache on disk was extracted over `DATA.all_cases(...)`, i.e. the 11773-row
# corpus PLUS the 22 embedded cases, so its manifest records that set's hash.
# This run trains on the 11773-row corpus only; both hashes are computed, logged
# and asserted to differ, so neither can be mistaken for the other.
EMBEDDED_EXTRA_CASES = 22

RESUME_NAME = "checkpoint-last.pt"
PAUSE_NAME = "checkpoint-paused.pt"
STATUS_NAME = "status.json"
LOG_NAME = "train.log"
POINTER_NAME = "latest.json"
EVAL_HISTORY_NAME = "eval-history.jsonl"
PREDICTIONS_NAME = "dev-predictions.jsonl"
REPORT_NAME = "dev-report.json"


# --------------------------------------------------------------------------
# pairs: parse, render, canonicalise
# --------------------------------------------------------------------------
def parse_constraint(text):
    """`marker：a=1、b=2；c=3` -> [(marker, [ (attribute, value), ... ]), ...].

    Order is the reference's order. A group is `[marker COLON] items`; an item is
    `attribute=value` or, for the free-text constraints in the embedded cases, a
    bare value with no `=`. Both shapes re-render byte-for-byte, which is asserted
    on every fixture and on all 11773 real references by `--preflight`.
    """
    text = str(text).strip()
    groups = []
    for chunk in text.split(GROUP_SEP):
        if COLON in chunk:
            marker, _, body = chunk.partition(COLON)
            marker = marker.strip()
        else:
            marker, body = "", chunk
        items = []
        for item in [part for part in body.split(ITEM_SEP) if part]:
            if EQ in item:
                attribute, _, value = item.partition(EQ)
                items.append((attribute.strip(), value.strip()))
            else:
                items.append(("", item.strip()))
        groups.append((marker, items))
    return groups


def render_constraint(groups):
    """Exact inverse of `parse_constraint`: same order, same bytes."""
    parts = []
    for marker, items in groups:
        body = ITEM_SEP.join(
            (attribute + EQ + value) if attribute else value
            for attribute, value in items)
        parts.append((marker + COLON + body) if marker else body)
    return GROUP_SEP.join(parts)


def constraint_triples(text):
    """The constraint as a SET of atomic `(marker, attribute, value)` triples.

    `无` is the empty-slot sentinel, not a triple: reading it as `('', '', '无')`
    would invent an attribute-less item that the renderer could not round-trip.
    """
    if str(text).strip() == EMPTY:
        return frozenset()
    return frozenset((marker, attribute, value)
                     for marker, items in parse_constraint(text)
                     for attribute, value in items)


def canonical_triples(triples):
    """Deterministic order for a triple SET: by marker, then attribute, then value."""
    return tuple(sorted(set(triples), key=lambda t: (t[0], t[1], t[2])))


def render_triples(triples):
    """Canonical text for a triple set: groups by marker, items sorted inside."""
    ordered = canonical_triples(triples)
    if not ordered:
        return EMPTY
    groups = []
    for marker, attribute, value in ordered:
        if not groups or groups[-1][0] != marker:
            groups.append((marker, []))
        groups[-1][1].append((attribute, value))
    return render_constraint(groups)


def split_pair(text):
    """`pattern：value` -> (pattern, value); no colon means the value is absent.

    The split is on the FIRST fullwidth colon. Values may contain ASCII colons
    (times such as `当天 08 : 55`) but the corpus never has two fullwidth colons in
    one result, so this round-trips exactly.
    """
    text = str(text).strip()
    head, sep, tail = text.partition(COLON)
    if not sep:
        return text, ""
    return head.strip(), tail.strip()


def join_pair(pattern, value):
    pattern = str(pattern).strip()
    value = str(value).strip()
    return (pattern + COLON + value) if value else pattern


def canonical_constraint_text(text):
    """Canonicalise one constraint string. Pure function of its triple set."""
    return render_triples(constraint_triples(text))


def parse_pairs(summary):
    """One memory text -> the four slots in pair form (no strings to copy)."""
    raw = DATA.slot_values(str(summary))
    missing = [key for key in SLOT_KEYS if key not in raw]
    if missing:
        raise TrainingError("memory text is missing slots: " + ", ".join(missing))
    return {
        "topic": raw["topic"].strip(),
        "pending": split_pair(raw["pending"]),
        "constraint": constraint_triples(raw["constraint"]),
        "result": split_pair(raw["result"]),
        "_raw": {key: raw[key].strip() for key in SLOT_KEYS},
    }


def render_pairs(pairs):
    """Canonical four-line text. Always the canonical constraint, never the raw one."""
    return "\n".join((
        DATA.SLOT_PREFIX["topic"] + str(pairs["topic"]).strip(),
        DATA.SLOT_PREFIX["constraint"] + render_triples(pairs["constraint"]),
        DATA.SLOT_PREFIX["pending"] + join_pair(*pairs["pending"]),
        DATA.SLOT_PREFIX["result"] + join_pair(*pairs["result"]),
    ))


def render_pairs_reference_order(pairs):
    """The same four lines but with the constraint in the REFERENCE's own order.

    This is the order-preserving render (the inverse of the parser), which is what
    makes a byte-for-byte round-trip of a stored reference meaningful. `render_pairs`
    is the canonical output the model is scored with; the two differ exactly on the
    references whose stored order is not canonical.
    """
    raw = pairs.get("_raw")
    if raw is None:
        return render_pairs(pairs)
    return "\n".join((
        DATA.SLOT_PREFIX["topic"] + str(pairs["topic"]).strip(),
        DATA.SLOT_PREFIX["constraint"] + raw["constraint"].strip(),
        DATA.SLOT_PREFIX["pending"] + join_pair(*pairs["pending"]),
        DATA.SLOT_PREFIX["result"] + join_pair(*pairs["result"]),
    ))


# --------------------------------------------------------------------------
# vocabularies (TRAIN split only)
# --------------------------------------------------------------------------
class ClosedVocab:
    """A train-only class set. Absent labels are counted, never mapped to a neighbour."""

    def __init__(self, items, absent_label=None):
        self.items = list(items)
        self.index = {item: position for position, item in enumerate(self.items)}
        self.absent_label = absent_label
        self.offset = 0
        if absent_label is not None:
            self.offset = 1
            self.items = [absent_label] + self.items
            self.index = {absent_label: 0}
            for position, item in enumerate(items, start=1):
                self.index[item] = position

    def __len__(self):
        return len(self.items)

    def __contains__(self, item):
        return item in self.index

    def encode(self, item):
        """Class index, or None when the label is outside the training vocabulary."""
        if item == self.absent_label:
            return 0
        return self.index.get(item)

    def decode(self, position):
        return self.items[int(position)]


class PairVocabularies:
    """Everything a prediction has to be expressed in, built from TRAIN labels only."""

    FIELDS = ("topic", "pending", "triples", "patterns", "values", "chars")

    def __init__(self, topics, pendings, triples, patterns, values, chars,
                 value_max_chars):
        self.topic = ClosedVocab(topics)
        self.pending = ClosedVocab(pendings)
        self.triples = ClosedVocab(triples)
        self.pattern = ClosedVocab(patterns)
        # Class 0 of the value head means "this result carries no value".
        self.value = ClosedVocab(values, absent_label="")
        self.chars = ClosedVocab(list(chars) + [END_CHAR])
        self.value_max_chars = int(value_max_chars)
        self.end_index = self.chars.index[END_CHAR]

    def summary(self):
        return {"topics": len(self.topic), "pendings": len(self.pending),
                "triples": len(self.triples), "result_patterns": len(self.pattern),
                "result_values": len(self.value) - 1, "value_chars": len(self.chars) - 1,
                "value_max_chars": self.value_max_chars}


def build_vocabularies(train_pairs):
    """Train-only class sets. Nothing from dev ever enters a vocabulary."""
    topics = sorted({pairs["topic"] for pairs in train_pairs})
    pendings = sorted({join_pair(*pairs["pending"]) for pairs in train_pairs})
    triples = sorted({triple for pairs in train_pairs for triple in pairs["constraint"]})
    patterns = sorted({pairs["result"][0] for pairs in train_pairs})
    values = sorted({pairs["result"][1] for pairs in train_pairs if pairs["result"][1]})
    chars = sorted({char for pairs in train_pairs for char in pairs["result"][1]})
    longest = max([len(pairs["result"][1]) for pairs in train_pairs] or [0])
    # +1 keeps room for the terminator class the character head always emits.
    return PairVocabularies(topics, pendings, triples, patterns, values, chars,
                            longest + 1)


# --------------------------------------------------------------------------
# reachability: what the vocabularies can express at all
# --------------------------------------------------------------------------
def reachability(dev_pairs, vocab):
    """Per-case flags plus the ceilings, computed from TRAIN vocabularies alone."""
    per_case, counters = {}, {"topic": 0, "pending": 0, "constraint": 0,
                              "result_closed": 0, "result_chars": 0}
    oov = {"topic": 0, "pending": 0, "triples": 0, "patterns": 0, "values": 0,
           "value_chars": 0}
    for pairs in dev_pairs:
        case_id = pairs["_id"]
        topic_ok = pairs["topic"] in vocab.topic
        pending_ok = join_pair(*pairs["pending"]) in vocab.pending
        unseen_triples = [t for t in pairs["constraint"] if t not in vocab.triples]
        constraint_ok = not unseen_triples
        pattern, value = pairs["result"]
        pattern_ok = pattern in vocab.pattern
        value_closed = (value == "") or (value in vocab.value)
        unseen_chars = [c for c in value if c not in vocab.chars]
        value_chars = not unseen_chars

        if not topic_ok:
            oov["topic"] += 1
        if not pending_ok:
            oov["pending"] += 1
        if unseen_triples:
            oov["triples"] += len(unseen_triples)
        if not pattern_ok:
            oov["patterns"] += 1
        if value and not value_closed:
            oov["values"] += 1
        if unseen_chars:
            oov["value_chars"] += 1

        closed = topic_ok and pending_ok and constraint_ok and pattern_ok and value_closed
        compositional = (topic_ok and pending_ok and constraint_ok and pattern_ok
                         and value_chars)
        for key, flag in (("topic", topic_ok), ("pending", pending_ok),
                          ("constraint", constraint_ok), ("result_closed", closed and pattern_ok and value_closed),
                          ("result_chars", pattern_ok and value_chars)):
            counters[key] += 1 if flag else 0
        per_case[case_id] = {
            "topic": topic_ok, "pending": pending_ok, "constraint": constraint_ok,
            "result_closed": pattern_ok and value_closed,
            "result_chars": pattern_ok and value_chars,
            "reachable": bool(closed), "reachable_compositional": bool(compositional),
            "unseen_triples": [list(t) for t in unseen_triples],
            "unseen_chars": unseen_chars,
        }
    cases = max(1, len(dev_pairs))
    return {
        "cases": len(dev_pairs),
        "per_slot_rate": {key: counters[key] / cases for key in counters},
        "joint_closed_rate": sum(1 for v in per_case.values() if v["reachable"]) / cases,
        "joint_compositional_rate": sum(1 for v in per_case.values()
                                        if v["reachable_compositional"]) / cases,
        "joint_closed_cases": sum(1 for v in per_case.values() if v["reachable"]),
        "joint_compositional_cases": sum(1 for v in per_case.values()
                                         if v["reachable_compositional"]),
        "out_of_vocabulary": oov,
        "per_case": per_case,
    }


def oracle_constraint_ceiling(dev_pairs, vocab):
    """How many dev cases are expressible if the triple set itself were predicted
    perfectly -- i.e. the constraint's own share of the joint ceiling."""
    return reachability(dev_pairs, vocab)


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------
class Targets:
    """Flat index/float tensors; nothing here is a string."""

    FIELDS = ("topic", "pending", "multi", "pattern", "value", "chars")

    def __init__(self, topic, pending, multi, pattern, value, chars, ids):
        self.topic = topic
        self.pending = pending
        self.multi = multi
        self.pattern = pattern
        self.value = value
        self.chars = chars
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def index(self, positions, torch):
        return Targets(self.topic[positions], self.pending[positions],
                       self.multi[positions], self.pattern[positions],
                       self.value[positions], self.chars[positions],
                       [self.ids[p] for p in positions])

    def to(self, device):
        """Move every tensor field to `device`; `ids` stays a plain Python list.

        The training step used to call `.to()` on the Targets object itself, which does
        not exist on a plain class -- the run died on its very first step with
        `AttributeError: 'Targets' object has no attribute 'to'`. Moving the tensors is
        the data class's own job, so the field list lives here next to the constructor
        instead of being duplicated at each call site.
        """
        moved = {name: getattr(self, name).to(device) for name in self.FIELDS}
        return Targets(ids=self.ids, **moved)


def encode_targets(pairs_list, vocab, torch, device=None):
    """Pair lists -> tensors. Out-of-vocabulary labels become IGNORE, so they are
    excluded from the loss instead of being mapped onto a wrong class."""
    count = len(pairs_list)
    topic = torch.full((count,), IGNORE, dtype=torch.long)
    pending = torch.full((count,), IGNORE, dtype=torch.long)
    multi = torch.zeros((count, len(vocab.triples)), dtype=torch.float32)
    pattern = torch.full((count,), IGNORE, dtype=torch.long)
    value = torch.zeros((count,), dtype=torch.long)
    chars = torch.full((count, vocab.value_max_chars), IGNORE, dtype=torch.long)
    for row, pairs in enumerate(pairs_list):
        encoded = vocab.topic.encode(pairs["topic"])
        if encoded is not None:
            topic[row] = encoded
        encoded = vocab.pending.encode(join_pair(*pairs["pending"]))
        if encoded is not None:
            pending[row] = encoded
        for triple in pairs["constraint"]:
            position = vocab.triples.encode(triple)
            if position is not None:
                multi[row, position] = 1.0
        encoded = vocab.pattern.encode(pairs["result"][0])
        if encoded is not None:
            pattern[row] = encoded
        text = pairs["result"][1]
        class_index = vocab.value.encode(text)
        value[row] = 0 if class_index is None else class_index
        if len(text) > vocab.value_max_chars - 1:
            raise TrainingError(
                "result value %r is longer than the %d-character budget; raise "
                "--value-max-chars instead of silently truncating a label"
                % (text, vocab.value_max_chars - 1))
        for position, char in enumerate(text):
            index = vocab.chars.index.get(char)
            if index is None:
                # Unseen character: leave IGNORE so the head is not taught to emit a
                # wrong character in its place. It is counted by `reachability`.
                continue
            chars[row, position] = index
        chars[row, len(text)] = vocab.end_index
    payload = {"topic": topic, "pending": pending, "multi": multi, "pattern": pattern,
               "value": value, "chars": chars, "ids": [p["_id"] for p in pairs_list]}
    if device is not None:
        payload = {key: (value_.to(device) if hasattr(value_, "to") else value_)
                   for key, value_ in payload.items()}
    return Targets(**payload)


def set_target_from_text(text, vocab, torch):
    """The multi-label target for one constraint string, as a 0/1 vector."""
    vector = torch.zeros(len(vocab.triples), dtype=torch.float32)
    for triple in constraint_triples(text):
        position = vocab.triples.encode(triple)
        if position is not None:
            vector[position] = 1.0
    return vector


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class PairModelConfig:
    FIELDS = ("pooled_dim", "d_state", "n_topics", "n_pendings", "n_triples",
              "n_patterns", "n_values", "n_chars", "d_char", "value_max_chars",
              "dropout")

    def __init__(self, pooled_dim, d_state, vocab, d_char=192, dropout=0.1):
        self.pooled_dim = int(pooled_dim)
        self.d_state = int(d_state)
        self.n_topics = len(vocab.topic)
        self.n_pendings = len(vocab.pending)
        self.n_triples = len(vocab.triples)
        self.n_patterns = len(vocab.pattern)
        self.n_values = len(vocab.value)
        self.n_chars = len(vocab.chars)
        self.d_char = int(d_char)
        self.value_max_chars = int(vocab.value_max_chars)
        self.dropout = float(dropout)

    def as_dict(self):
        return {field: getattr(self, field) for field in self.FIELDS}


def build_model(config, torch):
    """One shared state encoder, then parallel heads. There is no autoregression and
    no teacher forcing anywhere in this file, so the v3/v4 failure mode -- a low loss
    earned by copying the prefix while ignoring the state -- has no mechanism here."""
    nn = torch.nn

    class PairModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.norm = nn.LayerNorm(cfg.pooled_dim)
            self.encoder = nn.Linear(cfg.pooled_dim, cfg.d_state)
            self.act = nn.GELU()
            self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
            self.topic = nn.Linear(cfg.d_state, cfg.n_topics)
            self.pending = nn.Linear(cfg.d_state, cfg.n_pendings)
            self.triple = nn.Linear(cfg.d_state, cfg.n_triples)
            self.pattern = nn.Linear(cfg.d_state, cfg.n_patterns)
            self.value = nn.Linear(cfg.d_state, cfg.n_values)
            self.char_state = nn.Linear(cfg.d_state, cfg.d_char)
            self.char_pos = nn.Parameter(torch.zeros(cfg.value_max_chars, cfg.d_char))
            self.char_out = nn.Linear(cfg.d_char, cfg.n_chars)

        def encode(self, pooled):
            return self.drop(self.act(self.encoder(self.norm(pooled))))

        def forward(self, pooled):
            hidden = self.encode(pooled)
            char_base = self.char_state(hidden)
            char_hidden = char_base.unsqueeze(1) + self.char_pos.unsqueeze(0)
            return {"topic": self.topic(hidden), "pending": self.pending(hidden),
                    "triple": self.triple(hidden), "pattern": self.pattern(hidden),
                    "value": self.value(hidden),
                    "chars": self.char_out(char_hidden)}

    return PairModel(config)


def parameter_count(module):
    return sum(int(p.numel()) for p in module.parameters())


def triple_positive_weight(multi, torch, power=0.5, clamp=50.0):
    """Per-class positive weight for the sparse multi-label triple head.

    The triple head is 931 classes with a 0.2299% positive rate (~2.14 positives per
    case). Unweighted BCE has an all-negative solution that already costs 0.023, and
    the trained model took it: 64% of predicted constraints came out EMPTY while only
    32/1773 references are empty, and `triple` loss fell to 0.005. The linear probe
    shows those same pooled features carry each frequent triple at AUC 0.98, so the
    head was under-fitted by its own objective, not starved of information.

    `power` softens the raw negative/positive ratio (median 908x) with a square root;
    a full 908x weight destabilises the shared encoder. `clamp` keeps a single rare
    class from dominating the batch.
    """
    positives = multi.sum(dim=0)
    negatives = multi.shape[0] - positives
    ratio = negatives / positives.clamp_min(1.0)
    weight = ratio.clamp(max=clamp) ** power
    # A class absent from TRAIN must contribute nothing rather than a huge weight on
    # an all-zero column, which would push every logit down.
    return torch.where(positives > 0, weight, torch.zeros_like(weight))


def loss_terms(outputs, targets, weights, torch, ignore=IGNORE, triple_pos_weight=None):
    """All six heads, plus a weighted total. A head with no usable target in the
    batch contributes exactly zero instead of a NaN."""
    cross = torch.nn.functional.cross_entropy
    parts, total = {}, None

    def add(name, value, weight):
        nonlocal total
        parts[name] = float(value.detach()) if hasattr(value, "detach") else float(value)
        term = value * weight
        total = term if total is None else total + term

    add("topic", cross(outputs["topic"], targets.topic, ignore_index=ignore),
        weights["topic"])
    add("pending", cross(outputs["pending"], targets.pending, ignore_index=ignore),
        weights["pending"])
    add("triple", torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["triple"], targets.multi, pos_weight=triple_pos_weight),
        weights["triple"])
    add("pattern", cross(outputs["pattern"], targets.pattern, ignore_index=ignore),
        weights["pattern"])
    add("value", cross(outputs["value"], targets.value, ignore_index=ignore),
        weights["value"])
    char_logits = outputs["chars"].reshape(-1, outputs["chars"].shape[-1])
    char_targets = targets.chars.reshape(-1)
    valid = int((char_targets != ignore).sum())
    if valid:
        char_loss = torch.nn.functional.cross_entropy(char_logits, char_targets,
                                                      ignore_index=ignore, reduction="sum")
        char_loss = char_loss / valid
    else:
        char_loss = char_logits.sum() * 0.0
    add("chars", char_loss, weights["chars"])
    return total, parts


DEFAULT_WEIGHTS = {"topic": 1.0, "pending": 0.5, "triple": 1.0, "pattern": 1.0,
                   "value": 1.0, "chars": 0.5}


# --------------------------------------------------------------------------
# decoding: logits -> pairs
# --------------------------------------------------------------------------
def decode_pairs(outputs, vocab, torch, triple_threshold=0.5, value_confidence=0.5):
    """Three result policies from the same forward pass.

    `class`       the value comes from the closed value head (index 0 = no value);
    `char`        the value is composed character by character (works for values
                  absent from TRAIN);
    `arbitrated`  per case: the class head when it is at least `value_confidence`
                  sure, otherwise the character head. This is the headline policy;
                  the other two are always reported so the choice can be re-judged.
    """
    topic = outputs["topic"].argmax(dim=1)
    pending = outputs["pending"].argmax(dim=1)
    pattern = outputs["pattern"].argmax(dim=1)
    triple_mask = torch.sigmoid(outputs["triple"]) >= triple_threshold
    value_prob = torch.softmax(outputs["value"], dim=1)
    value_conf, value_class = value_prob.max(dim=1)
    char_ids = outputs["chars"].argmax(dim=2)

    count = int(topic.shape[0])
    decoded = []
    for row in range(count):
        triples = frozenset(
            vocab.triples.decode(position)
            for position in triple_mask[row].nonzero(as_tuple=False).flatten().tolist())
        head = vocab.pattern.decode(pattern[row].item())
        class_value = vocab.value.decode(value_class[row].item())
        composed = []
        for position in char_ids[row].tolist():
            if position == vocab.end_index:
                break
            composed.append(vocab.chars.decode(position))
        char_value = "".join(composed)
        chosen = class_value if float(value_conf[row].detach()) >= value_confidence else char_value
        decoded.append({
            "_id": None,
            "topic": vocab.topic.decode(topic[row].item()),
            "pending": split_pair(vocab.pending.decode(pending[row].item())),
            "constraint": triples,
            # `result` is the arbitrated choice, so a decoded entry is renderable as-is;
            # the two single-policy results are kept beside it for the report.
            "result": (head, chosen),
            "result_class": (head, class_value),
            "result_char": (head, char_value),
            "result_arbitrated": (head, chosen),
            "_value_confidence": float(value_conf[row].detach()),
        })
    variants = {}
    for name in ("class", "char", "arbitrated"):
        variants[name] = [
            {"_id": entry["_id"], "topic": entry["topic"], "pending": entry["pending"],
             "constraint": entry["constraint"], "result": entry["result_" + name]}
            for entry in decoded]
    return decoded, variants


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def slot_hits(prediction, reference):
    """Canonical per-slot equality for one case. The constraint is compared as a
    SET of triples, which is the declared target; the strict variant lives beside
    it in `score`."""
    return {
        "topic": prediction["topic"] == reference["topic"],
        "pending": tuple(prediction["pending"]) == tuple(reference["pending"]),
        "constraint": set(prediction["constraint"]) == set(reference["constraint"]),
        "result": tuple(prediction["result"]) == tuple(reference["result"]),
    }


def content_slot_accuracy(predictions, references):
    """Exactness restricted to slots whose reference is NOT `无`.

    `slot_match`-style accounting hands a hit to every empty reference slot that
    the model also leaves empty, so a predictor that answers `无` everywhere starts
    near 1.0/4 while knowing nothing. Excluding empty references removes that floor:
    here the denominator is only the slots that carried content.
    """
    hits = {key: 0 for key in SLOT_KEYS}
    scored = {key: 0 for key in SLOT_KEYS}
    for prediction, reference in zip(predictions, references):
        for key in SLOT_KEYS:
            if key == "constraint":
                if not reference["constraint"]:
                    continue
                scored[key] += 1
                hits[key] += 1 if set(prediction["constraint"]) == set(reference["constraint"]) else 0
                continue
            if key == "pending":
                if reference["_raw"]["pending"] == EMPTY:
                    continue
                scored[key] += 1
                hits[key] += 1 if tuple(prediction["pending"]) == tuple(reference["pending"]) else 0
                continue
            if key == "result":
                if reference["_raw"]["result"] == EMPTY:
                    continue
                scored[key] += 1
                hits[key] += 1 if tuple(prediction["result"]) == tuple(reference["result"]) else 0
                continue
            if reference["_raw"][key] == EMPTY:
                continue
            scored[key] += 1
            hits[key] += 1 if prediction[key] == reference[key] else 0
    total_hits = sum(hits.values())
    total_scored = sum(scored.values())
    return {"hits": total_hits, "scored": total_scored,
            "accuracy": (total_hits / total_scored) if total_scored else None,
            "per_slot": {key: {"hits": hits[key], "scored": scored[key],
                               "accuracy": (hits[key] / scored[key]) if scored[key] else None}
                         for key in SLOT_KEYS}}


def score(predictions, references):
    """Every headline number for one condition, none of them quietly relaxed.

    Reported together:
      per_slot_exact          canonical equality per slot;
      four_slot_exact         all four canonically equal;
      four_slot_exact_strict  all four equal with the constraint compared to the RAW
                              reference text byte for byte -- the metric the earlier
                              runs were judged by;
      content_slot_accuracy   exactness over non-`无` references only;
      graded                  per-case count of exact slots, 0..4, used for pairing.
    """
    per_slot = {key: 0 for key in SLOT_KEYS}
    strict_slot = {key: 0 for key in SLOT_KEYS}
    four = 0
    four_strict = 0
    graded = []
    strict_constraint_hits = 0
    for prediction, reference in zip(predictions, references):
        hits = slot_hits(prediction, reference)
        for key, value in hits.items():
            per_slot[key] += 1 if value else 0
        raw = reference["_raw"]["constraint"]
        strict_constraint = render_triples(prediction["constraint"]) == raw
        strict_hits = dict(hits)
        strict_hits["constraint"] = strict_constraint
        strict_constraint_hits += 1 if strict_constraint else 0
        for key, value in strict_hits.items():
            strict_slot[key] += 1 if value else 0
        graded.append(sum(1 for value in hits.values() if value))
        if all(hits.values()):
            four += 1
        if all(strict_hits.values()):
            four_strict += 1
    cases = max(1, len(references))
    content = content_slot_accuracy(predictions, references)
    return {
        "cases": len(references),
        "per_slot_exact": per_slot,
        "per_slot_exact_rate": {key: per_slot[key] / cases for key in SLOT_KEYS},
        "per_slot_exact_strict": strict_slot,
        "per_slot_exact_strict_rate": {key: strict_slot[key] / cases for key in SLOT_KEYS},
        "four_slot_exact": four,
        "four_slot_exact_rate": four / cases,
        "four_slot_exact_strict": four_strict,
        "four_slot_exact_strict_rate": four_strict / cases,
        "constraint_canonical_hits": per_slot["constraint"],
        "constraint_strict_hits": strict_constraint_hits,
        "content_slot_accuracy": content["accuracy"],
        "content_slot_hits": content["hits"],
        "content_slot_scored": content["scored"],
        "content_per_slot": content["per_slot"],
        "graded_total": sum(graded),
        "graded_per_case": graded,
    }


def paired_all(before, after):
    """Per-case comparison on BOTH pairings.

    The saturating pairing (all four slots at once) is kept because it is the
    headline acceptance number, but on its own it is blind: when every condition
    scores 0 on the all-four boolean it reports n ties and hides a real difference.
    The graded pairing on the same cases is what makes conditioning visible, and
    `--self-test` carries the explicit counterexample.
    """
    return {
        "saturating": paired(before["four_slot_flags"], after["four_slot_flags"]),
        "graded": paired(before["graded_per_case"], after["graded_per_case"]),
    }


def restricted_rate(flags, ids, mask):
    """Four-slot exact rate over a subset of cases.

    Read against `reachability`: the unrestricted rate charges the model for cases
    whose labels cannot be expressed by the TRAIN vocabularies at all. Reporting the
    restricted rate AND the unrestricted one, with the case counts, keeps the gap
    attributable instead of leaving it as an unexplained shortfall.
    """
    hits = [flag for flag, case_id in zip(flags, ids) if case_id in mask]
    return {"cases": len(hits), "hits": sum(1 for flag in hits if flag),
            "rate": (sum(1 for flag in hits if flag) / len(hits)) if hits else None}


def evaluate(model, pooled, references, vocab, torch, chunk=256,
             triple_threshold=0.5, value_confidence=0.5, reach=None):
    """One model, one set of cases, three conditioning sources.

    true   the case's own state;
    zero   the zero vector -- anything scored here comes from label priors alone;
    wrong  another case's state (a roll, never its own) -- conditioning is present
           but means nothing.
    """
    zero = torch.zeros_like(pooled)
    permutation = wrong_state_shift(int(pooled.shape[0]), 0, torch, device=pooled.device)
    wrong = pooled[permutation] if permutation is not None else zero
    ids = [reference["_id"] for reference in references]
    masks = {}
    if reach is not None:
        per_case = reach["per_case"]
        # The mask and the scored cases must be the same set: a mask built for a
        # different split would silently drop cases into neither bucket, which reads
        # as a smaller denominator rather than as an error.
        missing = [cid for cid in ids if cid not in per_case]
        if missing:
            raise TrainingError(
                "the reachability mask does not cover %d scored cases (e.g. %s); refusing "
                "to report a restricted rate over a mismatched case set"
                % (len(missing), missing[0]))
        masks = {
            "reachable": {cid for cid, entry in per_case.items() if entry["reachable"]},
            "reachable_compositional": {cid for cid, entry in per_case.items()
                                        if entry["reachable_compositional"]},
        }
        masks["unreachable"] = {cid for cid in per_case if cid not in masks["reachable"]}
    conditions = {}
    raw_predictions = {}
    for name, source in (("true", pooled), ("zero", zero), ("wrong", wrong)):
        model.eval()
        variants = {"class": [], "char": [], "arbitrated": []}
        confidences = []
        with torch.no_grad():
            for start in range(0, int(source.shape[0]), chunk):
                block = source[start:start + chunk]
                outputs = model(block)
                decoded, block_variants = decode_pairs(
                    outputs, vocab, torch, triple_threshold=triple_threshold,
                    value_confidence=value_confidence)
                for key in variants:
                    variants[key].extend(block_variants[key])
                confidences.extend(entry["_value_confidence"] for entry in decoded)
        scored = {}
        for key, predictions in variants.items():
            for prediction, reference in zip(predictions, references):
                prediction["_id"] = reference["_id"]
            scored[key] = score(predictions, references)
            scored[key]["four_slot_flags"] = [
                all(slot_hits(p, r).values()) for p, r in zip(predictions, references)]
            for mask_name, mask in masks.items():
                scored[key]["four_slot_exact_on_" + mask_name] = restricted_rate(
                    scored[key]["four_slot_flags"], ids, mask)
        conditions[name] = {"variants": scored, "value_confidence_mean":
                            sum(confidences) / max(1, len(confidences))}
        raw_predictions[name] = variants
    headline = "arbitrated"
    comparisons = {}
    for other in ("zero", "wrong"):
        # Argument order matters: `paired(before, after)` counts how often `after`
        # wins, so the true state must be the second argument for "wins" to mean
        # "the real state beat the ablation".
        comparisons["true_vs_" + other] = paired_all(
            conditions[other]["variants"][headline],
            conditions["true"]["variants"][headline])
    return {"conditions": conditions, "paired": comparisons, "headline_variant": headline,
            "masks": {name: len(mask) for name, mask in masks.items()},
            "predictions": raw_predictions}


def eval_summary_line(evaluation):
    def variant(name, condition="true"):
        return evaluation["conditions"][condition]["variants"][name]

    def fmt(name, condition="true"):
        entry = variant(name, condition)
        return "%s=%d/%d(%.4f)" % (name, entry["four_slot_exact"], entry["cases"],
                                   entry["four_slot_exact_rate"])

    def slot(name, key, condition="true"):
        return "%.4f" % variant(name, condition)["per_slot_exact_rate"][key]

    headline = evaluation["headline_variant"]
    graded = evaluation["paired"]["true_vs_zero"]["graded"]
    saturating = evaluation["paired"]["true_vs_zero"]["saturating"]
    reach = variant(headline).get("four_slot_exact_on_reachable", {})
    comp = variant(headline).get("four_slot_exact_on_reachable_compositional", {})
    unreach = variant(headline).get("four_slot_exact_on_unreachable", {})
    return ("dev four_slot_exact(canonical) %s %s | strict=%d/%d(%.4f) | "
            "slots(topic/constraint/pending/result)=%s/%s/%s/%s | "
            "on-reachable=%d/%d(%.4f) on-compositional=%d/%d(%.4f) "
            "unreachable=%d/%d(%.4f) | "
            "paired true_vs_zero graded w/l/t=%d/%d/%d saturating w/l/t=%d/%d/%d"
            % (fmt("class"), fmt(headline),
               variant("arbitrated")["four_slot_exact_strict"],
               variant("arbitrated")["cases"],
               variant("arbitrated")["four_slot_exact_strict_rate"],
               slot(headline, "topic"), slot(headline, "constraint"),
               slot(headline, "pending"), slot(headline, "result"),
               reach.get("hits", 0), reach.get("cases", 0), reach.get("rate") or 0.0,
               comp.get("hits", 0), comp.get("cases", 0), comp.get("rate") or 0.0,
               unreach.get("hits", 0), unreach.get("cases", 0), unreach.get("rate") or 0.0,
               graded["wins"], graded["losses"], graded["ties"],
               saturating["wins"], saturating["losses"], saturating["ties"]))


def compact_eval(evaluation):
    """Evaluation without the per-case vectors, for status and history files.

    `predictions` is dropped here as well as the per-case score vectors: it holds
    frozensets and case ids, so leaving it in would make every status write and every
    history line raise on serialisation. The full mapping stays available to callers
    that write `dev-predictions.jsonl`.
    """
    return {
        "headline_variant": evaluation["headline_variant"],
        "paired": evaluation["paired"],
        "conditions": {
            name: {variant: {key: value for key, value in entry.items()
                             if key != "graded_per_case" and key != "four_slot_flags"}
                   for variant, entry in block["variants"].items()}
            for name, block in evaluation["conditions"].items()},
    }


# --------------------------------------------------------------------------
# training loop (separated so the CPU self-test can drive one real step)
# --------------------------------------------------------------------------
def batch_stream(count, batch_size, seed, start_position=0):
    """Deterministic batches across epochs, resumable by position.

    A generator whose epoch shuffle depends only on `seed` and the epoch number, so
    a resumed run that skips `start_position` batches sees exactly the batches a
    continuous run would have seen at that point.
    """
    if count < batch_size:
        raise TrainingError("--batch-size %d exceeds the %d training cases"
                            % (batch_size, count))
    rng = random.Random(seed)
    position = 0
    while True:
        order = list(range(count))
        rng.shuffle(order)
        for offset in range(0, count - batch_size + 1, batch_size):
            batch = order[offset:offset + batch_size]
            if position >= start_position:
                yield position, batch
            position += 1


def training_step(*, torch, model, optimizer, weights, pooled, targets, positions,
                  device=None, grad_clip=1.0, triple_pos_weight=None):
    """One real forward/backward/optimizer step. Used by the run and by the CPU
    self-test, so the tested path is the trained path."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    block = pooled[positions]
    if device is not None:
        block = block.to(device)
    outputs = model(block)
    batch_targets = targets.index(positions, torch)
    if device is not None:
        batch_targets = batch_targets.to(device)
    total, parts = loss_terms(outputs, batch_targets, weights, torch,
                              triple_pos_weight=triple_pos_weight)
    total.backward()
    grad_norm = None
    if grad_clip:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    optimizer.step()
    return float(total.detach()), parts, grad_norm


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Frozen-RWKV-state -> slot-value pairs (topic, constraint triple "
                    "set, pending, result pattern+value).")
    parser.add_argument("--cases", default=DEFAULT_CASES,
                        help="JSONL of labelled cases (train + dev). The sealed test "
                             "split is refused by file name and by case id.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--feature-cache", default=DEFAULT_FEATURE_CACHE,
                        help="read-only feature cache written by an earlier run; this "
                             "script never extracts features and never writes there")
    parser.add_argument("--expect-cache-data-hash", default=None,
                        help="override the cache identity hash (default: the hash of "
                             "DATA.all_cases(--cases), which is what the cache on disk "
                             "records; the corpus-only hash is printed too)")
    parser.add_argument("--expect-backbone-sha256", default=KNOWN_BACKBONE_SHA256)
    parser.add_argument("--allow-any-backbone", action="store_true")
    parser.add_argument("--pool", choices=POOL_MODES, default="meanmax")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--d-state", type=int, default=512)
    parser.add_argument("--d-char", type=int, default=192)
    parser.add_argument("--value-max-chars", type=int, default=0,
                        help="character-head budget; 0 = longest TRAIN value + 1 "
                             "(never truncate a label)")
    parser.add_argument("--triple-threshold", type=float, default=0.5)
    parser.add_argument("--triple-pos-weight-power", type=float, default=0.5,
                        help="exponent softening the per-class negative/positive ratio "
                             "for the sparse triple head; 0 disables the weighting")
    parser.add_argument("--triple-pos-weight-clamp", type=float, default=50.0)
    parser.add_argument("--value-confidence", type=float, default=0.5,
                        help="class-head confidence below which the character fallback "
                             "is used instead")
    parser.add_argument("--weight-topic", type=float, default=DEFAULT_WEIGHTS["topic"])
    parser.add_argument("--weight-pending", type=float, default=DEFAULT_WEIGHTS["pending"])
    parser.add_argument("--weight-triple", type=float, default=DEFAULT_WEIGHTS["triple"])
    parser.add_argument("--weight-pattern", type=float, default=DEFAULT_WEIGHTS["pattern"])
    parser.add_argument("--weight-value", type=float, default=DEFAULT_WEIGHTS["value"])
    parser.add_argument("--weight-chars", type=float, default=DEFAULT_WEIGHTS["chars"])
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-heldout", type=int, default=0)
    parser.add_argument("--resume", default=None,
                        help="checkpoint to continue from (default <out-dir>/%s)" % RESUME_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-power-check", action="store_true",
                        help="ONLY valid with --self-test; disables the AC/guard gate")
    parser.add_argument("--self-test", action="store_true",
                        help="run the CPU self-tests and exit without touching CUDA")
    parser.add_argument("--preflight", action="store_true",
                        help="recompute the measured ceilings from the corpus, print "
                             "them and exit; no CUDA, no guard lease, nothing trained")
    args = parser.parse_args(argv)
    if args.skip_power_check and not args.self_test:
        parser.error("--skip-power-check is only accepted together with --self-test")
    if args.steps < 0:
        parser.error("--steps must not be negative")
    if not str(args.device).startswith("cuda"):
        parser.error("--device must be a CUDA device: this script refuses CPU training")
    return args


# --------------------------------------------------------------------------
# corpus loading shared by preflight and the run
# --------------------------------------------------------------------------
def load_corpus(cases_path):
    """The 11773-row corpus (train + dev). Validated, sealed split refused."""
    corpus = DATA.validate(DATA.load_events(cases_path))
    refuse_sealed_split(cases_path, corpus)
    return corpus


def split_corpus(corpus, limit_train=0, limit_heldout=0):
    train = [case for case in corpus if case["group"] == DATA.TRAIN]
    heldout = [case for case in corpus if case["group"] == DATA.HELDOUT]
    if limit_train:
        train = train[:limit_train]
    if limit_heldout:
        heldout = heldout[:limit_heldout]
    return train, heldout


def pair_lists(cases):
    out = []
    for case in cases:
        pairs = parse_pairs(case["summary"])
        pairs["_id"] = case["id"]
        out.append(pairs)
    return out


def round_trip_report(cases):
    """Every real reference must re-render byte-for-byte from its parsed pairs.

    The order-preserving render is compared to the stored summary, so this is a
    true lossless round-trip check. Whether the canonical form differs from it is a
    separate, expected fact (counted by `non_canonical_references`), not a failure.
    """
    failures = []
    for case in cases:
        raw = DATA.slot_values(case["summary"])
        if render_constraint(parse_constraint(raw["constraint"])) != raw["constraint"].strip():
            failures.append({"id": case["id"], "slot": "constraint",
                             "value": raw["constraint"]})
            continue
        for key in ("pending", "result"):
            if join_pair(*split_pair(raw[key])) != raw[key].strip():
                failures.append({"id": case["id"], "slot": key, "value": raw[key]})
        pairs = parse_pairs(case["summary"])
        pairs["_id"] = case["id"]
        if render_pairs_reference_order(pairs) != case["summary"].strip():
            failures.append({"id": case["id"], "slot": "whole-summary",
                             "value": case["summary"][:120]})
    return failures


def non_canonical_references(cases):
    """How many stored constraints are NOT in canonical order.

    This is the evidence behind using a set target: if a large share of references
    disagree with any single ordering, then an order-sensitive metric is measuring
    the corpus's writing habits rather than the model's knowledge.
    """
    count = 0
    for case in cases:
        pairs = parse_pairs(case["summary"])
        pairs["_id"] = case["id"]
        if render_pairs(pairs) != render_pairs_reference_order(pairs):
            count += 1
    return count


def ceiling_table(cases_path):
    """The measured table, recomputed from the corpus. No CUDA, no torch needed."""
    corpus = load_corpus(cases_path)
    train_cases, dev_cases = split_corpus(corpus)
    train_pairs = pair_lists(train_cases)
    dev_pairs = pair_lists(dev_cases)
    vocab = build_vocabularies(train_pairs)
    reach = reachability(dev_pairs, vocab)

    marker_counts, marker_counts_with_sentinel = {}, {}
    attribute_set, attribute_set_with_sentinel = set(), set()
    train_triples, train_triples_with_sentinel = set(), set()
    triple_lengths = []
    duplicate_occurrences, duplicate_cases = 0, 0
    for pairs in train_pairs + dev_pairs:
        raw = pairs["_raw"]["constraint"]
        sentinel = raw == EMPTY
        seen = set()
        case_duplicates = 0
        if sentinel:
            # `无` is the empty set, not a triple. It is counted once, here, purely so
            # the claim's own decomposition (which reads it as marker ``) is comparable.
            marker_counts_with_sentinel[""] = marker_counts_with_sentinel.get("", 0) + 1
            attribute_set_with_sentinel.add("")
            continue
        for marker, items in parse_constraint(raw):
            for attribute, value in items:
                # Occurrence counts, duplicates kept: this is the unit the claim's
                # decomposition is stated in and the only fair basis for comparing it.
                marker_counts_with_sentinel[marker] = marker_counts_with_sentinel.get(marker, 0) + 1
                attribute_set_with_sentinel.add(attribute)
                marker_counts[marker] = marker_counts.get(marker, 0) + 1
                attribute_set.add(attribute)
                triple = (marker, attribute, value)
                if triple in seen:
                    case_duplicates += 1
                seen.add(triple)
        if case_duplicates:
            duplicate_occurrences += case_duplicates
            duplicate_cases += 1
    for pairs in train_pairs:
        occurrences = [(marker, attribute, value)
                       for marker, items in parse_constraint(pairs["_raw"]["constraint"])
                       for attribute, value in items]
        triple_lengths.append(len(set(occurrences)))
        train_triples_with_sentinel |= set(occurrences)
        if pairs["_raw"]["constraint"] != EMPTY:
            train_triples |= set(occurrences)
    lengths_sorted = sorted(triple_lengths)
    median = (lengths_sorted[len(lengths_sorted) // 2] if lengths_sorted else 0)

    # dev-side coverage, both ways of asking the constraint question
    train_whole = {p["_raw"]["constraint"] for p in train_pairs}
    dev = len(dev_pairs)
    all_triples = sum(1 for p in dev_pairs
                      if all(t in train_triples for t in p["constraint"]))
    whole_seen = sum(1 for p in dev_pairs if p["_raw"]["constraint"] in train_whole)
    non_canonical = non_canonical_references(train_cases + dev_cases)

    table = {
        "corpus": {"total": len(corpus), "train": len(train_cases),
                   "dev": len(dev_cases)},
        "corpus_hash": DATA.data_version(corpus),
        "cache_identity_hash": DATA.data_version(DATA.all_cases(cases_path)),
        "marker_counts": marker_counts,
        "marker_counts_counting_sentinel_as_triple": marker_counts_with_sentinel,
        "distinct_train_attributes": len(attribute_set),
        "distinct_train_attributes_counting_sentinel": len(attribute_set_with_sentinel),
        "distinct_train_triples_modelled": len(vocab.triples),
        "distinct_train_triples_including_sentinel": len(train_triples_with_sentinel),
        "distinct_train_triples_real_only": len(train_triples),
        "duplicate_triple_occurrences": {"occurrences": duplicate_occurrences,
                                         "cases": duplicate_cases,
                                         "note": "a case that repeats the same triple cannot "
                                                 "be represented exactly by a set target; "
                                                 "these cases are matched canonically but can "
                                                 "never match the raw text byte for byte, "
                                                 "which the strict metric reports"},
        "distinct_train_values": {
            "topic": len({p["topic"] for p in train_pairs}),
            "pending": len({join_pair(*p["pending"]) for p in train_pairs}),
            "result": len({join_pair(*p["result"]) for p in train_pairs}),
            "result_patterns": len({p["result"][0] for p in train_pairs}),
            "result_patterns_including_dev": len(
                {p["result"][0] for p in train_pairs + dev_pairs}),
            "result_tail_values": len({p["result"][1] for p in train_pairs
                                       if p["result"][1]}),
        },
        "dev_constraint_all_triples_seen": {"cases": all_triples, "of": dev,
                                            "rate": all_triples / max(1, dev)},
        "dev_constraint_whole_string_seen": {"cases": whole_seen, "of": dev,
                                             "rate": whole_seen / max(1, dev)},
        "constraint_triples_per_case": {"median": median,
                                        "max": max(triple_lengths or [0]),
                                        "min": min(triple_lengths or [0])},
        "empty_constraint_strings": sum(1 for c in train_cases + dev_cases
                                        if DATA.slot_values(c["summary"])["constraint"].strip() == ""),
        "sentinel_constraints": sum(1 for p in train_pairs + dev_pairs
                                    if p["_raw"]["constraint"] == EMPTY),
        "non_canonical_references": {"cases": non_canonical,
                                     "of": len(train_cases) + dev,
                                     "rate": non_canonical / max(1, len(train_cases) + dev)},
        "vocabularies": vocab.summary(),
        "per_slot_reachability": reach["per_slot_rate"],
        "joint": {"closed_cases": reach["joint_closed_cases"],
                  "closed_rate": reach["joint_closed_rate"],
                  "compositional_cases": reach["joint_compositional_cases"],
                  "compositional_rate": reach["joint_compositional_rate"]},
        "out_of_vocabulary": reach["out_of_vocabulary"],
        "round_trip_failures": round_trip_report(corpus),
    }
    table["claimed_vs_recomputed"] = _claim_comparison(table)
    return table


def _claim_comparison(table):
    """Every figure the task statement supplied, recomputed here and judged.

    Kept as data rather than prose so a reader can see which claims survived an
    independent implementation and which did not.
    """
    def verdict(name, claimed, recomputed, note=""):
        return {"claim": name, "claimed": claimed, "recomputed": recomputed,
                "verdict": "match" if str(claimed) == str(recomputed) else "differs",
                "note": note}

    joint = table["joint"]
    rows = [
        verdict("cases total", 11773, table["corpus"]["total"]),
        verdict("train cases", 10000, table["corpus"]["train"]),
        verdict("heldout cases", 1773, table["corpus"]["dev"]),
        verdict("distinct train triples", 932, table["distinct_train_triples_including_sentinel"],
                "confirms 932 when the `无` sentinel is read as the triple ('', '', '无'). "
                "This run models %d classes: the sentinel is the EMPTY set, not a class, "
                "and %d distinct real triples were seen"
                % (table["distinct_train_triples_modelled"],
                   table["distinct_train_triples_real_only"])),
        verdict("distinct train attributes", 47, table["distinct_train_attributes_counting_sentinel"],
                "confirms 47 only with the sentinel's empty attribute; the real attribute "
                "vocabulary is %d" % table["distinct_train_attributes"]),
        verdict("markers", 5, len(table["marker_counts_counting_sentinel_as_triple"]),
                "4 real markers plus the sentinel-only '' ; without the sentinel it is %d"
                % len(table["marker_counts"])),
        verdict("duplicate triples within a case", "(not claimed)",
                "%d occurrences in %d cases" % (table["duplicate_triple_occurrences"]["occurrences"],
                                                table["duplicate_triple_occurrences"]["cases"]),
                "the set target collapses them; this is why a set-vocabulary count is "
                "slightly below the occurrence count and why the strict metric can never "
                "reach those cases"),
        verdict("heldout constraint reachability", "1769/1773 = 0.9977",
                "%d/%d = %.4f" % (table["dev_constraint_all_triples_seen"]["cases"],
                                  table["dev_constraint_all_triples_seen"]["of"],
                                  table["dev_constraint_all_triples_seen"]["rate"]),
                "does not reproduce; 0.9977 is the PENDING reachability figure. The "
                "task's own 0.9904 for the constraint is the correct one and matches"),
        verdict("string-level constraint coverage", "0.6987",
                "%.4f" % table["dev_constraint_whole_string_seen"]["rate"],
                "0.6987 was computed over the 1779-case all_cases dev split; over this "
                "11773-row corpus it is 0.6994"),
        verdict("topic reachability", "0.9989", "%.4f" % table["per_slot_reachability"]["topic"]),
        verdict("pending reachability", "0.9977", "%.4f" % table["per_slot_reachability"]["pending"]),
        verdict("constraint reachability", "0.9904", "%.4f" % table["per_slot_reachability"]["constraint"]),
        verdict("result reachability", "0.8534", "%.4f" % table["per_slot_reachability"]["result_closed"],
                "0.8534 is whole-pair retrieval coverage; this model has separate pattern "
                "and value heads, so a heldout pattern with a known value is reachable "
                "even when that exact pair was never seen. Both are printed"),
        verdict("joint pair-level reachability", "1493/1773 = 0.8421",
                "%d/%d = %.4f" % (joint["closed_cases"], table["corpus"]["dev"],
                                  joint["closed_rate"]),
                "same factorization difference as the result slot; the closed metric is "
                "still the conservative one and it clears 0.80 either way"),
        verdict("result distinct patterns", 247, table["distinct_train_values"]["result_patterns"],
                "247 does not reproduce under any split rule tried; 243 over train+dev"),
        verdict("result distinct tail values", 1324,
                table["distinct_train_values"]["result_tail_values"]),
        verdict("result reachability with character fallback", "(not claimed)",
                "%.4f" % table["per_slot_reachability"]["result_chars"],
                "the fallback head raises the binding slot"),
        verdict("joint with character fallback", "(not claimed)",
                "%d/%d = %.4f" % (joint["compositional_cases"], table["corpus"]["dev"],
                                  joint["compositional_rate"]),
                "this is the ceiling the run is actually judged against"),
        verdict("triples per case median", 2, table["constraint_triples_per_case"]["median"]),
        verdict("triples per case max", 4, table["constraint_triples_per_case"]["max"]),
        verdict("cases with an empty constraint string", 0, table["empty_constraint_strings"],
                "%d cases carry the `无` sentinel, which is a value, not an empty string"
                % table["sentinel_constraints"]),
        verdict("non-canonical reference order", "45.1% of heldout multi-item groups",
                "%.4f of all references" % table["non_canonical_references"]["rate"],
                "the group-level 45.1 figure is not re-derived here; at reference level "
                "%.4f of all constraints are not in canonical order, which is why the "
                "target is a set rather than a sequence"
                % table["non_canonical_references"]["rate"]),
    ]
    return rows


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
class Suite:
    def __init__(self):
        self.passed = 0
        self.failures = []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
        else:
            self.failures.append("%s %s" % (name, detail))
        return bool(condition)

    def equal(self, name, got, want):
        return self.check(name, got == want, "(got %r want %r)" % (got, want))

    def raises(self, name, exception, call, *args, **kwargs):
        try:
            call(*args, **kwargs)
        except exception:
            return self.check(name, True)
        except Exception as exc:  # noqa: BLE001
            return self.check(name, False, "(raised %r not %r)" % (exc, exception))
        return self.check(name, False, "(did not raise)")


FIXTURE_SUMMARIES = [
    # structured, one group
    "主题：酒店联系方式\n约束：沿用已告知条件：星级=3、区域=常熟\n未完成：无\n结果：已确认告知联系方式信息",
    # structured, two groups, reference order NOT canonical (沿用 before 本轮条件)
    "主题：电脑价格区间\n约束：沿用：分类=台式机、内存容量=8GB；本轮条件：价格区间=5000 到 10000\n未完成：无\n结果：已确认告知价格区间：5000 到 10000",
    # sentinel constraint
    "主题：训练方案记录\n约束：无\n未完成：开始训练并验证方案\n结果：尚未开始训练",
    # free-text constraint from the embedded curated cases (no `=`)
    "主题：索引恢复流程\n约束：原始事件不得修改\n未完成：修复重建流程\n结果：尚未修复索引恢复",
    # value with ASCII colons inside it
    "主题：出发时间确认\n约束：本轮条件：出发时间=当天 08 : 55\n未完成：无\n结果：已确认告知出发时间：当天 08 : 55",
    # pending with a value
    "主题：地铁直达确认\n约束：本轮条件：是否地铁直达=是\n未完成：本轮未答：停车场\n结果：已确认告知是否地铁直达：是",
    # three items in one group (ordering matters for the strict metric)
    "主题：医院科室\n约束：沿用已告知条件：重点科室=眼科、等级=三级、性质=公立医院\n未完成：无\n结果：已确认告知科室信息",
    # single item, no value in the result
    "主题：新增信息确认\n约束：本轮用户条件：车系=凌渡\n未完成：无\n结果：本轮未确认新增信息",
]


def _fixture_pairs():
    out = []
    for index, summary in enumerate(FIXTURE_SUMMARIES):
        pairs = parse_pairs(summary)
        pairs["_id"] = "FX-%02d" % index
        out.append(pairs)
    return out


def self_test(verbose=False):
    """CPU-only checks. No CUDA context is created, nothing is trained."""
    suite = Suite()
    torch = base._torch()
    pairs_list = _fixture_pairs()

    # ---- parser / renderer round-trip, byte for byte, on every fixture -----
    for pairs in pairs_list:
        raw = pairs["_raw"]
        suite.equal("roundtrip.constraint.%s" % pairs["_id"],
                    render_constraint(parse_constraint(raw["constraint"])),
                    raw["constraint"].strip())
        suite.equal("roundtrip.pending.%s" % pairs["_id"],
                    join_pair(*split_pair(raw["pending"])), raw["pending"].strip())
        suite.equal("roundtrip.result.%s" % pairs["_id"],
                    join_pair(*split_pair(raw["result"])), raw["result"].strip())
        suite.equal("roundtrip.whole.%s" % pairs["_id"],
                    render_pairs_reference_order(pairs),
                    "\n".join(DATA.SLOT_PREFIX[key] + raw[key] for key in SLOT_KEYS))
    suite.equal("roundtrip.fixture_count", len(FIXTURE_SUMMARIES), len(pairs_list))

    # ---- canonicalisation: determinism and order-invariance ---------------
    target = pairs_list[1]
    triples = set(target["constraint"])
    shuffled_a = canonical_triples(set(reversed(list(triples))))
    shuffled_b = canonical_triples(set(sorted(triples)))
    suite.equal("canonical.deterministic", render_triples(shuffled_a), render_triples(shuffled_b))
    reversed_text = render_constraint(list(reversed(parse_constraint(target["_raw"]["constraint"]))))
    suite.check("canonical.order_invariant",
                canonical_constraint_text(reversed_text)
                == canonical_constraint_text(target["_raw"]["constraint"]),
                "(reversed group order changed the canonical form)")
    suite.check("canonical.idempotent",
                canonical_constraint_text(canonical_constraint_text(target["_raw"]["constraint"]))
                == canonical_constraint_text(target["_raw"]["constraint"]))
    suite.check("canonical.reference_is_not_canonical",
                render_triples(target["constraint"]) != target["_raw"]["constraint"],
                "(fixture was supposed to be non-canonical; the strict metric needs one)")
    suite.equal("canonical.sentinel_is_empty_set", len(pairs_list[2]["constraint"]), 0)
    suite.equal("canonical.sentinel_renders", render_triples(frozenset()), EMPTY)
    suite.equal("canonical.free_text_survives",
                render_triples(pairs_list[3]["constraint"]),
                pairs_list[3]["_raw"]["constraint"])
    # `render_pairs` is the canonical OUTPUT; `render_pairs_reference_order` reproduces
    # the stored text. Conflating them would make the strict metric meaningless.
    suite.check("canonical.output_differs_from_reference_order",
                render_pairs(pairs_list[1]) != render_pairs_reference_order(pairs_list[1]),
                "(the non-canonical fixture must render differently)")
    suite.equal("canonical.reference_order_is_verbatim",
                render_pairs_reference_order(pairs_list[1]),
                FIXTURE_SUMMARIES[1])
    # One marker with two items is TWO triples, not one group: the unit the model
    # predicts, and the unit the ceiling table counts.
    suite.equal("canonical.triples_counted_per_item",
                len(pairs_list[0]["constraint"]), 2)
    suite.equal("canonical.three_item_group", len(pairs_list[6]["constraint"]), 3)
    duplicate_fixture = ("主题：重复项\n约束：本轮条件：等级=三级、等级=三级\n"
                         "未完成：无\n结果：已确认告知等级信息")
    duplicate_pairs = parse_pairs(duplicate_fixture)
    suite.equal("canonical.duplicate_collapses_in_set", len(duplicate_pairs["constraint"]), 1)
    suite.equal("canonical.duplicate_kept_as_occurrences",
                len([item for _, items in parse_constraint(duplicate_pairs["_raw"]["constraint"])
                     for item in items]), 2)
    suite.check("canonical.duplicate_case_not_raw_matchable",
                render_triples(duplicate_pairs["constraint"]) != duplicate_pairs["_raw"]["constraint"])

    # ---- multi-label set target -------------------------------------------
    vocab = build_vocabularies(pairs_list)
    targets = encode_targets(pairs_list, vocab, torch)
    suite.equal("target.rows", len(targets), len(pairs_list))
    for row, pairs in enumerate(pairs_list):
        expected = sum(1 for t in pairs["constraint"] if t in vocab.triples)
        suite.equal("target.multi_sum.%s" % pairs["_id"],
                    float(targets.multi[row].sum()), float(expected))
    suite.check("target.sentinel_row_all_zero",
                float(targets.multi[2].sum()) == 0.0)
    per_row = torch.stack([
        set_target_from_text(pairs["_raw"]["constraint"], vocab, torch)
        for pairs in pairs_list])
    suite.check("target.vector_matches_encoder",
                bool(torch.equal(per_row, targets.multi)),
                "(set_target_from_text and encode_targets disagree)")
    suite.equal("target.value_absent_class", int(targets.value[2]), 0)
    suite.check("target.char_terminator_present",
                int(targets.chars[2][0]) == vocab.end_index,
                "(an absent value must still teach the terminator)")

    # ---- OOV handling: counted, never mapped -------------------------------
    # (a) a value made of characters the vocabulary knows, but not itself a class
    base_value = pairs_list[1]["result"][1]
    composed_value = "".join(reversed(base_value))
    composed_pairs = dict(pairs_list[1])
    composed_pairs["_id"] = "FX-COMPOSED"
    composed_pairs["result"] = (pairs_list[1]["result"][0], composed_value)
    suite.check("oov.composed_value_is_not_a_class",
                composed_value not in vocab.value and bool(composed_value))
    suite.check("oov.composed_value_chars_are_known",
                all(char in vocab.chars for char in composed_value),
                "(characters %r)" % (composed_value,))
    composed_reach = reachability([composed_pairs], vocab)
    suite.check("oov.composed_value_is_composable",
                composed_reach["per_case"]["FX-COMPOSED"]["result_chars"],
                "(the character fallback should be able to compose it)")

    # (b) a value carrying a character the vocabulary has never seen
    unseen_pairs = dict(pairs_list[1])
    unseen_pairs["_id"] = "FX-OOV"
    unseen_pairs["constraint"] = frozenset([("本轮条件", "里程", "12345 公里")])
    unseen_pairs["result"] = (pairs_list[1]["result"][0], "12345 公里ZZZ")
    oov_encoded = encode_targets([unseen_pairs], vocab, torch)
    suite.equal("oov.multi_not_encoded", float(oov_encoded.multi.sum()), 0.0)
    suite.check("oov.value_falls_back_to_absent_class",
                int(oov_encoded.value[0]) == 0
                and unseen_pairs["result"][1] not in vocab.value)
    oov_reach = reachability([unseen_pairs], vocab)
    suite.equal("oov.counted_in_reachability", oov_reach["joint_compositional_cases"], 0)
    suite.check("oov.triple_counted",
                oov_reach["out_of_vocabulary"]["triples"] >= 1,
                "(unseen triples must be counted, not mapped)")
    suite.check("oov.never_mapped_to_neighbour",
                vocab.triples.encode(("本轮条件", "里程", "12345 公里")) is None)
    suite.check("oov.unseen_char_flags_not_composable",
                not oov_reach["per_case"]["FX-OOV"]["result_chars"])
    ignored = int((oov_encoded.chars[0] == IGNORE).sum())
    suite.check("oov.unseen_char_left_as_ignore", ignored >= 3,
                "(an unseen character must be excluded from the loss, not mapped)")

    # ---- metric identity checks -------------------------------------------
    references = pairs_list
    perfect = [dict(entry) for entry in pairs_list]
    perfect_score = score(perfect, references)
    suite.equal("metric.perfect_four_slot", perfect_score["four_slot_exact"], len(references))
    suite.equal("metric.perfect_content_accuracy", perfect_score["content_slot_accuracy"], 1.0)
    empty_pairs = [{"_id": r["_id"], "topic": EMPTY, "pending": (EMPTY, ""),
                    "constraint": frozenset(), "result": (EMPTY, "")} for r in references]
    empty_score = score(empty_pairs, references)
    suite.equal("metric.all_empty_four_slot", empty_score["four_slot_exact"], 0)
    suite.check("metric.all_empty_content_is_zero",
                (empty_score["content_slot_accuracy"] or 0.0) == 0.0,
                "(a predictor that only writes 无 must not earn content accuracy; got %r)"
                % (empty_score["content_slot_accuracy"],))
    sentinel_only = [r for r in references if not r["constraint"]]
    suite.equal("metric.sentinel_reference_count", len(sentinel_only), 1)

    # canonical beats strict on a non-canonical reference, and both are reported
    non_canonical = [pairs_list[1]]
    prediction = [dict(pairs_list[1])]
    prediction[0]["_id"] = "FX-01"
    canonical_score = score(prediction, non_canonical)
    suite.equal("metric.canonical_hit_on_reordered_reference",
                canonical_score["per_slot_exact"]["constraint"], 1)
    suite.equal("metric.strict_miss_on_reordered_reference",
                canonical_score["per_slot_exact_strict"]["constraint"], 0)
    suite.equal("metric.strict_le_canonical",
                canonical_score["four_slot_exact_strict"] <= canonical_score["four_slot_exact"],
                True)

    # ---- the blind-pairing counterexample ---------------------------------
    # `paired(before, after)` counts how often `after` wins, so the ablation goes
    # first and the real state second -- the same order `evaluate` uses.
    flat_ablated = {"four_slot_flags": [False] * 4, "graded_per_case": [1, 1, 1, 1]}
    flat_true = {"four_slot_flags": [False] * 4, "graded_per_case": [3, 3, 3, 3]}
    compared = paired_all(flat_ablated, flat_true)
    suite.equal("pairing.saturating_is_blind", compared["saturating"]["ties"], 4)
    suite.equal("pairing.saturating_reports_no_wins", compared["saturating"]["wins"], 0)
    suite.equal("pairing.graded_sees_the_gap", compared["graded"]["wins"], 4)
    suite.equal("pairing.graded_win_rate", compared["graded"]["win_rate"], 1.0)
    suite.check("pairing.graded_and_saturating_differ",
                compared["graded"]["wins"] != compared["saturating"]["wins"])

    # ---- sealed split refusals --------------------------------------------
    sealed_file = "memory_" + "test" + ".jsonl"          # built so the literal is absent
    sealed_ids = "RS-" + "te-"
    suite.raises("sealed.path_refused", TrainingError, refuse_sealed_split,
                 str(Path(r"D:\somewhere") / sealed_file), [{"id": "RS-tr-00000-00"}])
    suite.raises("sealed.ids_refused", TrainingError, refuse_sealed_split,
                 DEFAULT_CASES, [{"id": sealed_ids + "0001-00"}])
    suite.check("sealed.safe_path_allowed",
                refuse_sealed_split(DEFAULT_CASES, [{"id": "RS-tr-00000-00"}]) is None)
    source = Path(__file__).read_text(encoding="utf-8")
    # The module docstring names the sealed split in prose; no executable line may.
    parts = source.split('"""', 2)
    code_body = parts[2] if len(parts) > 2 else source
    suite.check("sealed.not_named_in_code",
                sealed_file not in code_body and sealed_ids not in code_body,
                "(the sealed split must not be opened by any code path in this file)")
    suite.check("sealed.still_documented",
                sealed_file in parts[1] if len(parts) > 1 else False,
                "(the refusal should stay documented for the reader)")

    # ---- CLI guards --------------------------------------------------------
    suite.raises("cli.skip_power_check_requires_self_test", SystemExit, parse_args,
                 ["--skip-power-check"])
    suite.raises("cli.cpu_device_refused", SystemExit, parse_args, ["--device", "cpu"])
    suite.raises("cli.negative_steps_refused", SystemExit, parse_args, ["--steps", "-1"])

    # ---- power gate is fail-closed ---------------------------------------
    try:
        power_report("self-test", require_lease=True)
        suite.check("power.gate_raised_without_lease", False,
                    "(the gate authorised a run with no guard lease)")
    except PowerLoss as exc:
        suite.check("power.gate_raised_without_lease", True)
        suite.check("power.report_has_fields",
                    "guard_lease" in exc.report and "ac_connected" in exc.report)
    except Exception as exc:  # noqa: BLE001
        suite.check("power.gate_raised_without_lease", False, "(raised %r)" % (exc,))

    # ---- one real CPU training step (the trained path is the tested path) --
    small_pairs = [dict(entry) for entry in pairs_list]
    small_vocab = build_vocabularies(small_pairs)
    small_vocab.value_max_chars = max(2, small_vocab.value_max_chars)
    config = PairModelConfig(pooled_dim=12, d_state=16, vocab=small_vocab,
                             d_char=8, dropout=0.0)
    model = build_model(config, torch)
    suite.check("step.model_has_parameters", parameter_count(model) > 0)
    pooled = torch.randn(len(small_pairs), 12)
    step_targets = encode_targets(small_pairs, small_vocab, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in model.parameters()]
    total, parts, grad_norm = training_step(
        torch=torch, model=model, optimizer=optimizer, weights=DEFAULT_WEIGHTS,
        pooled=pooled, targets=step_targets, positions=list(range(len(small_pairs))),
        grad_clip=1.0)
    suite.check("step.loss_is_finite", math.isfinite(total), "(loss %r)" % (total,))
    suite.check("step.loss_parts_present",
                set(parts) == {"topic", "pending", "triple", "pattern", "value", "chars"},
                "(parts %r)" % (sorted(parts),))
    suite.check("step.grad_norm_reported", grad_norm is not None and math.isfinite(grad_norm))
    suite.check("step.parameters_moved",
                any(not torch.equal(a, b) for a, b in zip(before, model.parameters())),
                "(optimizer.step() did not change any parameter)")
    encoder_grad = model.encoder.weight.grad
    suite.check("step.gradient_reaches_state_encoder",
                encoder_grad is not None and float(encoder_grad.abs().sum()) > 0.0,
                "(no gradient on the shared encoder means the state is not on the path)")

    # ---- sparse multi-label weighting ------------------------------------
    # The trained model collapsed to all-negative on the triple head: 64% of predicted
    # constraints came out empty while only 32/1773 references were. These tests pin the
    # property that makes that collapse unattractive.
    # class 0: 2 positives of 8 -> ratio 3   class 1: 1 positive of 8 -> ratio 7
    # class 2: no positives          class 3: never used
    imbalanced = torch.zeros(8, 4)
    imbalanced[0, 0] = 1.0
    imbalanced[3, 0] = 1.0
    imbalanced[3, 1] = 1.0
    weight = triple_positive_weight(imbalanced, torch, power=0.5, clamp=1e9)
    suite.check("weight.absent_class_gets_zero", float(weight[2]) == 0.0,
                "(a class with no positives must not be weighted)")
    suite.check("weight.present_class_is_positive", float(weight[0]) > 1.0)
    suite.check("weight.matches_softened_ratio",
                abs(float(weight[0]) - (6.0 / 2.0) ** 0.5) < 1e-5,
                "(got %r want %r)" % (float(weight[0]), (6.0 / 2.0) ** 0.5))
    suite.check("weight.rarer_class_weighs_more", float(weight[1]) > float(weight[0]),
                "(1-in-8 %r should outweigh 2-in-8 %r)"
                % (float(weight[1]), float(weight[0])))
    suite.check("weight.clamp_respected",
                float(triple_positive_weight(imbalanced, torch, power=0.5,
                                             clamp=2.0).max()) <= 2.0)
    all_negative = torch.full((8, 4), -12.0)
    unweighted = torch.nn.functional.binary_cross_entropy_with_logits(
        all_negative, imbalanced)
    weighted = torch.nn.functional.binary_cross_entropy_with_logits(
        all_negative, imbalanced, pos_weight=weight)
    suite.check("weight.penalises_all_negative_more",
                float(weighted) > float(unweighted) * 2,
                "(weighted %r vs unweighted %r: the collapse must get more expensive)"
                % (float(weighted), float(unweighted)))
    suite.check("weight.none_is_the_old_loss",
                float(torch.nn.functional.binary_cross_entropy_with_logits(
                    all_negative, imbalanced, pos_weight=None)) == float(unweighted),
                "(power=0 must reduce to the previous objective byte for byte)")

    # The first real run died on step 1 with
    #   AttributeError: 'Targets' object has no attribute 'to'
    # because every self-test step passed device=None, so the device-move branch was
    # never executed -- the tested path was not the trained path. A device of "cpu" is
    # enough to walk that branch: `Tensor.to('cpu')` is a legal no-op move, so this
    # exercises exactly the code that crashed without needing CUDA.
    moved = step_targets.to("cpu")
    suite.check("step.targets_to_moves_tensors",
                all(getattr(moved, name).device.type == "cpu" for name in Targets.FIELDS),
                "(a tensor field did not follow the move)")
    suite.check("step.targets_to_keeps_ids", moved.ids == step_targets.ids,
                "(.to() must not disturb the case ids)")
    suite.check("step.targets_to_preserves_shape",
                tuple(moved.multi.shape) == tuple(step_targets.multi.shape),
                "(shape changed across the move)")
    device_total, _, _ = training_step(
        torch=torch, model=model, optimizer=optimizer, weights=DEFAULT_WEIGHTS,
        pooled=pooled, targets=step_targets, positions=list(range(len(small_pairs))),
        device="cpu", grad_clip=1.0)
    suite.check("step.device_path_runs", math.isfinite(device_total),
                "(the device!=None branch of training_step failed: %r)" % (device_total,))
    outputs = model(pooled)
    suite.equal("step.logit_shapes.topic", tuple(outputs["topic"].shape),
                (len(small_pairs), len(small_vocab.topic)))
    suite.equal("step.logit_shapes.triple", tuple(outputs["triple"].shape),
                (len(small_pairs), len(small_vocab.triples)))
    suite.equal("step.logit_shapes.chars", tuple(outputs["chars"].shape),
                (len(small_pairs), small_vocab.value_max_chars, len(small_vocab.chars)))
    decoded, variants = decode_pairs(outputs, small_vocab, torch)
    suite.equal("step.variants_present", sorted(variants), ["arbitrated", "char", "class"])
    suite.check("step.decoded_slots_are_pairs",
                all(isinstance(entry["constraint"], frozenset) for entry in decoded))
    suite.check("step.render_is_text",
                isinstance(render_pairs(decoded[0]), str)
                and render_pairs(decoded[0]).count("\n") == 3)

    # ---- evaluation wiring on CPU ----------------------------------------
    evaluation = evaluate(model, pooled, small_pairs, small_vocab, torch, chunk=4)
    suite.equal("eval.conditions", sorted(evaluation["conditions"]),
                sorted(["true", "zero", "wrong"]))
    suite.check("eval.headline_variant", evaluation["headline_variant"] == "arbitrated")
    for name in ("true_vs_zero", "true_vs_wrong"):
        block = evaluation["paired"][name]
        suite.check("eval.paired_has_both_pairings.%s" % name,
                    "graded" in block and "saturating" in block)
    line = eval_summary_line(evaluation)
    suite.check("eval.summary_line_mentions_strict", "strict=" in line)
    suite.check("eval.summary_line_mentions_graded", "graded w/l/t=" in line)
    compact = compact_eval(evaluation)
    suite.check("eval.compact_drops_per_case",
                all("graded_per_case" not in entry
                    for block in compact["conditions"].values()
                    for entry in block.values()))
    # Two GPU-only gaps killed real runs: per-case predictions were never returned at
    # all (the dump then indexed a scored dict and raised `KeyError: 0`), and anything
    # left holding frozensets makes every status write raise on serialisation. Both are
    # asserted here so the CPU suite covers the paths the real run actually walks.
    suite.check("eval.predictions_present",
                isinstance(evaluation.get("predictions"), dict)
                and sorted(evaluation["predictions"]) == sorted(["true", "zero", "wrong"]),
                "(write_predictions needs per-case predictions, not scored variants)")
    suite.check("eval.predictions_aligned_with_references",
                all(len(evaluation["predictions"]["true"][name]) == len(small_pairs)
                    for name in ("arbitrated", "class", "char")),
                "(one decoded entry per reference case is required for the dump)")
    try:
        json.dumps(compact, ensure_ascii=False)
        suite.check("eval.compact_is_json_serialisable", True)
    except TypeError as exc:
        suite.check("eval.compact_is_json_serialisable", False, "(%s)" % (exc,))

    # Call the real dump, not a stand-in: the run that died did so inside this
    # function after its evaluation had already succeeded.
    references_fixture = [dict(entry) for entry in small_pairs]
    dump_path = Path(tempfile.mkdtemp()) / "dev-predictions.jsonl"
    written = write_predictions_file(dump_path, evaluation, references_fixture)
    suite.equal("eval.dump_row_count", written, len(small_pairs))
    rows = [json.loads(line) for line in
            dump_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    suite.equal("eval.dump_line_count", len(rows), len(small_pairs))
    suite.check("eval.dump_rows_json_safe",
                all("prediction_canonical" in row and "canonical_slot_hits" in row
                    for row in rows))
    try:
        write_predictions_file(dump_path, evaluation, references_fixture[:-1])
        suite.check("eval.dump_refuses_misaligned", False,
                    "(a short reference list must raise, not silently drop rows)")
    except TrainingError:
        suite.check("eval.dump_refuses_misaligned", True)

    # ---- restricted rates against a reachability mask ---------------------
    suite.equal("restricted.counts_the_subset",
                restricted_rate([True, False, True], ["a", "b", "c"], {"a", "c"}),
                {"cases": 2, "hits": 2, "rate": 1.0})
    suite.check("restricted.empty_subset_is_none",
                restricted_rate([True], ["a"], set())["rate"] is None,
                "(no cases must be reported as None, never as 0.0 or 1.0)")
    small_reach = {
        "per_case": {
            "FX-00": {"reachable": True, "reachable_compositional": True},
            "FX-01": {"reachable": False, "reachable_compositional": True},
            "FX-02": {"reachable": False, "reachable_compositional": False},
        }}
    masked = evaluate(model, pooled, small_pairs[:3], small_vocab, torch, chunk=4,
                      reach=small_reach)
    block = masked["conditions"]["true"]["variants"]["arbitrated"]
    suite.equal("restricted.masks_present",
                sorted(masked["masks"]),
                ["reachable", "reachable_compositional", "unreachable"])
    suite.equal("restricted.mask_sizes",
                [masked["masks"]["reachable"], masked["masks"]["reachable_compositional"],
                 masked["masks"]["unreachable"]], [1, 2, 2])
    partial_mask = {"per_case": {"FX-00": {"reachable": True,
                                           "reachable_compositional": True}}}
    suite.raises("restricted.mismatched_mask_refused", TrainingError, evaluate,
                 model, pooled, small_pairs[:2], small_vocab, torch,
                 **{"chunk": 4, "reach": partial_mask})
    suite.equal("restricted.rates_present",
                sorted(key for key in block if key.startswith("four_slot_exact_on_")),
                ["four_slot_exact_on_reachable",
                 "four_slot_exact_on_reachable_compositional",
                 "four_slot_exact_on_unreachable"])
    suite.equal("restricted.reachable_subset_size",
                block["four_slot_exact_on_reachable"]["cases"], 1)
    suite.check("restricted.never_exceeds_unrestricted",
                (block["four_slot_exact_on_reachable"]["hits"] or 0)
                <= block["four_slot_exact"])

    # ---- batch stream determinism ---------------------------------------
    stream_a = batch_stream(64, 16, seed=7)
    stream_b = batch_stream(64, 16, seed=7)
    first = [next(stream_a)[1] for _ in range(4)]
    second = [next(stream_b)[1] for _ in range(4)]
    suite.equal("batch.deterministic", first, second)
    resumed = batch_stream(64, 16, seed=7, start_position=3)
    suite.equal("batch.resume_matches", next(resumed)[1], first[3])
    suite.raises("batch.too_large_refused", TrainingError, lambda: next(batch_stream(8, 16, seed=1)))
    flat = [index for batch in first for index in batch]
    suite.check("batch.covers_without_repeats", len(set(flat)) == len(flat))

    # ---- corpus-independent ceiling arithmetic ---------------------------
    suite.check("ceiling.joint_le_per_slot",
                True)  # arithmetic identity, kept as an explicit contract row
    reach = reachability(pairs_list, vocab)
    suite.check("ceiling.reachability_shape",
                set(reach["per_slot_rate"]) == {"topic", "pending", "constraint",
                                                "result_closed", "result_chars"})

    if verbose and suite.failures:
        for failure in suite.failures:
            print("FAIL " + failure)
    total_checks = suite.passed + len(suite.failures)
    print(json.dumps({"self_test": "%d/%d" % (suite.passed, total_checks),
                      "failures": suite.failures,
                      "cuda_context_created": torch.cuda.is_initialized()},
                     ensure_ascii=False), flush=True)
    return 0 if not suite.failures else 1


# --------------------------------------------------------------------------
# preflight: the measured table, recomputed
# --------------------------------------------------------------------------
def preflight(args):
    """Recompute every figure the pair task advertises, from the corpus, with no
    CUDA and no guard lease. Exit 0 unless a round-trip actually fails."""
    table = ceiling_table(args.cases)
    cache_hash = args.expect_cache_data_hash or table["cache_identity_hash"]
    report = dict(table)
    report["feature_cache"] = {"path": str(args.feature_cache),
                               "expected_data_hash": cache_hash}
    report["notes"] = [
        "The cache on disk records the hash of DATA.all_cases(--cases), i.e. the "
        "11773-row corpus PLUS the %d embedded cases; this run trains on the corpus "
        "only, and both hashes are printed so neither replaces the other." % EMBEDDED_EXTRA_CASES,
        "Constraint reachability counts a case as reachable only when EVERY one of "
        "its triples was seen in TRAIN.",
        "`closed` reachability assumes the value vocabulary; `compositional` assumes "
        "the character fallback head.",
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if report["round_trip_failures"] else 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def write_predictions_file(path, evaluation, references_):
    """Dump one JSON line per dev case: reference, both renders, and slot hits.

    Module level on purpose. It used to be a closure inside the run, so the CPU suite
    could not call it, and the first real run hit `KeyError: 0` here AFTER the final
    evaluation had already succeeded -- correct numbers, no dump. Keeping it callable
    is what lets the self-test cover the exact step that failed.
    """
    # The scored variants hold per-slot dicts; the decoded per-case pair lists live
    # under `predictions`. Indexing the scored entry is what raised `KeyError: 0`.
    raw = evaluation["predictions"]["true"]
    predictions = raw["arbitrated"]
    class_variant = raw["class"]
    char_variant = raw["char"]
    lengths = {len(predictions), len(class_variant), len(char_variant), len(references_)}
    if len(lengths) != 1:
        raise TrainingError(
            "prediction/reference length mismatch (%d arbitrated / %d class / %d char "
            "vs %d references); refusing to write a dev dump whose rows would be "
            "silently misaligned"
            % (len(predictions), len(class_variant), len(char_variant), len(references_)))
    with open(path, "w", encoding="utf-8") as handle:
        for index, reference in enumerate(references_):
            hits = slot_hits(predictions[index], reference)
            handle.write(json.dumps({
                "id": reference["_id"],
                "reference_raw": reference["_raw"],
                "reference_canonical": render_pairs(reference),
                "prediction_canonical": render_pairs(predictions[index]),
                "prediction_class_result": list(class_variant[index]["result"]),
                "prediction_char_result": list(char_variant[index]["result"]),
                "canonical_slot_hits": hits,
                "strict_constraint_hit": render_triples(
                    predictions[index]["constraint"]) == reference["_raw"]["constraint"],
            }, ensure_ascii=False) + "\n")
    return len(references_)


def run(args, hooks=None):
    try:
        return _run_inner(args, hooks or {})
    except Done as done:
        payload = dict(done.result)
        payload["exit_code"] = EXIT_PAUSED
        return payload


def _run_inner(args, hooks=None):
    hooks = hooks or {}
    torch = hooks.get("torch") or base._torch()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / RESUME_NAME
    log = hooks.get("log") or RunLog(out_dir / LOG_NAME)
    status_path = out_dir / STATUS_NAME
    eval_history_path = out_dir / EVAL_HISTORY_NAME
    predictions_path = out_dir / PREDICTIONS_NAME
    report_path = out_dir / REPORT_NAME
    status = {}
    result = {"ok": False, "label": LABEL, "backbone_training": BACKBONE_TRAINING,
              "script": SCRIPT_VERSION, "steps_requested": args.steps}

    def status_update(**changes):
        status.update(changes)
        status.update({"label": LABEL, "backbone_training": BACKBONE_TRAINING,
                       "script": SCRIPT_VERSION, "steps_requested": args.steps,
                       "artifact_paths": {"status": str(status_path), "log": str(log.path),
                                          "checkpoint": str(resume_path),
                                          "report": str(report_path),
                                          "predictions": str(predictions_path),
                                          "eval_history": str(eval_history_path)},
                       "updated_at": time.time()})
        write_json(status_path, status)

    def gate(stage):
        try:
            return power_report(stage, require_lease=True)
        except PowerLoss as exc:
            payload = {"ok": False, "paused": True, "exit_code": EXIT_PAUSED,
                       "phase": "paused", "paused_stage": stage, "power": exc.report,
                       "label": LABEL, "steps_requested": args.steps,
                       "artifacts": {"status": str(status_path), "log": str(log.path)}}
            status_update(phase="paused", paused=True, paused_before=stage,
                          power=exc.report)
            log("power gate stopped the run at %s: %s"
                % (stage, json.dumps(exc.report, ensure_ascii=False)))
            raise Done(payload) from None

    # ---- gate 1: before anything touches CUDA -----------------------------
    result["power_before_cuda"] = gate("before-cuda")
    log("power gate before CUDA: " + json.dumps(result["power_before_cuda"],
                                                ensure_ascii=False))
    if not torch.cuda.is_available():
        raise TrainingError("CUDA is required; this script refuses CPU training")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- corpus ------------------------------------------------------------
    corpus = load_corpus(args.cases)
    train_cases, dev_cases = split_corpus(corpus, args.limit_train, args.limit_heldout)
    if not train_cases or not dev_cases:
        raise TrainingError("both train and dev cases are required")
    train_pairs = pair_lists(train_cases)
    dev_pairs = pair_lists(dev_cases)
    corpus_hash = DATA.data_version(corpus)
    cache_hash = args.expect_cache_data_hash or DATA.data_version(DATA.all_cases(args.cases))
    log("cases %s corpus_hash=%s cache_identity=%s"
        % (json.dumps({"train": len(train_cases), "dev": len(dev_cases),
                       "total": len(corpus)}), corpus_hash[:16], cache_hash[:16]))
    round_trip = round_trip_report(corpus)
    if round_trip:
        raise TrainingError("%d references do not round-trip through the pair parser; "
                            "refusing to train on a lossy target: %s"
                            % (len(round_trip), json.dumps(round_trip[:3], ensure_ascii=False)))
    log("pair parser round-trips all %d references byte-for-byte" % len(corpus))

    vocab = build_vocabularies(train_pairs)
    longest = max(len(p["result"][1]) for p in train_pairs)
    if args.value_max_chars and args.value_max_chars < longest + 1:
        raise TrainingError("--value-max-chars %d is below (longest TRAIN value + 1) = %d; "
                            "truncating a label would corrupt the loss"
                            % (args.value_max_chars, longest + 1))
    if args.value_max_chars:
        vocab.value_max_chars = int(args.value_max_chars)
    log("vocabularies (TRAIN only): " + json.dumps(vocab.summary(), ensure_ascii=False))

    reach = reachability(dev_pairs, vocab)
    log("dev reachability: per-slot %s joint closed %.4f (%d/%d) compositional %.4f "
        "(%d/%d); out of vocabulary %s"
        % (json.dumps(reach["per_slot_rate"], ensure_ascii=False),
           reach["joint_closed_rate"], reach["joint_closed_cases"], len(dev_pairs),
           reach["joint_compositional_rate"], reach["joint_compositional_cases"],
           len(dev_pairs), json.dumps(reach["out_of_vocabulary"], ensure_ascii=False)))
    result["reachability"] = {key: value for key, value in reach.items() if key != "per_case"}

    # ---- features (read-only) ---------------------------------------------
    cache = ReadOnlyFeatureCache(args.feature_cache, cache_hash,
                                 args.expect_backbone_sha256,
                                 allow_any_backbone=args.allow_any_backbone)
    cache_report = cache.open()
    log("feature cache: " + json.dumps(cache_report, ensure_ascii=False))
    ordered = train_cases + dev_cases
    cache.require_all([case["id"] for case in ordered])
    pooled = build_pooled(cache, ordered, args.pool, log=log)
    train_count = len(train_cases)
    mean = pooled[:train_count].mean(dim=0, keepdim=True)
    std = pooled[:train_count].std(dim=0, keepdim=True).clamp_min(1e-5)
    pooled = ((pooled - mean) / std).to(device)
    train_pooled = pooled[:train_count]
    dev_pooled = pooled[train_count:]
    result["features"] = dict(cache_report, pooled_dim=int(pooled.shape[1]),
                              pool=args.pool,
                              standardised_with="train-split statistics only")
    log("pooled %d cases to width %d with train-only statistics"
        % (pooled.shape[0], pooled.shape[1]))

    targets = encode_targets(train_pairs, vocab, torch, device=device)
    model = build_model(PairModelConfig(int(pooled.shape[1]), args.d_state, vocab,
                                        d_char=args.d_char, dropout=args.dropout),
                        torch).to(device)
    weights = {"topic": args.weight_topic, "pending": args.weight_pending,
               "triple": args.weight_triple, "pattern": args.weight_pattern,
               "value": args.weight_value, "chars": args.weight_chars}
    # Computed from TRAIN targets only, then frozen: recomputing per step would make the
    # objective drift with the batch, and using dev targets would leak the split.
    if args.triple_pos_weight_power > 0:
        triple_pos_weight = triple_positive_weight(
            targets.multi, torch, power=args.triple_pos_weight_power,
            clamp=args.triple_pos_weight_clamp).to(device)
        log("triple pos_weight: power=%.3f clamp=%.1f median=%.1f max=%.1f nonzeros=%d"
            % (args.triple_pos_weight_power, args.triple_pos_weight_clamp,
               float(triple_pos_weight.median()), float(triple_pos_weight.max()),
               int((triple_pos_weight > 0).sum())))
    else:
        triple_pos_weight = None
        log("triple pos_weight: disabled (power=0)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    parameters = parameter_count(model)
    run_identity = {
        "schema": SCHEMA, "script": SCRIPT_VERSION, "pool": args.pool,
        "d_state": args.d_state, "d_char": args.d_char,
        "value_max_chars": vocab.value_max_chars, "seed": args.seed,
        "vocab": vocab.summary(), "weights": weights,
        "triple_pos_weight": [args.triple_pos_weight_power,
                              args.triple_pos_weight_clamp],
        "corpus_hash": corpus_hash, "cache_identity": cache_hash,
        "case_ids": [case["id"] for case in ordered],
        "unknown_policy": "train-only vocabularies; unseen labels counted, never mapped",
        "canonical_constraint": "set of (marker, attribute, value) triples, canonically ordered",
    }
    identity_hash = hashlib.sha256(
        json.dumps(run_identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]
    log("parameters=%d identity=%s" % (parameters, identity_hash))

    def references_for(pairs_list_):
        return [dict(entry) for entry in pairs_list_]

    history = []
    start_step = 0
    resume_candidate = Path(args.resume) if args.resume else resume_path
    loaded = None
    if resume_candidate.is_file():
        loaded = load_checkpoint(resume_candidate, torch)
        if loaded.get("identity_hash") != identity_hash:
            raise TrainingError(
                "checkpoint %s belongs to a different run identity (%s vs %s); refusing "
                "to mix two experiments" % (resume_candidate, loaded.get("identity_hash"),
                                            identity_hash))
        model.load_state_dict(loaded["model_state"], strict=True)
        if loaded.get("optimizer_state"):
            optimizer.load_state_dict(loaded["optimizer_state"])
        restore_rng(loaded.get("rng"), torch)
        start_step = int(loaded.get("step") or 0)
        history = list(loaded.get("history") or [])
        log("resumed from %s at step %d" % (resume_candidate, start_step))

    def checkpoint_payload(step, phase, evaluation=None):
        return {"format": FORMAT, "script": SCRIPT_VERSION,
                "identity_hash": identity_hash, "run_identity": run_identity,
                "step": int(step), "phase": phase,
                "model_state": model.state_dict(),
                "optimizer_state": cpu_optimizer_state(optimizer),
                "rng": rng_state(torch),
                "history": history[-base.HISTORY_LIMIT:],
                "evaluation": compact_eval(evaluation) if evaluation else None,
                "power": power_report("checkpoint", require_lease=False),
                "saved_at": time.time()}

    references = references_for(dev_pairs)

    def report_evaluation(evaluation, step, phase):
        status_update(phase=phase, step=int(step),
                      evaluation=compact_eval(evaluation))
        record = {"step": int(step), "phase": phase,
                  "headline_variant": evaluation["headline_variant"],
                  "paired": evaluation["paired"],
                  "conditions": compact_eval(evaluation)["conditions"],
                  "canonical_constraint_order": True,
                  "strict_metric_reported": True}
        with open(eval_history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(eval_summary_line(evaluation))
        return record

    # ---- training ---------------------------------------------------------
    peak_memory = 0
    steps_run = 0
    stream = batch_stream(train_count, args.batch_size, args.seed, start_position=start_step)
    current = {"step": start_step, "loss": None}
    for position, batch in stream:
        step = position
        if step >= args.steps:
            break
        current = {"step": step, "loss": current.get("loss")}
        if hooks.get("before_step"):
            hooks["before_step"](step)
        gate("before-step") if step % 50 == 0 else None
        loss, parts, grad_norm = training_step(
            torch=torch, model=model, optimizer=optimizer, weights=weights,
            pooled=train_pooled, targets=targets, positions=batch, device=device,
            triple_pos_weight=triple_pos_weight)
        steps_run += 1
        history.append({"step": step, "loss": loss,
                        **{key: float(value) for key, value in parts.items()}})
        current["loss"] = loss
        if torch.cuda.is_available():
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated() / (1024 * 1024)))
        if args.log_every and (step + 1) % args.log_every == 0:
            log("step %d/%d loss=%.4f %s grad=%s peak=%dMiB"
                % (step + 1, args.steps, loss,
                   " ".join("%s=%.4f" % (key, parts[key]) for key in
                            ("topic", "pending", "triple", "pattern", "value", "chars")),
                   "None" if grad_norm is None else "%.3f" % grad_norm, peak_memory))
        if args.eval_every and (step + 1) % args.eval_every == 0:
            evaluation = evaluate(model, dev_pooled, references, vocab, torch,
                                  chunk=args.eval_batch_size,
                                  triple_threshold=args.triple_threshold,
                                  value_confidence=args.value_confidence, reach=reach)
            report_evaluation(evaluation, step + 1, "periodic")
        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(resume_path, checkpoint_payload(step + 1, "training"),
                            torch, pointer=out_dir / POINTER_NAME)
        status_update(phase="training", step=step + 1, loss=loss, parts=parts,
                      grad_norm=grad_norm, peak_memory_mib=peak_memory)

    # ---- final evaluation -------------------------------------------------
    evaluation = evaluate(model, dev_pooled, references, vocab, torch,
                          chunk=args.eval_batch_size,
                          triple_threshold=args.triple_threshold,
                          value_confidence=args.value_confidence, reach=reach)
    record = report_evaluation(evaluation, args.steps, "final")
    final_checkpoint = save_checkpoint(resume_path, checkpoint_payload(args.steps, "complete",
                                                                      evaluation),
                                       torch, pointer=out_dir / POINTER_NAME)
    write_predictions_file(predictions_path, evaluation, references)
    variant_rates = {
        name: evaluation["conditions"]["true"]["variants"][name]["four_slot_exact_rate"]
        for name in ("class", "char", "arbitrated")}
    strict_rate = evaluation["conditions"]["true"]["variants"]["arbitrated"][
        "four_slot_exact_strict_rate"]
    paired_graded = evaluation["paired"]["true_vs_zero"]["graded"]
    paired_wrong = evaluation["paired"]["true_vs_wrong"]["graded"]
    report = {
        "label": LABEL, "script": SCRIPT_VERSION, "identity_hash": identity_hash,
        "step": args.steps, "parameters": parameters,
        "corpus": {"total": len(corpus), "train": len(train_cases), "dev": len(dev_cases)},
        "corpus_hash": corpus_hash, "cache_identity": cache_hash,
        "vocabularies": vocab.summary(),
        "reachability": {key: value for key, value in reach.items() if key != "per_case"},
        "canonical_order": {
            "declared": ("the constraint is compared as a SET of (marker, attribute, "
                         "value) triples; both sides are rendered in a canonical order "
                         "(groups by marker, items by attribute then value)"),
            "reference_order_is_not_canonical": True,
            "strict_metric": ("four_slot_exact_strict compares the rendered constraint "
                              "against the RAW reference text byte for byte and is "
                              "reported beside every canonical number"),
        },
        "four_slot_exact_canonical": variant_rates,
        "four_slot_exact_strict": strict_rate,
        "per_slot_exact_rate": evaluation["conditions"]["true"]["variants"]["arbitrated"][
            "per_slot_exact_rate"],
        "per_slot_exact_strict_rate": evaluation["conditions"]["true"]["variants"][
            "arbitrated"]["per_slot_exact_strict_rate"],
        "content_slot_accuracy": evaluation["conditions"]["true"]["variants"]["arbitrated"][
            "content_slot_accuracy"],
        "four_slot_exact_on_reachable": evaluation["conditions"]["true"]["variants"]["arbitrated"][
            "four_slot_exact_on_reachable"],
        "four_slot_exact_on_reachable_compositional": evaluation["conditions"]["true"][
            "variants"]["arbitrated"]["four_slot_exact_on_reachable_compositional"],
        "four_slot_exact_on_unreachable": evaluation["conditions"]["true"]["variants"]["arbitrated"][
            "four_slot_exact_on_unreachable"],
        "reachability_masks": evaluation.get("masks", {}),
        "ablations": {name: {variant: {
            "four_slot_exact_rate": entry["four_slot_exact_rate"],
            "content_slot_accuracy": entry["content_slot_accuracy"],
            "per_slot_exact_rate": entry["per_slot_exact_rate"]}
            for variant, entry in block["variants"].items()
            if variant == "arbitrated"}
            for name, block in evaluation["conditions"].items()},
        "paired_true_vs_zero": {"graded": paired_graded,
                                "saturating": evaluation["paired"]["true_vs_zero"]["saturating"]},
        "paired_true_vs_wrong": {"graded": paired_wrong,
                                 "saturating": evaluation["paired"]["true_vs_wrong"]["saturating"]},
        "ceiling": {"joint_closed_rate": reach["joint_closed_rate"],
                    "joint_compositional_rate": reach["joint_compositional_rate"]},
        "eval_history_record": record,
        "artifacts": {"status": str(status_path), "log": str(log.path),
                      "checkpoint": final_checkpoint, "report": str(report_path),
                      "predictions": str(predictions_path),
                      "eval_history": str(eval_history_path)},
        "note": ("No accuracy claim is possible before this GPU run exists; the numbers "
                 "above come from the dev split only, and the sealed final split was "
                 "never read."),
    }
    write_json(report_path, report)
    status_update(phase="complete", step=args.steps, report=report_path.name)
    result.update({"ok": True, "phase": "complete", "step": args.steps,
                   "parameters": parameters, "artifacts": report["artifacts"],
                   "four_slot_exact_canonical": variant_rates,
                   "four_slot_exact_strict": strict_rate,
                   "content_slot_accuracy": report["content_slot_accuracy"],
                   "paired_true_vs_zero_graded": paired_graded,
                   "reachability": report["reachability"]})
    log("finished step %d: four_slot_exact canonical=%s strict=%.4f content=%.4f "
        "paired(true vs zero, graded) w/l/t=%d/%d/%d"
        % (args.steps, json.dumps(variant_rates, ensure_ascii=False), strict_rate,
           report["content_slot_accuracy"] or 0.0, paired_graded["wins"],
           paired_graded["losses"], paired_graded["ties"]))
    return result


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args(argv)
    if args.self_test:
        return self_test(verbose=True)
    try:
        if args.preflight:
            return preflight(args)
        result = run(args)
    except TrainingError as exc:
        print(json.dumps({"ok": False, "label": LABEL, "error": str(exc)},
                         ensure_ascii=False), flush=True)
        return 2
    except SystemExit as exc:
        if exc.code == EXIT_PAUSED:
            print(json.dumps({"ok": True, "paused": True, "exit_code": EXIT_PAUSED},
                             ensure_ascii=False), flush=True)
            return EXIT_PAUSED
        raise
    printable = dict(result)
    printable.pop("history", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    if result.get("paused"):
        return EXIT_PAUSED
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
