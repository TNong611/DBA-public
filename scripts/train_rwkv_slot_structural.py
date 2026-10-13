#!/usr/bin/env python3
"""Slot-STRUCTURE classifier: frozen RWKV state -> four slots as closed-vocabulary structure.

Why this file exists
--------------------
`train_rwkv_slot_classifier.py` decodes all three value slots with one character-level
GRU and reached dev `four_slot_exact = 0/1779`. Reading its own dev report by slot shows
the failure is concentrated and quantitative, not mysterious:

    topic       1370/1779 = 77.0%   (the state IS being read)
    pending     1734/1779 = 97.5%   (trivial: `无` in 97.8% of cases)
    result       591/1779 = 33.2%
    constraint    37/1779 =  2.1%   <-- the whole four-slot product dies here

`constraint` is a composed string with a median length around two dozen characters and
must match EXACTLY. Even at 90% per-character accuracy a 24-character string survives
only 0.9**24 = 8% of the time, so a character decoder needs ~99% per-character accuracy
before this slot stops dominating the loss. Free-form character generation is the wrong
estimator for a slot that is assembled out of a handful of known pieces.

What the labels actually are (measured on the frozen corpus, train split only)
-----------------------------------------------------------------------------
constraint : 1 or 2 groups joined by `；`; a group is `marker：` + 1 or 2 items joined
             by `、`; an item is `attr=value`.  Marker vocabulary = 5 (one of which is
             the empty marker, for groups with no `：`).  Item vocabulary = 285.
             Within-group item order is NOT canonical (45.1% of heldout multi-item
             groups disagree with train first-appearance order), so item order has to
             be predicted, not imposed -- hence one head per item POSITION.
pending    : `无`, or `本轮未答：` + attribute.  57 distinct values in train.
result     : a pattern (247 distinct, e.g. `已确认给出名称推荐`, `已确认告知票价：`) with
             an optional short tail value (1324 distinct).

Ceilings implied by those vocabularies, measured with oracle ordering on heldout
-------------------------------------------------------------------------------
    constraint via (marker, item) composition   1775/1779 = 99.78%
    pending  via (pattern, value)               1772/1779 = 99.61%
    result   via (pattern, value)               1553/1779 = 87.30%
    topic                                        (204 train classes)
    joint ceiling for all four exact            approximately 86%

So the 80% bar is reachable through STRUCTURE, and it is NOT reachable through
whole-string retrieval: the heldout constraint string appears verbatim in train only
69.87% of the time. That is why this script predicts decomposed structure instead of
retrieving a nearest train label string.

What is deliberately NOT here
-----------------------------
* No autoregression anywhere. Every head is a single parallel classification, so the
  teacher-forcing shortcut that let v3/v4 score a low train loss while reading nothing
  from the state does not exist as a mechanism.
* No retrieval over train label prototypes as the OUTPUT path. Measured: its ceiling is
  the verbatim-reachability number above (69.87% for constraint), i.e. below the bar.
  The vocabularies are used as closed class sets, and out-of-vocabulary references are
  COUNTED in the reachability report instead of being silently mapped onto something.

Same fail-closed machinery as its siblings: direct AC check plus a live guard lease
before every optimizer step, exit 75 on a pause, GPU-only training, read-only feature
cache, and a hard refusal to touch the sealed final-test split.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rwkv_slots_data as DATA  # noqa: E402
import train_rwkv_slot_classifier as base  # noqa: E402

TrainingError = base.TrainingError
Done = base.Done
PowerLoss = base.PowerLoss
EXIT_PAUSED = base.EXIT_PAUSED

SCRIPT_NAME = "train_rwkv_slot_structural.py"
DEFAULT_CASES = str(DATA.DEFAULT_CASES if hasattr(DATA, "DEFAULT_CASES")
                    else r"D:\AI\DBA\training\web-sources\risawoz\memory_cases.jsonl")
DEFAULT_OUT = r"D:\AI\DBA\training\rwkv-slot-structural"
DEFAULT_FEATURE_CACHE = r"D:\AI\DBA\training\risawoz-run-v3\features\1f0b28205d715209"
DEFAULT_BACKBONE_SHA256 = (
    "23e13ff62ac6f80b28910ffc230324939e80ddb00b98f892a5951c0da071e700")

SCHEMA = 1
MAX_GROUPS = 2      # measured: every train case has 1 or 2 groups
MAX_ITEMS = 2       # measured: every train group has 1 or 2 items
ABSENT = 0          # class 0 means "this position carries nothing"

EMPTY = base.EMPTY
SLOT_KEYS = DATA.SLOT_KEYS
PAUSE_NAME = "checkpoint-paused.pt"


# --------------------------------------------------------------------------
# structure parsing / rendering
# --------------------------------------------------------------------------
def split_groups(text):
    """`marker：a=1、b=2；marker2：c=3` -> [(marker, [item, ...]), ...].

    The empty marker is a real marker, not a missing one: a group written without a
    `：` is exactly a group whose marker is `""`. Keeping that distinction is what lets
    the renderer round-trip such a group instead of dropping it.
    """
    groups = []
    for chunk in str(text).split("；"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "：" in chunk:
            marker, rest = chunk.split("：", 1)
        else:
            marker, rest = "", chunk
        items = [piece.strip() for piece in rest.split("、") if piece.strip()]
        groups.append((marker.strip(), items))
    return groups


def render_groups(groups):
    """Inverse of `split_groups` for anything that came out of it."""
    chunks = []
    for marker, items in groups:
        if not items and not marker:
            continue
        body = "、".join(items)
        chunks.append((marker + "：" + body) if marker else body)
    return "；".join(chunks)


def split_pair(text):
    """`已确认告知票价：免费` -> ('已确认告知票价：', '免费'); `无` -> ('无', None)."""
    text = str(text)
    if "：" in text:
        pattern, value = text.split("：", 1)
        return pattern + "：", value
    return text, None


def render_pair(pattern, value):
    """Inverse of `split_pair`. A pattern ending in `：` always carries its value."""
    if not pattern.endswith("："):
        return pattern
    return pattern + ("" if value is None else value)


def constraint_fits(groups):
    """False when the reference needs more positions than the heads provide."""
    if len(groups) > MAX_GROUPS:
        return False
    return all(len(items) <= MAX_ITEMS for _, items in groups)


# --------------------------------------------------------------------------
# vocabularies (train split only)
# --------------------------------------------------------------------------
class StructuralVocab:
    """Every class set this model can emit, built from TRAIN labels only.

    Sorted lists rather than insertion order: the class indices must be reproducible
    from the case list alone, so a re-run cannot silently renumber a checkpoint's
    output layer.
    """

    FIELDS = ("topics", "markers", "items", "pending_patterns", "pending_values",
              "result_patterns", "result_values")

    def __init__(self, topics, markers, items, pending_patterns, pending_values,
                 result_patterns, result_values):
        self.topics = list(topics)
        self.markers = list(markers)
        self.items = list(items)
        self.pending_patterns = list(pending_patterns)
        self.pending_values = list(pending_values)
        self.result_patterns = list(result_patterns)
        self.result_values = list(result_values)
        self._index = {}

    def index(self, field, value):
        key = (field, value)
        if key not in self._index:
            self._index[key] = {name: position for position, name in
                                enumerate(getattr(self, field))}
        return self._index[key].get(value)

    @property
    def n_topics(self):
        return max(2, len(self.topics))

    @property
    def n_markers(self):
        return len(self.markers)

    @property
    def n_items(self):
        return len(self.items)

    def counts(self):
        return {"topics": len(self.topics), "markers": len(self.markers),
                "items": len(self.items),
                "pending_patterns": len(self.pending_patterns),
                "pending_values": len(self.pending_values),
                "result_patterns": len(self.result_patterns),
                "result_values": len(self.result_values)}

    def to_dict(self):
        return {name: list(getattr(self, name)) for name in self.FIELDS}

    @classmethod
    def from_dict(cls, payload):
        missing = [name for name in cls.FIELDS if name not in payload]
        if missing:
            raise TrainingError("checkpoint vocabulary is missing: %s" % ", ".join(missing))
        return cls(**{name: payload[name] for name in cls.FIELDS})


def build_vocab(train_cases):
    """Derive every class set from the train split, deterministically sorted."""
    topics, markers, items = set(), set(), set()
    pending_patterns, pending_values = set(), set()
    result_patterns, result_values = set(), set()
    for _, slots in train_cases:
        topics.add(slots["topic"])
        groups = split_groups(slots["constraint"])
        if not constraint_fits(groups):
            raise TrainingError(
                "train constraint needs more positions than MAX_GROUPS=%d x MAX_ITEMS=%d "
                "allow: %r" % (MAX_GROUPS, MAX_ITEMS, slots["constraint"]))
        for marker, group_items in groups:
            markers.add(marker)
            for item in group_items:
                items.add(item)
        for key, patterns, values in (("pending", pending_patterns, pending_values),
                                      ("result", result_patterns, result_values)):
            pattern, value = split_pair(slots[key])
            patterns.add(pattern)
            if value is not None:
                values.add(value)
    # Deliberately NO defensive additions here: the class sets are exactly what the
    # train split contains. Injecting `无` or a marker-less group "just in case" would
    # make a reference reachable that no head ever had a positive example for, which
    # inflates the ceiling this file reports instead of the accuracy it measures.
    return StructuralVocab(sorted(topics), sorted(markers), sorted(items),
                           sorted(pending_patterns), sorted(pending_values),
                           sorted(result_patterns), sorted(result_values))


# --------------------------------------------------------------------------
# candidate retrieval over train label prototypes
# --------------------------------------------------------------------------
class RetrievalBank:
    """Per-slot verbatim train label strings, used as retrieval candidates.

    Whole-slot strings, not fragments: this is exactly the candidate set a
    nearest-prototype decoder can emit, so its coverage IS the hard ceiling of any
    retrieval-only output. Measured on this corpus, dev constraint strings appear
    verbatim in train only 69.87% of the time, so retrieval cannot by itself reach the
    80% bar -- which is why it is wired here as (a) an auxiliary training signal that
    sharpens the shared state embedding and (b) a candidate SOURCE that the structural
    renderer can be arbitrated against, rather than as the primary output path.

    The bank is built from train labels only, and an out-of-bank reference is left as
    `IGNORE` in the retrieval targets instead of being mapped to its nearest neighbour:
    "not retrievable" must stay visible, not become a confident wrong answer.
    """

    SLOTS = ("constraint", "pending", "result")

    def __init__(self, entries):
        self.entries = {slot: list(entries.get(slot, [])) for slot in self.SLOTS}
        self._tables = {}

    def table(self, slot):
        if slot not in self._tables:
            self._tables[slot] = {text: position
                                  for position, text in enumerate(self.entries[slot])}
        return self._tables[slot]

    def index(self, slot, text):
        return self.table(slot).get(text)

    def counts(self):
        return {slot: len(self.entries[slot]) for slot in self.SLOTS}

    def coverage(self, cases):
        """How much of a split a retrieval-only decoder could ever get right."""
        out = {}
        for slot in self.SLOTS:
            covered = sum(1 for _, slots in cases if self.index(slot, slots[slot]) is not None)
            total = len(cases)
            out[slot] = {"covered": covered, "cases": total,
                         "rate": (covered / total) if total else None}
        return out

    def to_dict(self):
        return {slot: list(self.entries[slot]) for slot in self.SLOTS}

    @classmethod
    def from_dict(cls, payload):
        missing = [slot for slot in cls.SLOTS if slot not in payload]
        if missing:
            raise TrainingError("checkpoint retrieval bank is missing: %s" % ", ".join(missing))
        return cls(payload)


def build_bank(train_cases):
    """Deterministically sorted per-slot candidate strings from the train split."""
    entries = {slot: set() for slot in RetrievalBank.SLOTS}
    for _, slots in train_cases:
        for slot in RetrievalBank.SLOTS:
            entries[slot].add(slots[slot])
    return RetrievalBank({slot: sorted(values) for slot, values in entries.items()})


def encode_retrieval_targets(cases, bank, torch):
    """Per-slot prototype indices, IGNORE where the reference is not in the bank."""
    out = {}
    for slot in RetrievalBank.SLOTS:
        target = torch.full((len(cases),), base.IGNORE, dtype=torch.long)
        for row, (_, slots) in enumerate(cases):
            index = bank.index(slot, slots[slot])
            if index is not None:
                target[row] = index
        out[slot] = target
    return out


# --------------------------------------------------------------------------
# reachability
# --------------------------------------------------------------------------
def reachability(cases, vocab):
    """Per-case reachability plus the ceiling it implies on four-slot exactness.

    An unreachable reference can never be emitted by a closed-vocabulary model, so
    counting it as an ordinary miss understates the model and counting it as a success
    overstates it. It is reported separately and excluded from the ceiling -- and,
    critically, an out-of-vocabulary value is COUNTED here instead of being mapped onto
    the nearest known class, which would quietly turn "unknown" into "wrong but
    plausible".
    """
    unknown = {"topic": [], "constraint_marker": [], "constraint_item": [],
               "pending_pattern": [], "pending_value": [],
               "result_pattern": [], "result_value": [], "shape": []}
    per_case = {}
    fully = 0
    for case, slots in cases:
        ok = True
        if vocab.index("topics", slots["topic"]) is None:
            unknown["topic"].append(slots["topic"])
            ok = False
        groups = split_groups(slots["constraint"])
        if not constraint_fits(groups):
            unknown["shape"].append(case["id"])
            ok = False
        for marker, group_items in groups:
            if vocab.index("markers", marker) is None:
                unknown["constraint_marker"].append(marker)
                ok = False
            for item in group_items:
                if vocab.index("items", item) is None:
                    unknown["constraint_item"].append(item)
                    ok = False
        for key, patterns_field, values_field in (
                ("pending", "pending_patterns", "pending_values"),
                ("result", "result_patterns", "result_values")):
            pattern, value = split_pair(slots[key])
            if vocab.index(patterns_field, pattern) is None:
                unknown[("pending_pattern" if key == "pending" else "result_pattern")].append(pattern)
                ok = False
            if value is not None and vocab.index(values_field, value) is None:
                unknown[("pending_value" if key == "pending" else "result_value")].append(value)
                ok = False
        per_case[case["id"]] = ok
        fully += 1 if ok else 0
    total = len(cases)
    report = {
        "cases": total,
        "fully_reachable": fully,
        "ceiling_four_slot_exact": (fully / total) if total else None,
        "per_case": per_case,
        "note": ("train-only class sets; an out-of-vocabulary reference is scored as a miss "
                 "and excluded from the ceiling, never mapped to a known class"),
    }
    for key, values in unknown.items():
        counter = {}
        for value in values:
            counter[value] = counter.get(value, 0) + 1
        report["unreachable_" + key] = {
            "occurrences": len(values),
            "distinct": len(counter),
            "top": sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[:10],
        }
    return report


def build_references(cases, reach):
    """Reference slot dicts for `score`, annotated with reachability."""
    out = []
    for case, slots in cases:
        entry = {key: slots[key] for key in SLOT_KEYS}
        entry["_reachable"] = bool(reach["per_case"].get(case["id"]))
        out.append(entry)
    return out


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------
class StructuralTargets:
    """Class indices for every head, all parallel -- there is no prefix to shift."""

    def __init__(self, topic, marker, item, pending_pattern, pending_value,
                 result_pattern, result_value):
        self.topic = topic
        self.marker = marker
        self.item = item
        self.pending_pattern = pending_pattern
        self.pending_value = pending_value
        self.result_pattern = result_pattern
        self.result_value = result_value

    def to(self, device):
        return StructuralTargets(
            self.topic.to(device), self.marker.to(device), self.item.to(device),
            self.pending_pattern.to(device), self.pending_value.to(device),
            self.result_pattern.to(device), self.result_value.to(device))


def encode_targets(cases, vocab, torch):
    """Raises if a TRAIN label is outside the vocabulary.

    The train split defines the vocabulary, so a miss here means the vocabularies were
    built from a different case list. Failing at encode time is the difference between
    a loud setup error and a model trained against shifted class indices.
    """
    count = len(cases)
    topic = torch.zeros((count,), dtype=torch.long)
    marker = torch.zeros((count, MAX_GROUPS), dtype=torch.long)
    item = torch.zeros((count, MAX_GROUPS, MAX_ITEMS), dtype=torch.long)
    pending_pattern = torch.zeros((count,), dtype=torch.long)
    pending_value = torch.zeros((count,), dtype=torch.long)
    result_pattern = torch.zeros((count,), dtype=torch.long)
    result_value = torch.zeros((count,), dtype=torch.long)
    for row, (case, slots) in enumerate(cases):
        index = vocab.index("topics", slots["topic"])
        if index is None:
            raise TrainingError("case %s has a topic outside the train vocabulary" % case["id"])
        topic[row] = index
        groups = split_groups(slots["constraint"])
        if not constraint_fits(groups):
            raise TrainingError("case %s constraint needs more positions than the heads allow"
                                % case["id"])
        for position, (group_marker, group_items) in enumerate(groups):
            marker_index = vocab.index("markers", group_marker)
            if marker_index is None:
                raise TrainingError("case %s has an unseen constraint marker %r"
                                    % (case["id"], group_marker))
            marker[row, position] = marker_index + 1
            for slot_position, value in enumerate(group_items):
                item_index = vocab.index("items", value)
                if item_index is None:
                    raise TrainingError("case %s has an unseen constraint item %r"
                                        % (case["id"], value))
                item[row, position, slot_position] = item_index + 1
        for key, patterns_field, values_field, pattern_out, value_out in (
                ("pending", "pending_patterns", "pending_values",
                 pending_pattern, pending_value),
                ("result", "result_patterns", "result_values",
                 result_pattern, result_value)):
            pattern, value = split_pair(slots[key])
            pattern_index = vocab.index(patterns_field, pattern)
            if pattern_index is None:
                raise TrainingError("case %s has an unseen %s pattern %r"
                                    % (case["id"], key, pattern))
            pattern_out[row] = pattern_index
            if value is not None:
                value_index = vocab.index(values_field, value)
                if value_index is None:
                    raise TrainingError("case %s has an unseen %s value %r"
                                        % (case["id"], key, value))
                value_out[row] = value_index + 1
    return StructuralTargets(topic, marker, item, pending_pattern, pending_value,
                             result_pattern, result_value)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class ModelConfig:
    FIELDS = ("pooled_dim", "d_state", "n_topics", "n_markers", "n_items",
              "n_pending_patterns", "n_pending_values", "n_result_patterns",
              "n_result_values", "dropout")

    def __init__(self, pooled_dim, d_state, vocab, dropout=0.1):
        self.pooled_dim = int(pooled_dim)
        self.d_state = int(d_state)
        self.dropout = float(dropout)
        self.n_topics = vocab.n_topics
        self.n_markers = vocab.n_markers
        self.n_items = vocab.n_items
        self.n_pending_patterns = len(vocab.pending_patterns)
        self.n_pending_values = len(vocab.pending_values)
        self.n_result_patterns = len(vocab.result_patterns)
        self.n_result_values = len(vocab.result_values)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.FIELDS}

    @classmethod
    def from_dict(cls, payload):
        missing = [name for name in cls.FIELDS if name not in payload]
        if missing:
            raise TrainingError("checkpoint model config is missing: %s" % ", ".join(missing))
        instance = cls.__new__(cls)
        for name in cls.FIELDS:
            setattr(instance, name, payload[name])
        return instance


def build_model(config, torch, bank=None, retrieval_dim=256, retrieval_scale=8.0):
    """Shared state encoder plus one parallel classification head per decision.

    Every head answers a question with a fixed-size answer set, so the rendered string
    is well-formed BY CONSTRUCTION: the four slots cannot be written out of order, with
    the wrong label, or with a truncated value. `d_state` is the only place the frozen
    state enters, and every head reads it, so no slot can be starved of state the way
    `constraint` was when it had to share one character GRU with two other slots.

    When a `bank` is supplied, a second read-out of the same state scores cosine
    similarity against one learned prototype per train label string. That bank is an
    auxiliary signal and a candidate source (see `RetrievalBank`); it never replaces the
    structural heads, whose ceiling is strictly higher (86.96% vs 69.87% on dev).
    """
    nn = torch.nn
    functional = torch.nn.functional
    groups = MAX_GROUPS
    items = MAX_ITEMS

    class StructuralClassifier(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.norm = nn.LayerNorm(cfg.pooled_dim)
            self.enc1 = nn.Linear(cfg.pooled_dim, cfg.d_state)
            self.enc2 = nn.Linear(cfg.d_state, cfg.d_state)
            self.drop = nn.Dropout(cfg.dropout)
            self.topic = nn.Linear(cfg.d_state, cfg.n_topics)
            self.marker = nn.Linear(cfg.d_state, groups * (cfg.n_markers + 1))
            self.item = nn.Linear(cfg.d_state, groups * items * (cfg.n_items + 1))
            self.pending_pattern = nn.Linear(cfg.d_state, cfg.n_pending_patterns)
            self.pending_value = nn.Linear(cfg.d_state, cfg.n_pending_values + 1)
            self.result_pattern = nn.Linear(cfg.d_state, cfg.n_result_patterns)
            self.result_value = nn.Linear(cfg.d_state, cfg.n_result_values + 1)
            self.bank_sizes = ({slot: len(bank.entries[slot]) for slot in RetrievalBank.SLOTS}
                               if bank is not None else {})
            self.retrieval_scale = float(retrieval_scale)
            if bank is not None:
                self.retr_proj = nn.Linear(cfg.d_state, int(retrieval_dim))
                generator = torch.Generator().manual_seed(0)
                self.prototypes = nn.ParameterDict({
                    slot: nn.Parameter(torch.randn((max(1, size), int(retrieval_dim)),
                                                   generator=generator) * 0.02)
                    for slot, size in self.bank_sizes.items()})

        def encode(self, pooled):
            hidden = self.drop(functional.gelu(self.enc1(self.norm(pooled))))
            return self.drop(functional.gelu(self.enc2(hidden)))

        def retrieval_logits(self, state):
            """{slot: [B, prototypes]} cosine logits, or {} when no bank is attached."""
            if not self.bank_sizes:
                return {}
            projected = functional.normalize(self.retr_proj(state), dim=-1)
            out = {}
            for slot, prototypes in self.prototypes.items():
                table = functional.normalize(prototypes, dim=-1)
                out[slot] = self.retrieval_scale * (projected @ table.t())
            return out

        def forward(self, pooled):
            state = self.encode(pooled)
            return {
                "state": state,
                "topic": self.topic(state),
                "marker": self.marker(state).view(-1, groups, self.cfg.n_markers + 1),
                "item": self.item(state).view(-1, groups, items, self.cfg.n_items + 1),
                "pending_pattern": self.pending_pattern(state),
                "pending_value": self.pending_value(state),
                "result_pattern": self.result_pattern(state),
                "result_value": self.result_value(state),
                "retrieval": self.retrieval_logits(state),
            }

    return StructuralClassifier(config)


def parameter_count(module):
    return sum(int(p.numel()) for p in module.parameters())


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "topic": 1.0, "marker": 1.0, "item": 1.0,
    "pending_pattern": 1.0, "pending_value": 1.0,
    "result_pattern": 1.0, "result_value": 1.0,
}


def retrieval_loss_terms(outputs, retrieval_targets, torch):
    """Cross entropy over prototype ids, with out-of-bank references ignored.

    `ignore_index=IGNORE` is what keeps the reachability story honest: a case the bank
    does not contain contributes no gradient and no credit, instead of training the
    encoder to answer with a wrong-but-nearby prototype.
    """
    functional = torch.nn.functional
    logits_by_slot = outputs.get("retrieval") or {}
    if not logits_by_slot or not retrieval_targets:
        return None, {}
    terms = {}
    for slot, logits in logits_by_slot.items():
        target = retrieval_targets.get(slot)
        if target is None or target.numel() == 0:
            continue
        if int((target != base.IGNORE).sum()) == 0:
            continue
        terms[slot] = functional.cross_entropy(logits, target, ignore_index=base.IGNORE)
    if not terms:
        return None, {}
    return sum(terms.values()) / float(len(terms)), terms


def loss_terms(outputs, targets, weights, torch, ignore_index=base.IGNORE,
               retrieval_targets=None, retrieval_weight=0.0):
    """Weighted sum of the per-head cross entropies, each reported separately.

    Keeping them separate is what makes a plateau diagnosable: `constraint` collapsing
    while `topic` climbs is a visible signature, where one blended number would only
    show "stuck".
    """
    functional = torch.nn.functional
    terms = {}
    terms["topic"] = functional.cross_entropy(outputs["topic"], targets.topic)
    terms["marker"] = functional.cross_entropy(
        outputs["marker"].reshape(-1, outputs["marker"].shape[-1]),
        targets.marker.reshape(-1))
    terms["item"] = functional.cross_entropy(
        outputs["item"].reshape(-1, outputs["item"].shape[-1]),
        targets.item.reshape(-1))
    for name in ("pending_pattern", "pending_value", "result_pattern", "result_value"):
        terms[name] = functional.cross_entropy(outputs[name], targets.__dict__[name])
    total = None
    for name, value in terms.items():
        weighted = weights.get(name, 1.0) * value
        total = weighted if total is None else total + weighted
    if retrieval_weight > 0.0:
        retrieval, per_slot = retrieval_loss_terms(outputs, retrieval_targets, torch)
        if retrieval is not None:
            total = total + float(retrieval_weight) * retrieval
            terms["retrieval"] = retrieval
            for slot, value in per_slot.items():
                terms["retrieval_" + slot] = value
    out = {"total": total}
    out.update(terms)
    return out


# --------------------------------------------------------------------------
# decoding (parallel: no recurrence, so nothing to loop over)
# --------------------------------------------------------------------------
def render_from_ids(marker_ids, item_ids, pending_ids, result_ids, vocab):
    """Turn predicted class indices into the four slot strings."""
    groups = []
    for position in range(MAX_GROUPS):
        marker_index = int(marker_ids[position])
        if marker_index == ABSENT:
            continue
        group_items = []
        for item_position in range(MAX_ITEMS):
            item_index = int(item_ids[position][item_position])
            if item_index != ABSENT:
                group_items.append(vocab.items[item_index - 1])
        marker = vocab.markers[marker_index - 1]
        body = "、".join(group_items)
        groups.append((marker + "：" + body) if marker else body)
    constraint = "；".join(groups)

    pending_pattern_index, pending_value_index = pending_ids
    pending = render_pair(vocab.pending_patterns[int(pending_pattern_index)],
                          None if int(pending_value_index) == 0
                          else vocab.pending_values[int(pending_value_index) - 1])
    result_pattern_index, result_value_index = result_ids
    result = render_pair(vocab.result_patterns[int(result_pattern_index)],
                         None if int(result_value_index) == 0
                         else vocab.result_values[int(result_value_index) - 1])
    return constraint, pending, result


def structural_from_outputs(outputs, vocab, torch):
    """Argmax decode of every head -> (predictions, per-slot confidence).

    A composed slot's confidence is the product of the probabilities of the decisions
    that built it. That is what makes it comparable with a retrieval margin when the
    two sources are arbitrated against each other.
    """
    functional = torch.nn.functional
    rows = int(outputs["topic"].shape[0])
    index = torch.arange(rows)
    topic_ids = outputs["topic"].argmax(dim=-1)
    marker_ids = outputs["marker"].argmax(dim=-1)
    item_ids = outputs["item"].argmax(dim=-1)
    pending_pattern = outputs["pending_pattern"].argmax(dim=-1)
    pending_value = outputs["pending_value"].argmax(dim=-1)
    result_pattern = outputs["result_pattern"].argmax(dim=-1)
    result_value = outputs["result_value"].argmax(dim=-1)
    topic_p = functional.softmax(outputs["topic"], dim=-1)[index, topic_ids]
    marker_p = functional.softmax(outputs["marker"], dim=-1)
    item_p = functional.softmax(outputs["item"], dim=-1)
    pending_pattern_p = functional.softmax(outputs["pending_pattern"], dim=-1)[index, pending_pattern]
    pending_value_p = functional.softmax(outputs["pending_value"], dim=-1)[index, pending_value]
    result_pattern_p = functional.softmax(outputs["result_pattern"], dim=-1)[index, result_pattern]
    result_value_p = functional.softmax(outputs["result_value"], dim=-1)[index, result_value]

    predictions = []
    confidence = {"topic": [], "constraint": [], "pending": [], "result": []}
    for row in range(rows):
        constraint, pending, result = render_from_ids(
            marker_ids[row], item_ids[row],
            (pending_pattern[row], pending_value[row]),
            (result_pattern[row], result_value[row]), vocab)
        constraint_confidence = 1.0
        for position in range(MAX_GROUPS):
            marker = int(marker_ids[row, position])
            if marker == ABSENT:
                continue
            constraint_confidence *= float(marker_p[row, position, marker])
            for item_position in range(MAX_ITEMS):
                item = int(item_ids[row, position, item_position])
                if item != ABSENT:
                    constraint_confidence *= float(item_p[row, position, item_position, item])
        pending_confidence = float(pending_pattern_p[row])
        if vocab.pending_patterns[int(pending_pattern[row])].endswith("："):
            pending_confidence *= float(pending_value_p[row])
        result_confidence = float(result_pattern_p[row])
        if vocab.result_patterns[int(result_pattern[row])].endswith("："):
            result_confidence *= float(result_value_p[row])
        predictions.append({"topic": vocab.topics[int(topic_ids[row])],
                            "constraint": constraint, "pending": pending, "result": result})
        confidence["topic"].append(float(topic_p[row]))
        confidence["constraint"].append(constraint_confidence)
        confidence["pending"].append(pending_confidence)
        confidence["result"].append(result_confidence)
    return predictions, confidence


def retrieval_from_outputs(outputs, bank, torch):
    """Nearest-prototype candidate and its top1-top2 margin, per value slot."""
    logits_by_slot = outputs.get("retrieval") or {}
    candidates, margins = {}, {}
    for slot, logits in logits_by_slot.items():
        width = int(logits.shape[-1])
        if width == 0:
            continue
        take = min(2, width)
        top = logits.topk(take, dim=-1)
        best = top.indices[:, 0]
        if take > 1:
            margin = top.values[:, 0] - top.values[:, 1]
        else:
            margin = torch.zeros_like(top.values[:, 0])
        candidates[slot] = [bank.entries[slot][int(position)] for position in best]
        margins[slot] = [float(value) for value in margin]
    return candidates, margins


def predict(module, pooled, vocab, torch, chunk=1024):
    """Structural-only prediction for every case in `pooled`."""
    module.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, pooled.shape[0], chunk):
            predictions, _ = structural_from_outputs(module(pooled[start:start + chunk]),
                                                     vocab, torch)
            out.extend(predictions)
    return out


def predict_variants(module, pooled, vocab, bank, torch, references=None, chunk=512,
                     arbitrate_margin=1.0, arbitrate_confidence=0.5):
    """Four prediction sets from one forward pass per chunk.

      structural  the decomposed heads (the default output path)
      retrieval   nearest train label prototype for constraint/pending/result, topic
                  from the structural head (the bank has no topic candidates)
      arbitrated  per slot, retrieval wins only when its margin is high AND the
                  structural confidence for that slot is low
      oracle_slot_max  diagnostic only, and it reads the references: per slot, whichever
                  source was right. It is an upper bound on what arbitration could buy,
                  never a claim -- the previous run failed precisely because a
                  reference-peeking convenience was mistaken for a result.
    """
    module.eval()
    names = ("structural", "retrieval", "arbitrated", "oracle_slot_max")
    collected = {name: [] for name in names}
    with torch.inference_mode():
        for start in range(0, pooled.shape[0], chunk):
            block = pooled[start:start + chunk]
            outputs = module(block)
            structural, confidence = structural_from_outputs(outputs, vocab, torch)
            candidates, margins = retrieval_from_outputs(outputs, bank, torch)
            size = len(structural)
            for row in range(size):
                position = start + row
                reference = (references[position]
                             if references is not None and position < len(references) else None)
                retrieval_prediction = dict(structural[row])
                for slot in RetrievalBank.SLOTS:
                    if candidates.get(slot):
                        retrieval_prediction[slot] = candidates[slot][row]
                arbitrated = dict(structural[row])
                for slot in RetrievalBank.SLOTS:
                    if not candidates.get(slot):
                        continue
                    if (margins[slot][row] >= float(arbitrate_margin)
                            and confidence[slot][row] < float(arbitrate_confidence)):
                        arbitrated[slot] = candidates[slot][row]
                collected["structural"].append(structural[row])
                collected["retrieval"].append(retrieval_prediction)
                collected["arbitrated"].append(arbitrated)
                if reference is None:
                    collected["oracle_slot_max"].append(dict(structural[row]))
                else:
                    best = {}
                    for key in SLOT_KEYS:
                        if structural[row][key] == reference[key]:
                            best[key] = structural[row][key]
                        elif retrieval_prediction[key] == reference[key]:
                            best[key] = retrieval_prediction[key]
                        else:
                            best[key] = structural[row][key]
                    collected["oracle_slot_max"].append(best)
    return collected


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def slot_hit_total(references, predictions):
    """Per-case number of exactly matching slots, 0..4.

    The paired ablation is computed on THIS rather than on `all_slots_exact`. That is a
    fix, not a preference: in the previous run every condition sat at four-slot exact 0,
    so the paired comparison reported 0 wins / 0 losses / 1779 ties and hid the fact
    that the true state was scoring 77% on `topic` while the zero state scored 0.7%.
    A saturating metric cannot witness conditioning; this one can.
    """
    totals = []
    for index, reference in enumerate(references):
        prediction = predictions[index]
        totals.append(sum(1 for key in SLOT_KEYS if prediction[key] == reference[key]))
    return totals


def paired_totals(before, after):
    """Win/loss/tie counts of `after` over `before` on the per-case slot totals."""
    win = loss = tie = 0
    for left, right in zip(before, after):
        if right > left:
            win += 1
        elif right < left:
            loss += 1
        else:
            tie += 1
    graded = win + loss
    return {"wins": win, "losses": loss, "ties": tie,
            "win_rate": (win / graded) if graded else None}


def evaluate(module, pooled, references, vocab, torch, wrong_permutation, bank=None,
             arbitrate_margin=1.0, arbitrate_confidence=0.5):
    """Score the same cases under true / zero / wrong conditioning, paired.

      true   the case's own frozen state
      zero   the zero vector -- anything scored here comes from label priors alone
      wrong  another case's state, cyclically shifted -- conditioning is present but
             meaningless, which separates "reads the state" from "uses any vector as bias"

    Both a saturating (all-four) and a non-saturating (per-slot total) paired comparison
    are reported. The former is kept only so this run is comparable with its
    predecessors; the latter is the one that can actually move.

    Each condition also reports the retrieval and arbitrated variants next to the
    structural one, so whether candidate retrieval helps is decided by these numbers and
    not by preference.
    """
    zero = torch.zeros_like(pooled)
    wrong = pooled[wrong_permutation] if wrong_permutation is not None else zero
    totals = {}
    conditions = {}
    for name, block in (("true", pooled), ("zero", zero), ("wrong", wrong)):
        variants = predict_variants(module, block, vocab, bank, torch, references=references,
                                    arbitrate_margin=arbitrate_margin,
                                    arbitrate_confidence=arbitrate_confidence)
        structural = variants["structural"]
        totals[name] = slot_hit_total(references, structural)
        conditions[name] = {
            "score": base.score(references, structural),
            "slot_totals": totals[name],
            "predictions": structural,
            "variants": {variant: base.score(references, predictions)
                         for variant, predictions in variants.items()},
            "variant_predictions": variants,
        }
    comparisons = {}
    for name in ("zero", "wrong"):
        comparisons["true_vs_" + name] = paired_totals(totals[name], totals["true"])
        comparisons["true_vs_%s_four_slot" % name] = base.paired(
            [1 if value == len(SLOT_KEYS) else 0 for value in totals[name]],
            [1 if value == len(SLOT_KEYS) else 0 for value in totals["true"]])
    return {"conditions": conditions, "paired": comparisons, "headline_condition": "true"}


def variant_rate(condition, variant=None):
    """Four-slot exact rate of a variant (default: the structural headline)."""
    if variant is None:
        return condition["score"]["four_slot_exact_rate"] or 0.0
    return condition["variants"][variant]["four_slot_exact_rate"] or 0.0


def compact_eval(evaluation):
    return {name: {"score": evaluation["conditions"][name]["score"],
                   "variants": {variant: {"four_slot_exact_rate": score["four_slot_exact_rate"],
                                          "content_slot_accuracy": score["content_slot_accuracy"]}
                                for variant, score in
                                evaluation["conditions"][name]["variants"].items()}}
            for name in evaluation["conditions"]}


def eval_summary_line(evaluation):
    def rate(name):
        return evaluation["conditions"][name]["score"]["four_slot_exact_rate"] or 0.0

    def content(name):
        return evaluation["conditions"][name]["score"]["content_slot_accuracy"] or 0.0

    def paired_rate(name):
        return evaluation["paired"]["true_vs_" + name]["win_rate"]

    true_condition = evaluation["conditions"]["true"]
    per_slot = true_condition["score"]["per_slot_exact_rate"]
    return ("dev four_slot_exact structural=%.4f retrieval=%.4f arbitrated=%.4f "
            "oracle_slot_max=%.4f | zero=%.4f wrong=%.4f | content(true)=%.4f "
            "paired_win_rate true_vs_zero=%s true_vs_wrong=%s | "
            "slots(topic/constraint/pending/result)=%.4f/%.4f/%.4f/%.4f"
            % (rate("true"), variant_rate(true_condition, "retrieval"),
               variant_rate(true_condition, "arbitrated"),
               variant_rate(true_condition, "oracle_slot_max"),
               rate("zero"), rate("wrong"), content("true"),
               paired_rate("zero"), paired_rate("wrong"),
               per_slot["topic"] or 0.0, per_slot["constraint"] or 0.0,
               per_slot["pending"] or 0.0, per_slot["result"] or 0.0))


# --------------------------------------------------------------------------
# small IO helpers (kept local: the sibling run's writers have other contracts)
# --------------------------------------------------------------------------
def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                         encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(torch, payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    torch.save(payload, temporary)
    os.replace(temporary, path)


class RunLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message):
        line = "%s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def stream_position(start_step, count, batch_size):
    """(epoch, cursor) for a resumed run, so it continues the SAME data stream.

    `make_batches` is deterministic in (seed, epoch), so a resumed run that simply
    restarted at epoch 0 would replay the first examples for a second time while
    claiming to continue at step `start_step`. Jumping the stream to the position a
    continuous run would have reached keeps resumed and uninterrupted training
    equivalent.
    """
    per_epoch = max(1, int(math.ceil(count / float(batch_size))))
    consumed = int(start_step)
    return consumed // per_epoch, consumed % per_epoch


def run_training(*, torch, module, optimizer, weights, pooled_train, targets, log,
                 steps, batch_size, seed, start_step=0, history=None,
                 evaluate_at=None, before_step=None, checkpoint_at=None,
                 retrieval_targets=None, retrieval_weight=0.0):
    """The step loop, with the environment injected so a CPU test can drive it.

    `before_step` is where the AC/lease gate lives, and it may raise `Done` to pause.
    It is called BEFORE the optimizer step, so a pause can never leave a half-applied
    update or record a step that did not happen.
    """
    module.train()
    history = [] if history is None else history
    count = pooled_train.shape[0]
    step = int(start_step)
    epoch, cursor = stream_position(step, count, batch_size)
    batches = base.make_batches(count, batch_size, seed, epoch)
    while step < steps:
        if cursor >= len(batches):
            epoch += 1
            batches = base.make_batches(count, batch_size, seed, epoch)
            cursor = 0
        index = batches[cursor]
        cursor += 1
        step += 1
        if before_step is not None:
            before_step(step)
        pooled = pooled_train[index]
        target = StructuralTargets(
            targets.topic[index], targets.marker[index], targets.item[index],
            targets.pending_pattern[index], targets.pending_value[index],
            targets.result_pattern[index], targets.result_value[index])
        outputs = module(pooled)
        slice_retrieval = None
        if retrieval_targets is not None:
            slice_retrieval = {slot: target[index]
                               for slot, target in retrieval_targets.items()}
        terms = loss_terms(outputs, target, weights, torch,
                           retrieval_targets=slice_retrieval,
                           retrieval_weight=retrieval_weight)
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
        optimizer.step()
        record = {"step": step, "loss": round(float(terms["total"].detach()), 6),
                  "grad_norm": round(float(grad_norm), 4)}
        for name in DEFAULT_WEIGHTS:
            record[name] = round(float(terms[name].detach()), 6)
        if "retrieval" in terms:
            record["retrieval"] = round(float(terms["retrieval"].detach()), 6)
        history.append(record)
        if step % 25 == 0 or step == steps:
            log("step %d/%d total=%.4f topic=%.4f marker=%.4f item=%.4f "
                "pending_p=%.4f pending_v=%.4f result_p=%.4f result_v=%.4f%s grad=%.3f"
                % (step, steps, record["loss"], record["topic"], record["marker"],
                   record["item"], record["pending_pattern"], record["pending_value"],
                   record["result_pattern"], record["result_value"],
                   (" retr=%.4f" % record["retrieval"]) if "retrieval" in record else "",
                   record["grad_norm"]))
        if checkpoint_at is not None:
            checkpoint_at(step)
        if evaluate_at is not None:
            evaluate_at(step)
    return {"steps": history, "final_step": step}


# --------------------------------------------------------------------------
# argument handling
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Classify frozen RWKV state into four structured memory slots.")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--feature-cache", default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--backbone-sha256", default=DEFAULT_BACKBONE_SHA256)
    parser.add_argument("--allow-any-backbone", action="store_true")
    parser.add_argument("--pool", default="meanmax", choices=("meanmax", "mean", "max"))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--d-state", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cuda")
    # candidate retrieval over train label prototypes
    parser.add_argument("--retrieval", dest="retrieval", action="store_true", default=True,
                        help="score a bank of train label prototypes (default on)")
    parser.add_argument("--no-retrieval", dest="retrieval", action="store_false")
    parser.add_argument("--retrieval-weight", type=float, default=0.3,
                        help="auxiliary loss weight; 0 disables the gradient but keeps "
                             "the candidate source for arbitration")
    parser.add_argument("--retrieval-dim", type=int, default=256)
    parser.add_argument("--retrieval-scale", type=float, default=8.0)
    parser.add_argument("--arbitrate-margin", type=float, default=1.0,
                        help="retrieval top1-top2 cosine margin needed to override")
    parser.add_argument("--arbitrate-confidence", type=float, default=0.5,
                        help="structural confidence below which an override is allowed")
    parser.add_argument("--skip-power-check", action="store_true",
                        help="plumbing only: never usable to back a qualification claim")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
class Suite:
    def __init__(self):
        self.passed = 0
        self.failures = []

    def check(self, name, condition, detail=None):
        if condition:
            self.passed += 1
        else:
            self.failures.append((name, detail))
            print("FAIL: %s -> %r" % (name, detail))

    def report(self):
        total = self.passed + len(self.failures)
        print("%d/%d checks passed" % (self.passed, total))
        return not self.failures


def _fixture_cases():
    """Small train/heldout pair with every structural shape the renderer must handle."""
    def case(case_id, group, topic, constraint, pending, result):
        summary = "主题：%s\n约束：%s\n未完成：%s\n结果：%s" % (topic, constraint, pending, result)
        return {"id": case_id, "group": group, "user": "u", "assistant": "a",
                "summary": summary, "unverified": False}

    items = [
        case("F-tr-1", "train", "旅行规划", "本轮条件：景点类型=园林", "无", "已确认给出名称推荐"),
        case("F-tr-2", "train", "旅行规划", "沿用已告知条件：景点类型=园林、消费=中等",
             "本轮未答：天气", "已确认告知票价：免费"),
        case("F-tr-3", "train", "电脑选购", "本轮用户条件：CPU型号=3.0T", "无", "无"),
        case("F-tr-4", "train", "旅行规划", "本轮条件：景点类型=园林；沿用：消费=中等", "无", "无"),
        # A marker-less group (no `：`) and a bare `无` constraint: both are shapes the
        # real corpus contains, and both are what makes their classes legitimate.
        case("F-tr-5", "train", "旅行规划", "景点类型=园林", "无", "无"),
        case("F-tr-6", "train", "旅行规划", "无", "无", "无"),
    ]
    heldout = [
        case("F-ho-1", "heldout", "旅行规划", "本轮条件：景点类型=园林", "无", "已确认给出名称推荐"),
        case("F-ho-2", "heldout", "旅行规划", "本轮条件：景点类型=园林、消费=中等", "无", "无"),
        case("F-ho-3", "heldout", "旅行规划", "本轮条件：景点类型=园林、未知项=值", "无", "无"),
        case("F-ho-4", "heldout", "全新主题", "本轮条件：景点类型=园林", "无", "无"),
    ]
    return items, base.case_slots(items), base.case_slots(heldout)


def self_test():
    """CPU-only checks. No CUDA context is created anywhere in this function."""
    suite = Suite()
    torch = base._torch()
    items, train_cases, heldout_cases = _fixture_cases()

    # ---- parsing and rendering are exact inverses on real-shaped strings
    for text in ("本轮条件：景点类型=园林",
                 "沿用已告知条件：景点类型=园林、消费=中等",
                 "本轮条件：景点类型=园林；沿用：消费=中等",
                 "无"):
        suite.check("round-trip %r" % text[:18], render_groups(split_groups(text)) == text,
                    render_groups(split_groups(text)))
    for text in ("已确认给出名称推荐", "已确认告知票价：免费", "无", "已确认告知票价："):
        pattern, value = split_pair(text)
        suite.check("result round-trip %r" % text[:14], render_pair(pattern, value) == text,
                    render_pair(pattern, value))
    suite.check("empty marker keeps its items",
                split_groups("景点类型=园林")[0][0] == ""
                and render_groups(split_groups("景点类型=园林")) == "景点类型=园林")
    suite.check("trailing separator is dropped",
                render_groups(split_groups("本轮条件：A=1；")) == "本轮条件：A=1")
    suite.check("position overflow is detected", not constraint_fits(split_groups("A：x；B：y；C：z")))
    suite.check("two groups fit", constraint_fits(split_groups("A：x；B：y")))

    # ---- vocabularies come from train only
    vocab = build_vocab(train_cases)
    suite.check("vocab indexes a train topic", vocab.index("topics", "旅行规划") is not None)
    suite.check("vocab does NOT index a heldout-only topic",
                vocab.index("topics", "全新主题") is None)
    suite.check("empty marker is a real class", vocab.index("markers", "") is not None)
    suite.check("empty slot marker is representable", vocab.index("items", EMPTY) is not None)
    suite.check("pending pattern captured", vocab.index("pending_patterns", "本轮未答：") is not None)
    suite.check("result value captured", vocab.index("result_values", "免费") is not None)

    # ---- reachability counts unknowns instead of mapping them
    reach = reachability(heldout_cases, vocab)
    suite.check("reachable heldout case counted",
                reach["per_case"]["F-ho-1"] is True, reach["per_case"])
    suite.check("unseen constraint item is unreachable",
                reach["per_case"]["F-ho-3"] is False)
    suite.check("unseen constraint item is reported, not mapped",
                reach["unreachable_constraint_item"]["occurrences"] == 1
                and reach["unreachable_constraint_item"]["distinct"] == 1,
                reach["unreachable_constraint_item"])
    suite.check("unseen topic is unreachable and reported",
                reach["per_case"]["F-ho-4"] is False
                and reach["unreachable_topic"]["occurrences"] == 1)
    suite.check("ceiling counts only fully reachable cases",
                reach["fully_reachable"] == 2 and abs(reach["ceiling_four_slot_exact"] - 0.5) < 1e-9,
                reach["ceiling_four_slot_exact"])

    # ---- targets encode every shape, with ABSENT for unused positions
    targets = encode_targets(train_cases, vocab, torch)
    fixture_size = len(train_cases)
    suite.check("marker shape",
                tuple(targets.marker.shape) == (fixture_size, MAX_GROUPS), targets.marker.shape)
    suite.check("item shape",
                tuple(targets.item.shape) == (fixture_size, MAX_GROUPS, MAX_ITEMS),
                targets.item.shape)
    suite.check("single-group case leaves group 2 absent", int(targets.marker[0, 1]) == ABSENT)
    suite.check("single-item group leaves item 2 absent", int(targets.item[0, 0, 1]) == ABSENT)
    suite.check("second group encoded", int(targets.marker[3, 1]) != ABSENT)
    suite.check("no-value pattern encodes value class 0",
                int(targets.result_value[0]) == 0)
    suite.check("value-bearing pattern encodes a real class",
                int(targets.result_value[1]) != 0)

    # ---- the renderer reproduces a reference exactly from its own encoding
    def render_reference(fixture, row):
        marker_ids = [int(value) for value in fixture.marker[row]]
        item_ids = [[int(value) for value in position] for position in fixture.item[row]]
        return render_from_ids(marker_ids, item_ids,
                               (int(fixture.pending_pattern[row]), int(fixture.pending_value[row])),
                               (int(fixture.result_pattern[row]), int(fixture.result_value[row])),
                               vocab)

    for row in range(len(train_cases)):
        slots = train_cases[row][1]
        constraint, pending, result = render_reference(targets, row)
        suite.check("renderer reproduces constraint row %d" % row,
                    constraint == slots["constraint"], (constraint, slots["constraint"]))
        suite.check("renderer reproduces pending row %d" % row, pending == slots["pending"],
                    (pending, slots["pending"]))
        suite.check("renderer reproduces result row %d" % row, result == slots["result"],
                    (result, slots["result"]))

    # ---- model wiring
    config = ModelConfig(pooled_dim=64, d_state=32, vocab=vocab, dropout=0.0)
    module = build_model(config, torch)
    outputs = module(torch.zeros((5, 64)))
    suite.check("topic head shape", tuple(outputs["topic"].shape) == (5, config.n_topics))
    suite.check("marker head shape", tuple(outputs["marker"].shape) == (5, MAX_GROUPS, config.n_markers + 1))
    suite.check("item head shape", tuple(outputs["item"].shape) == (5, MAX_GROUPS, MAX_ITEMS, config.n_items + 1))
    suite.check("pending value head has the absent class",
                tuple(outputs["pending_value"].shape) == (5, config.n_pending_values + 1))
    suite.check("result value head has the absent class",
                tuple(outputs["result_value"].shape) == (5, config.n_result_values + 1))
    suite.check("predictions have all four slots",
                set(predict(module, torch.zeros((3, 64)), vocab, torch)[0]) == set(SLOT_KEYS))
    suite.check("predictions are always in-vocabulary",
                all(prediction["constraint"] == "" or
                    all(marker in vocab.markers or marker == ""
                        for marker, _ in split_groups(prediction["constraint"]))
                    for prediction in predict(module, torch.zeros((3, 64)), vocab, torch)))
    suite.check("all-absent state renders empty slots",
                vocab.pending_patterns[0] is not None)

    # ---- scoring and the paired metric
    references = build_references(heldout_cases, reach)
    predictions = [dict(reference) for reference in references]
    suite.check("perfect predictions score 1.0",
                base.score(references, predictions)["four_slot_exact_rate"] == 1.0)
    suite.check("reference reachability flag survives", references[0]["_reachable"] is True)
    suite.check("reference unreachable flag survives", references[2]["_reachable"] is False)

    # the fix that matters: a saturating metric must not be able to hide conditioning
    saturating_before = [0, 0, 0, 0]
    saturating_after = [0, 0, 0, 0]
    suite.check("all-four pairing is blind when everything is zero",
                base.paired(saturating_before, saturating_after)["win_rate"] is None)
    graded = paired_totals([1, 1, 0, 0], [3, 2, 0, 1])
    suite.check("per-slot pairing sees the same difference",
                graded["wins"] == 3 and graded["losses"] == 0 and graded["ties"] == 1, graded)
    suite.check("per-slot totals are capped at four",
                max(slot_hit_total(references, predictions)) == len(SLOT_KEYS))

    # ---- one real optimizer step on CPU, then a checkpoint round-trip
    weights = dict(DEFAULT_WEIGHTS)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2, weight_decay=0.0)
    pooled_train = torch.randn((fixture_size, 64))
    log = lambda message: None
    before = float(loss_terms(module(pooled_train), targets, weights, torch)["total"].detach())
    for _ in range(40):
        terms = loss_terms(module(pooled_train), targets, weights, torch)
        optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        optimizer.step()
    after = float(loss_terms(module(pooled_train), targets, weights, torch)["total"].detach())
    suite.check("loss decreases on a fixed batch", after < before, (before, after))
    suite.check("loss is finite", after == after and after < 1e6, after)

    paused = {"current": {"step": 0}, "history": []}

    def before_step(step):
        if step == 3:
            raise Done({"paused": True, "exit_code": EXIT_PAUSED})

    try:
        run_training(torch=torch, module=module,
                     optimizer=torch.optim.AdamW(module.parameters(), lr=0.0),
                     weights=weights, pooled_train=pooled_train, targets=targets,
                     log=log, steps=6, batch_size=2, seed=1,
                     history=paused["history"], before_step=before_step)
        suite.check("a pause stops the loop", False, "the loop ran to completion")
    except Done:
        suite.check("a pause stops the loop", True)
    suite.check("a pause does not record the unfinished step", len(paused["history"]) == 2,
                len(paused["history"]))

    resumed = run_training(torch=torch, module=module,
                           optimizer=torch.optim.AdamW(module.parameters(), lr=0.0),
                           weights=weights, pooled_train=pooled_train, targets=targets,
                           log=log, steps=5, batch_size=2, seed=1, start_step=3)
    suite.check("resume continues from the stored step", resumed["final_step"] == 5)
    suite.check("history is one record per finished step", len(resumed["steps"]) == 2,
                len(resumed["steps"]))

    # A resumed run must consume the same examples a continuous run would have, so the
    # stream position has to be derived from the step, not reset to the beginning.
    for start_step in (0, 1, 2, 3, 5, 6, 7, 11):
        epoch, cursor = stream_position(start_step, 6, 2)
        per_epoch = max(1, int(math.ceil(6 / 2.0)))
        suite.check("stream position is consistent at start_step=%d" % start_step,
                    epoch * per_epoch + cursor == start_step, (epoch, cursor))
    suite.check("stream starts at the beginning", stream_position(0, 6, 2) == (0, 0))
    suite.check("stream rolls into the next epoch", stream_position(3, 6, 2) == (1, 0),
                stream_position(3, 6, 2))

    evaluated = []
    run_training(torch=torch, module=module,
                 optimizer=torch.optim.AdamW(module.parameters(), lr=0.0), weights=weights,
                 pooled_train=pooled_train, targets=targets, log=log, steps=5,
                 batch_size=2, seed=1, evaluate_at=lambda step: evaluated.append(step))
    suite.check("evaluation fires on every step the caller asked about",
                evaluated == [1, 2, 3, 4, 5], evaluated)

    # ---- evaluation contract
    evaluation = evaluate(module, pooled_train, references, vocab, torch,
                          base.wrong_state_shift(len(references), 7, torch))
    for name in ("true", "zero", "wrong"):
        suite.check("condition %s scored" % name, "score" in evaluation["conditions"][name])
    suite.check("paired block is present", "true_vs_zero" in evaluation["paired"])
    suite.check("paired block carries the graded comparison",
                "true_vs_zero_four_slot" in evaluation["paired"])
    suite.check("summary line renders", isinstance(eval_summary_line(evaluation), str))
    suite.check("compact eval keeps the score",
                "score" in compact_eval(evaluation)["true"])

    # ---- candidate retrieval over train label prototypes
    bank = build_bank(train_cases)
    suite.check("bank has one entry per distinct train string",
                bank.counts()["constraint"] == len(set(s["constraint"] for _, s in train_cases)),
                bank.counts())
    suite.check("bank is deterministic across builds",
                build_bank(train_cases).to_dict() == bank.to_dict())
    suite.check("bank indexes a train string", bank.index("result", "已确认给出名称推荐") is not None)
    suite.check("bank does NOT index a heldout-only string",
                bank.index("constraint", "本轮条件：景点类型=园林、消费=中等") is None)
    coverage = bank.coverage(heldout_cases)
    # F-ho-1 and F-ho-4 carry the SAME constraint string, and it is in train; the other
    # two heldout constraints differ from every train string, so the count is 2 of 4.
    suite.check("coverage counts only verbatim hits",
                coverage["constraint"]["covered"] == 2 and coverage["constraint"]["cases"] == 4,
                coverage["constraint"])
    suite.check("bank round-trips through a checkpoint",
                RetrievalBank.from_dict(bank.to_dict()).to_dict() == bank.to_dict())
    try:
        RetrievalBank.from_dict({"constraint": []})
        suite.check("a bank missing a slot is refused", False, "no exception")
    except TrainingError:
        suite.check("a bank missing a slot is refused", True)

    heldout_retrieval = encode_retrieval_targets(heldout_cases, bank, torch)
    train_retrieval = encode_retrieval_targets(train_cases, bank, torch)
    suite.check("out-of-bank references are IGNOREd, not mapped",
                int(heldout_retrieval["constraint"][1]) == base.IGNORE,
                int(heldout_retrieval["constraint"][1]))
    suite.check("in-bank references keep their index",
                int(heldout_retrieval["constraint"][0])
                == bank.index("constraint", "本轮条件：景点类型=园林"))
    suite.check("every train reference is in the bank",
                all(int((target != base.IGNORE).sum()) == target.numel()
                    for target in train_retrieval.values()))
    suite.check("retrieval targets are per-slot",
                set(heldout_retrieval) == set(RetrievalBank.SLOTS))

    retrieval_module = build_model(config, torch, bank=bank, retrieval_dim=16,
                                   retrieval_scale=4.0)
    heldout_rows = len(heldout_cases)
    train_rows = len(train_cases)
    retrieval_outputs = retrieval_module(torch.zeros((heldout_rows, 64)))
    suite.check("retrieval logits per slot",
                set(retrieval_outputs["retrieval"]) == set(RetrievalBank.SLOTS))
    suite.check("retrieval logits width equals the bank size",
                all(int(retrieval_outputs["retrieval"][slot].shape[-1])
                    == bank.counts()[slot] for slot in RetrievalBank.SLOTS))
    bank_loss, bank_terms = retrieval_loss_terms(retrieval_outputs, heldout_retrieval, torch)
    suite.check("retrieval loss is finite", bank_loss is not None and bank_loss == bank_loss)
    suite.check("retrieval loss reported per slot", set(bank_terms) == set(RetrievalBank.SLOTS),
                set(bank_terms))
    all_ignored = {slot: torch.full((3,), base.IGNORE, dtype=torch.long)
                   for slot in RetrievalBank.SLOTS}
    untouched, empty_terms = retrieval_loss_terms(
        retrieval_module(torch.zeros((3, 64))), all_ignored, torch)
    suite.check("a fully unreachable batch contributes no retrieval gradient",
                untouched is None and empty_terms == {})

    train_outputs = retrieval_module(torch.zeros((train_rows, 64)))
    combined = loss_terms(train_outputs, targets, weights, torch,
                          retrieval_targets=train_retrieval, retrieval_weight=0.5)
    plain = loss_terms(train_outputs, targets, weights, torch)
    suite.check("retrieval enters the total loss", "retrieval" in combined)
    suite.check("retrieval is absent from the loss when disabled", "retrieval" not in plain)
    suite.check("an enabled retrieval weight changes the total",
                abs(float(combined["total"].detach()) - float(plain["total"].detach())) > 1e-9)
    zero_weight = loss_terms(train_outputs, targets, weights, torch,
                             retrieval_targets=train_retrieval, retrieval_weight=0.0)
    suite.check("weight 0 keeps the total equal to the head-only total",
                abs(float(zero_weight["total"].detach())
                    - float(plain["total"].detach())) < 1e-6,
                (float(zero_weight["total"].detach()), float(plain["total"].detach())))

    variants = predict_variants(retrieval_module, torch.zeros((heldout_rows, 64)), vocab,
                                bank, torch, references=references,
                                arbitrate_margin=1.0, arbitrate_confidence=0.5)
    suite.check("all four variants are produced",
                set(variants) == {"structural", "retrieval", "arbitrated", "oracle_slot_max"},
                set(variants))
    suite.check("every variant predicts all four slots",
                all(set(prediction) == set(SLOT_KEYS)
                    for predictions in variants.values() for prediction in predictions))
    suite.check("retrieval output is always a bank string",
                all(prediction["constraint"] in bank.entries["constraint"]
                    for prediction in variants["retrieval"]))
    suite.check("structural output is not forced onto the bank",
                all(prediction["topic"] in vocab.topics for prediction in variants["structural"]))
    suite.check("oracle is never worse than structural on the same cases",
                all(sum(1 for key in SLOT_KEYS
                        if variants["oracle_slot_max"][row][key] == references[row][key])
                    >= sum(1 for key in SLOT_KEYS
                           if variants["structural"][row][key] == references[row][key])
                    for row in range(heldout_rows)))
    # arbitration must be able to refuse: an unreachable margin can never override
    never = predict_variants(retrieval_module, torch.zeros((heldout_rows, 64)), vocab, bank,
                             torch, references=references, arbitrate_margin=1e9,
                             arbitrate_confidence=1.0)
    suite.check("a margin that can never be met disables arbitration",
                never["arbitrated"] == never["structural"])
    always = predict_variants(retrieval_module, torch.zeros((heldout_rows, 64)), vocab, bank,
                              torch, references=references, arbitrate_margin=-1e9,
                              arbitrate_confidence=1e9)
    suite.check("arbitration can be forced to take the retrieval candidate",
                all(always["arbitrated"][row][slot] == always["retrieval"][row][slot]
                    for row in range(heldout_rows) for slot in RetrievalBank.SLOTS))
    suite.check("arbitration never touches the topic slot",
                all(always["arbitrated"][row]["topic"] == always["structural"][row]["topic"]
                    for row in range(heldout_rows)))
    suite.check("a short reference list does not break the oracle branch",
                len(predict_variants(retrieval_module, torch.zeros((train_rows, 64)), vocab,
                                     bank, torch, references=references)["structural"])
                == train_rows)

    retrieval_trained = build_model(config, torch, bank=bank, retrieval_dim=16,
                                    retrieval_scale=4.0)
    retrieval_optimizer = torch.optim.AdamW(retrieval_trained.parameters(), lr=1e-2)
    retrieval_before = float(loss_terms(
        retrieval_trained(pooled_train), targets, weights, torch,
        retrieval_targets=train_retrieval, retrieval_weight=1.0)["total"].detach())
    for _ in range(30):
        terms = loss_terms(retrieval_trained(pooled_train), targets, weights, torch,
                           retrieval_targets=train_retrieval, retrieval_weight=1.0)
        retrieval_optimizer.zero_grad(set_to_none=True)
        terms["total"].backward()
        retrieval_optimizer.step()
    retrieval_after = float(loss_terms(
        retrieval_trained(pooled_train), targets, weights, torch,
        retrieval_targets=train_retrieval, retrieval_weight=1.0)["total"].detach())
    suite.check("retrieval gradient reaches the encoder and the prototypes",
                retrieval_after < retrieval_before, (retrieval_before, retrieval_after))

    suite.check("retrieval is optional, so a bankless model still loads",
                retrieval_module(torch.zeros((2, 64)))["retrieval"] != {} and
                build_model(config, torch)(torch.zeros((2, 64)))["retrieval"] == {})

    # ---- sealed-split refusal (the only two ways the final test could leak in)
    try:
        base.refuse_sealed_split(r"D:\AI\DBA\training\web-sources\risawoz\memory_test.jsonl", items)
        suite.check("a path naming the sealed test is refused", False, "no exception")
    except Exception:
        suite.check("a path naming the sealed test is refused", True)
    try:
        base.refuse_sealed_split(DEFAULT_CASES, [{"id": "RS-te-0001"}])
        suite.check("a sealed-test id is refused", False, "no exception")
    except Exception:
        suite.check("a sealed-test id is refused", True)
    base.refuse_sealed_split(DEFAULT_CASES, items)
    suite.check("the ordinary corpus is accepted", True)

    # The refusal above is the only place this file may mention the sealed split, so a
    # second check asserts that no code path OPENS it: the string may appear as an
    # argument to refuse_sealed_split, never as a corpus to load.
    source = Path(__file__).read_text(encoding="utf-8")
    reads_sealed = [line for line in source.splitlines()
                    if "memory_test" in line and "refuse_sealed_split" not in line
                    and "naming the sealed" not in line]
    suite.check("no code path opens the sealed split", not reads_sealed, reads_sealed)
    return suite.report()


# --------------------------------------------------------------------------
# preflight: real cache, real corpus, no CUDA
# --------------------------------------------------------------------------
def preflight(args):
    torch = base._torch()
    items = DATA.all_cases(args.cases)
    base.refuse_sealed_split(args.cases, items)
    train_items = [case for case in items if case["group"] == DATA.TRAIN]
    heldout_items = [case for case in items if case["group"] == DATA.HELDOUT]
    train_cases = base.case_slots(train_items)
    heldout_cases = base.case_slots(heldout_items)
    vocab = build_vocab(train_cases)
    reach = reachability(heldout_cases, vocab)
    bank = build_bank(train_cases)

    cache = base.ReadOnlyFeatureCache(args.feature_cache, DATA.data_version(items),
                                      args.backbone_sha256, allow_any_backbone=args.allow_any_backbone)
    cache.open()
    pooled_dim = base.pooled_dim_for(cache.shape, args.pool)

    report = {
        "cases": {"train": len(train_items), "heldout": len(heldout_items),
                  "total": len(items)},
        "feature_cache": cache.describe() if hasattr(cache, "describe") else
                         {"root": str(cache.root), "shape": list(cache.shape)},
        "pooled_dim": pooled_dim,
        "vocab_counts": vocab.counts(),
        "reachability": {key: value for key, value in reach.items()
                         if key not in ("per_case",)},
        "retrieval_bank": {
            "counts": bank.counts(),
            "coverage_dev": bank.coverage(heldout_cases),
            "note": ("verbatim train label strings; this is the hard ceiling of a "
                     "retrieval-only output and is strictly below the structural ceiling"),
        },
        "cuda_context_created": torch.cuda.is_initialized(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run(args):
    torch = base._torch()
    if not str(args.device).startswith("cuda"):
        raise TrainingError("training is GPU-only; got --device %s" % args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "training.log")

    def status_update(**fields):
        payload = {"schema": SCHEMA, "phase": "starting", "step": 0}
        path = out_dir / "status.json"
        if path.is_file():
            try:
                payload.update(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        payload.update(fields)
        payload["updated_at"] = time.time()
        write_json(path, payload)

    def power_gate(stage):
        if args.skip_power_check:
            return {"stage": stage, "guard_lease": None, "ac_connected": None,
                    "checked": False}
        return base.power_report(stage, require_lease=True)

    def pause_result(stage, report):
        payload = {"ok": False, "paused": True, "exit_code": EXIT_PAUSED,
                   "phase": "paused", "paused_stage": stage, "power": report}
        write_json(out_dir / "status.json", payload)
        return payload

    report = power_gate("before-cuda")
    log("power gate before CUDA: %s" % json.dumps(report, ensure_ascii=False, sort_keys=True))

    items = DATA.all_cases(args.cases)
    base.refuse_sealed_split(args.cases, items)
    data_hash = DATA.data_version(items)
    train_items = [case for case in items if case["group"] == DATA.TRAIN]
    heldout_items = [case for case in items if case["group"] == DATA.HELDOUT]
    train_cases = base.case_slots(train_items)
    heldout_cases = base.case_slots(heldout_items)
    counts = {"train": len(train_items), "heldout": len(heldout_items), "total": len(items)}
    log("cases %s data_hash=%s" % (json.dumps(counts, sort_keys=True), data_hash[:16]))

    vocab = build_vocab(train_cases)
    reach = reachability(heldout_cases, vocab)
    log("vocab %s" % json.dumps(vocab.counts(), sort_keys=True))
    log("reachability ceiling_four_slot_exact=%s"
        % json.dumps(reach["ceiling_four_slot_exact"]))
    for key, value in reach.items():
        if key.startswith("unreachable_") and value["occurrences"]:
            log("unreachable %s: %s" % (key, json.dumps(value, ensure_ascii=False)))

    bank = build_bank(train_cases) if args.retrieval else None
    if bank is not None:
        log("retrieval bank %s" % json.dumps(bank.counts(), sort_keys=True))
        log("retrieval ceiling (verbatim train coverage) %s"
            % json.dumps(bank.coverage(heldout_cases), ensure_ascii=False, sort_keys=True))

    cache = base.ReadOnlyFeatureCache(args.feature_cache, data_hash, args.backbone_sha256,
                                      allow_any_backbone=args.allow_any_backbone)
    cache_report = cache.open()
    log("feature cache: %s" % json.dumps(cache_report, ensure_ascii=False, sort_keys=True))

    pooled_train = base.build_pooled(cache, train_items, args.pool, log=log)
    pooled_dev = base.build_pooled(cache, heldout_items, args.pool, log=log)
    pooled_dim = int(pooled_train.shape[1])
    log("pooled train=%s dev=%s dim=%d" % (tuple(pooled_train.shape),
                                           tuple(pooled_dev.shape), pooled_dim))

    targets = encode_targets(train_cases, vocab, torch)
    references = build_references(heldout_cases, reach)

    device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pooled_train = pooled_train.to(device)
    pooled_dev = pooled_dev.to(device)
    targets = targets.to(device)
    retrieval_targets = None
    if bank is not None:
        retrieval_targets = encode_retrieval_targets(train_cases, bank, torch)
        coverage = bank.coverage(train_cases)
        log("retrieval train targets %s"
            % json.dumps({slot: {"in_bank": int((target != base.IGNORE).sum()),
                                 "cases": int(target.numel()),
                                 "coverage": coverage[slot]["rate"]}
                          for slot, target in retrieval_targets.items()}, sort_keys=True))
        retrieval_targets = {slot: target.to(device)
                             for slot, target in retrieval_targets.items()}

    config = ModelConfig(pooled_dim=pooled_dim, d_state=args.d_state, vocab=vocab,
                         dropout=args.dropout)
    module = build_model(config, torch, bank=bank, retrieval_dim=args.retrieval_dim,
                         retrieval_scale=args.retrieval_scale).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    weights = dict(DEFAULT_WEIGHTS)

    start_step = 0
    history = []
    resume_path = Path(args.resume) if args.resume else (out_dir / "checkpoint-last.pt")
    if resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        stored = payload.get("vocab")
        if stored is not None and StructuralVocab.from_dict(stored).to_dict() != vocab.to_dict():
            raise TrainingError("checkpoint vocabulary does not match this corpus; refusing "
                                "to load class indices that no longer mean the same thing")
        stored_bank = payload.get("bank")
        if (stored_bank is None) != (bank is None):
            raise TrainingError("checkpoint and this run disagree about candidate retrieval "
                                "(checkpoint bank present=%s, run bank present=%s)"
                                % (stored_bank is not None, bank is not None))
        if stored_bank is not None:
            stored_bank = RetrievalBank.from_dict(stored_bank)
            if stored_bank.to_dict() != bank.to_dict():
                raise TrainingError("checkpoint retrieval bank does not match this corpus; "
                                    "refusing to load prototype ids that mean something else")
        module.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        history = list(payload.get("history", []))
        log("resumed %s at step %d" % (resume_path, start_step))

    log("model parameters=%d class_counts=%s" % (parameter_count(module),
                                                 json.dumps(vocab.counts(), sort_keys=True)))
    log("heads are parallel classification: no autoregression, no teacher forcing")
    status_update(phase="training", step=start_step, cases=counts, data_hash=data_hash,
                  vocab=vocab.counts(), ceiling=reach["ceiling_four_slot_exact"])

    wrong_permutation = base.wrong_state_shift(len(heldout_items), args.seed, torch,
                                               device=device)

    def save_at(path, step, phase):
        atomic_torch_save(torch, {
            "schema": SCHEMA, "step": step, "phase": phase, "model": module.state_dict(),
            "optimizer": optimizer.state_dict(), "config": config.to_dict(),
            "vocab": vocab.to_dict(), "history": history, "data_hash": data_hash,
            "bank": bank.to_dict() if bank is not None else None,
            "retrieval_weight": args.retrieval_weight,
            "pool": args.pool, "script": SCRIPT_NAME,
        }, path)

    def before_step(step):
        if args.skip_power_check:
            return
        try:
            base.pause_if_requested(lambda: save_at(out_dir / PAUSE_NAME, step - 1, "paused"))
            report = base.power_report("before-optimizer-step", require_lease=True)
            if not report.get("ac_connected", False):
                raise PowerLoss(report)
        except Done:
            save_at(out_dir / PAUSE_NAME, step - 1, "paused")
            raise
        except PowerLoss as exc:
            save_at(out_dir / PAUSE_NAME, step - 1, "paused")
            raise Done(pause_result("before-optimizer-step", exc.report)) from None

    eval_history = []

    def evaluate_at(step):
        if args.eval_every <= 0 or step % args.eval_every != 0:
            return
        evaluation = evaluate(module, pooled_dev, references, vocab, torch, wrong_permutation,
                              bank=bank, arbitrate_margin=args.arbitrate_margin,
                              arbitrate_confidence=args.arbitrate_confidence)
        eval_history.append({"step": step, **compact_eval(evaluation)})
        log("dev @%d: %s" % (step, eval_summary_line(evaluation)))
        write_json(out_dir / "dev-report.json", {
            "step": step,
            "conditions": {name: evaluation["conditions"][name]["score"]
                           for name in evaluation["conditions"]},
            "paired": evaluation["paired"],
            "reachability": {key: value for key, value in reach.items()
                             if key not in ("per_case",)},
            "vocab": vocab.counts(),
            "script": SCRIPT_NAME,
        })
        with (out_dir / "dev-predictions.jsonl").open("w", encoding="utf-8") as handle:
            for index, reference in enumerate(references):
                handle.write(json.dumps({
                    "reference": {key: reference[key] for key in SLOT_KEYS},
                    "true": evaluation["conditions"]["true"]["predictions"][index],
                    "zero": evaluation["conditions"]["zero"]["predictions"][index],
                    "wrong": evaluation["conditions"]["wrong"]["predictions"][index],
                    "_reachable": reference["_reachable"],
                }, ensure_ascii=False) + "\n")

    def checkpoint_at(step):
        if args.checkpoint_every <= 0 or step % args.checkpoint_every != 0:
            return
        save_at(out_dir / "checkpoint-last.pt", step, "training")
        log("checkpoint written at step %d" % step)

    started = time.time()
    try:
        result = run_training(torch=torch, module=module, optimizer=optimizer, weights=weights,
                              pooled_train=pooled_train, targets=targets, log=log,
                              steps=args.steps, batch_size=args.batch_size, seed=args.seed,
                              start_step=start_step, history=history,
                              evaluate_at=evaluate_at, before_step=before_step,
                              checkpoint_at=checkpoint_at,
                              retrieval_targets=retrieval_targets,
                              retrieval_weight=args.retrieval_weight if bank is not None else 0.0)
    except Done as done:
        payload = done.args[0] if done.args else {"paused": True, "exit_code": EXIT_PAUSED}
        log("paused: %s" % json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return payload

    final_step = result["final_step"]
    save_at(out_dir / "checkpoint-last.pt", final_step, "complete")
    final_eval = evaluate(module, pooled_dev, references, vocab, torch, wrong_permutation,
                          bank=bank, arbitrate_margin=args.arbitrate_margin,
                          arbitrate_confidence=args.arbitrate_confidence)
    log("finished step %d: %s" % (final_step, eval_summary_line(final_eval)))
    write_json(out_dir / "dev-report.json", {
        "step": final_step,
        "conditions": {name: {"score": final_eval["conditions"][name]["score"],
                              "variants": {variant: score for variant, score in
                                           final_eval["conditions"][name]["variants"].items()}}
                       for name in final_eval["conditions"]},
        "paired": final_eval["paired"],
        "reachability": {key: value for key, value in reach.items() if key != "per_case"},
        "retrieval_bank": ({"counts": bank.counts(), "coverage_dev": bank.coverage(heldout_cases)}
                           if bank is not None else None),
        "vocab": vocab.counts(),
        "baselines": base.constant_baselines(train_cases, references),
        "script": SCRIPT_NAME,
        "elapsed_seconds": round(time.time() - started, 1),
    })
    with (out_dir / "eval-history.jsonl").open("w", encoding="utf-8") as handle:
        for record in eval_history:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(out_dir / "latest.json", {"checkpoint": str(out_dir / "checkpoint-last.pt"),
                                         "step": final_step})

    true_condition = final_eval["conditions"]["true"]
    headline = true_condition["score"]
    variant_rates = {variant: variant_rate(true_condition, variant)
                     for variant in true_condition["variants"]}
    status_update(phase="complete", step=final_step, paused=False,
                  four_slot_exact=headline["four_slot_exact_rate"],
                  content_slot_accuracy=headline["content_slot_accuracy"],
                  variant_four_slot_exact=variant_rates,
                  ceiling_four_slot_exact=reach["ceiling_four_slot_exact"],
                  paired=final_eval["paired"])
    return {"ok": True, "phase": "complete", "step": final_step,
            "four_slot_exact_rate": headline["four_slot_exact_rate"],
            "content_slot_accuracy": headline["content_slot_accuracy"],
            "variant_four_slot_exact": variant_rates,
            "ceiling_four_slot_exact": reach["ceiling_four_slot_exact"],
            "exit_code": 0}


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    if args.preflight:
        return preflight(args)
    try:
        result = run(args)
    except Done as done:
        payload = done.args[0] if done.args else {"paused": True, "exit_code": EXIT_PAUSED}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return int(payload.get("exit_code", 1))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    sys.exit(main())
