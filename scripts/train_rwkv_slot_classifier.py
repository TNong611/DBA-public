"""FROZEN RWKV STATE -> FOUR STRUCTURED SLOTS, BY CLASSIFICATION (no free-form generation).

WHY THIS FILE EXISTS
--------------------
Two runs already trained an autoregressive decoder to emit the whole four-line memory
text from the frozen state:

  v3  teacher-forced token loss on train 0.23, dev four-slot exact 0/1779
  v4  FiLM conditioning added; dev four-slot exact 0/1779, content-slot accuracy 3.3%

Both collapsed to the same failure: the decoder learned the label GRAMMAR (`主题：` /
`约束：` / `无`) and the high-frequency slot values, and never learned to read the
state. The floor that "always write 无" already earns is about 1.05/4 slots, and both
runs sat exactly there. A linear probe over the same cached features reaches 66.5%
topic accuracy on dev (chance 0.47%), so the state is demonstrably informative -- the
autoregressive objective, not the state, was the problem.

This script replaces generation with structured prediction. Format is not predicted at
all: it is fixed by construction. Four separate heads read one shared state encoding:

  topic       closed-vocabulary classification over the train split's own topic strings
  presence    per value slot: is the reference value the empty marker `无`?
  value       one shared character-level GRU decoder, told which slot it is writing,
              producing the constraint / pending / result strings

Nothing else is emitted, so the 1.05/4 "write the grammar" floor is unreachable: a
prediction only scores when the actual slot STRING is right.

VOCABULARY IS TRAIN-ONLY AND UNKNOWNS ARE COUNTED, NEVER MAPPED
---------------------------------------------------------------
The topic classes and the character alphabet come from TRAIN labels exclusively. A dev
topic or character that the train split never produced cannot be emitted, so it can
never be scored correct. Mapping it to some nearest class would inflate the score and
hide a distribution gap, so it is left wrong and *counted* instead:

  reachability.heldout_topics_absent_from_vocab
  reachability.heldout_slots_with_unknown_characters
  reachability.heldout_cases_fully_reachable        <-- the honest ceiling
  reachability.ceiling_all_four_exact               <-- max achievable headline rate

`ceiling_*` is the number the headline has to be read against. If the ceiling itself is
below the target, the target is unreachable by construction and the task definition,
not the training run, is what has to change.

METRICS THAT CANNOT BE EARNED TRIVIALLY
----------------------------------------
  four_slot_exact        topic AND all three value strings exactly right
  per_slot_exact         separately per slot, so one easy slot cannot carry the run
  content_slot_accuracy  restricted to slots whose reference is NOT `无`
                         (pending is `无` in 97.8% of train cases, so a pending head
                         that always says `无` scores 97.8% while knowing nothing;
                         this metric refuses to pay for that)
  baselines              most-common-topic + per-slot-majority, and all-`无`
  ablation_conditions    true vs zero vs wrong (cyclically shifted) state, evaluated
                         on the SAME cases so the comparison is paired, plus per-case
                         win/loss/tie counts -- a token-loss gap alone is not evidence

SAFETY
------
Same fail-closed contract as the trainer: direct AC check plus a live guard lease
before CUDA is touched and at every optimizer boundary, exit code 75 after an atomic
checkpoint when either is missing. `--skip-power-check` is accepted only together with
`--self-test`, because it is the one flag that can train on battery.

The sealed final split is unreachable from here by construction: a `--cases` path whose
name contains "test", or any case id in the `RS-te-` namespace, is refused outright.

FEATURE CACHE
-------------
Reads an existing run's cache strictly READ-ONLY and never re-extracts: the frozen 1.5B
backbone is not loaded at all. The reader validates the manifest against this run's own
data hash, the trainer's `FEATURE_VERSION` and `state_features` source fingerprint, the
expected backbone weight hash, and the required case ids; on any mismatch it refuses
instead of training on rows that mean something else. It never deletes or rewrites a
shard, so pointing it at the wrong directory cannot damage the trainer's cache.

Rows are pooled (mean, or mean+max) immediately on load. Per-element standardisation
commutes with mean pooling, so this is the same input as the trainer's, at ~100 MB
instead of ~6.9 GB resident.

Run:
  D:\\AI\\DBA\\.venv-rwkv\\Scripts\\python.exe train_rwkv_slot_classifier.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rwkv_slots_data as DATA

# Shared contracts live in the trainer so the cache format, the AC gate and the
# checkpoint format cannot drift between the two programs. Importing it defines
# classes only; it loads torch lazily and touches no device.
import train_rwkv_slots as TRAINER

LABEL = "structured slot classifier (frozen RWKV-7 recurrent state, no backbone training)"
BACKBONE_TRAINING = False
SCRIPT_VERSION = "train_rwkv_slot_classifier.py/1"
FORMAT = 1
EXIT_PAUSED = TRAINER.EXIT_PAUSED              # 75
STATE_MODES = ("true", "none", "shuffled")
POOL_MODES = ("mean", "meanmax")
CONDITIONS = ("true", "zero", "wrong")

DEFAULT_CASES = r"D:\AI\DBA\training\web-sources\risawoz\memory_cases.jsonl"
DEFAULT_OUT = Path(r"D:\AI\DBA\training\rwkv-slot-classifier")
RESUME_NAME = "checkpoint-last.pt"
PAUSE_NAME = "checkpoint-paused.pt"
STATUS_NAME = "status.json"
LOG_NAME = "train.log"
POINTER_NAME = "latest.json"
EVAL_HISTORY_NAME = "eval-history.jsonl"
PREDICTIONS_NAME = "dev-predictions.jsonl"
REPORT_NAME = "dev-report.json"

# The backbone the reused cache was extracted with. A cache built by another model
# would otherwise be accepted silently, which is exactly the identity confusion the
# cache key exists to prevent.
KNOWN_BACKBONE_SHA256 = "23e13ff62ac6f80b28910ffc230324939e80ddb00b98f892a5951c0da071e700"

VALUE_SLOTS = ("constraint", "pending", "result")
IGNORE = -100
PAD_ID = 0
START_ID = 1
MASK_ID = 2
CHAR_OFFSET = 3
EMPTY = DATA.EMPTY_SLOT                        # 无
FILM_GAIN_LIMIT = 2.0
FILM_BIAS_LIMIT = 4.0
HISTORY_LIMIT = TRAINER.HISTORY_LIMIT


def require(module, names):
    """Fail loudly if a shared contract was renamed, instead of at an odd moment."""
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise RuntimeError("train_rwkv_slots no longer exposes: " + ", ".join(missing))
    return [getattr(module, name) for name in names]


(EXIT_PAUSED, FEATURE_VERSION, FEATURE_CACHE_DIR, FEATURE_CACHE_MANIFEST,
 FEATURE_CACHE_FORMAT, feature_function_fingerprint, feature_cache_identity,
 feature_cache_key, write_json, atomic_torch_save, rng_state, restore_rng,
 cpu_optimizer_state, save_checkpoint, load_checkpoint, TrainingError, PowerLoss,
 Done, main_connected, power_report, pause_if_requested, _) = require(
    TRAINER,
    ["EXIT_PAUSED", "FEATURE_VERSION", "FEATURE_CACHE_DIR", "FEATURE_CACHE_MANIFEST",
     "FEATURE_CACHE_FORMAT", "feature_function_fingerprint", "feature_cache_identity",
     "feature_cache_key", "write_json", "atomic_torch_save", "rng_state", "restore_rng",
     "cpu_optimizer_state", "save_checkpoint", "load_checkpoint", "TrainingError",
     "PowerLoss", "Done", "main_connected", "power_report", "pause_if_requested",
     "training_allowed"])


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def parse_slots(summary):
    """{slot key -> value} for one memory text, strictly checked.

    `DATA.slot_values` is tolerant (it skips lines it does not recognise), which is
    right for scoring generated text and wrong for reading a label: a typo would
    silently drop a slot and the missing slot would be scored as a miss nobody
    notices. Here the four slots must be present, in order, and nonempty.
    """
    lines = [line.strip() for line in str(summary).split("\n") if line.strip()]
    if len(lines) != len(DATA.SLOT_KEYS):
        raise TrainingError("memory text must have exactly %d lines, got %d: %r"
                            % (len(DATA.SLOT_KEYS), len(lines), summary[:80]))
    values = DATA.slot_values(summary)
    out = {}
    for index, key in enumerate(DATA.SLOT_KEYS):
        label = DATA.SLOT_LABELS[key]
        if not lines[index].startswith(label + "："):
            raise TrainingError("slot %d is not %s: %r" % (index, label, lines[index][:40]))
        if key not in values:
            raise TrainingError("slot %s did not parse" % label)
        if not values[key].strip():
            raise TrainingError("slot %s is empty" % label)
        out[key] = values[key].strip()
    return out


def case_slots(items):
    """[(case, {slot -> value})] with the label parse done once per case."""
    out = []
    for case in items:
        out.append((case, parse_slots(case["summary"])))
    return out


# --------------------------------------------------------------------------
# vocabularies (train split only)
# --------------------------------------------------------------------------
class TopicVocab:
    """Closed set of topic strings seen in TRAIN labels."""

    def __init__(self, train_topics):
        self.itos = sorted(set(train_topics))
        if not self.itos:
            raise TrainingError("no train topics: cannot build a topic vocabulary")
        self.stoi = {name: index for index, name in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def index(self, topic):
        """Train-vocabulary index, or None when the topic was never seen in train."""
        return self.stoi.get(topic)

    def identity(self):
        return {"size": len(self.itos),
                "sha256": hashlib.sha256("\n".join(self.itos).encode("utf-8")).hexdigest()[:16]}


class CharVocab:
    """Character alphabet of TRAIN value labels, plus PAD/START/MASK inputs.

    No UNK output class on purpose. An UNK class would let the model emit a token
    that can never match any reference, and it would blur the difference between
    "predicted wrongly" and "could not be predicted at all". Characters outside
    this alphabet are simply not representable, and the run counts how many
    references that costs.
    """

    def __init__(self, train_values):
        self.itos = sorted(set("".join(train_values)))
        if not self.itos:
            raise TrainingError("no train characters: cannot build a character vocabulary")
        self.stoi = {ch: index for index, ch in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @property
    def n_chars(self):
        return len(self.itos)

    @property
    def embed_size(self):
        return len(self.itos) + CHAR_OFFSET

    @property
    def n_output(self):
        return len(self.itos) + 1

    @property
    def eos_class(self):
        return len(self.itos)

    def encode(self, text):
        """Character class ids. Raises on a character outside the alphabet."""
        out = []
        for ch in text:
            index = self.stoi.get(ch)
            if index is None:
                raise TrainingError("character %r is not in the train alphabet" % ch)
            out.append(index)
        return out

    def decode(self, classes):
        return "".join(self.itos[index] for index in classes)

    def covers(self, text):
        return all(ch in self.stoi for ch in text)

    def missing_chars(self, text):
        return sorted({ch for ch in text if ch not in self.stoi})

    def identity(self):
        return {"size": len(self.itos),
                "sha256": hashlib.sha256("".join(self.itos).encode("utf-8")).hexdigest()[:16]}


def build_vocabularies(train_cases):
    """Vocabularies come from TRAIN labels only; dev never contributes a class."""
    topic = TopicVocab([slots["topic"] for _, slots in train_cases])
    values = [slots[key] for _, slots in train_cases for key in VALUE_SLOTS]
    chars = CharVocab(values)
    return topic, chars


def max_value_length(train_cases):
    return max(len(slots[key]) for _, slots in train_cases for key in VALUE_SLOTS)


# --------------------------------------------------------------------------
# reachability: what the train-only vocabulary makes impossible, counted
# --------------------------------------------------------------------------
def reachability(cases, topic_vocab, char_vocab, max_chars):
    """Per-case reachability plus the resulting ceiling on the headline metric.

    A reference slot is reachable when every character it contains is in the train
    alphabet AND it is not longer than the decode budget. An unreachable reference
    can never be produced, so counting it as a plain miss would understate the
    model and counting it as a success would overstate it -- it is reported
    separately and excluded from the ceiling.
    """
    topic_absent, per_slot_unknown, per_slot_over_length = [], {k: 0 for k in VALUE_SLOTS}, {k: 0 for k in VALUE_SLOTS}
    unknown_chars = {}
    unknown_topic_names = {}
    fully_reachable = 0
    records = {}
    for case, slots in cases:
        topic_ok = slots["topic"] in topic_vocab.stoi
        if not topic_ok:
            topic_absent.append(case["id"])
            unknown_topic_names[slots["topic"]] = unknown_topic_names.get(slots["topic"], 0) + 1
        slots_ok = {}
        for key in VALUE_SLOTS:
            value = slots[key]
            over = len(value) > max_chars
            missing = char_vocab.missing_chars(value)
            if missing:
                per_slot_unknown[key] += 1
                for ch in missing:
                    unknown_chars[ch] = unknown_chars.get(ch, 0) + 1
            if over:
                per_slot_over_length[key] += 1
            slots_ok[key] = (not missing) and (not over)
        reachable = topic_ok and all(slots_ok.values())
        fully_reachable += 1 if reachable else 0
        records[case["id"]] = {"topic": topic_ok, "slots": slots_ok, "reachable": reachable}
    total = len(cases)
    return {
        "cases": total,
        "heldout_cases_fully_reachable": fully_reachable,
        "ceiling_all_four_exact": (fully_reachable / total) if total else None,
        "heldout_topics_absent_from_vocab": {"count": len(topic_absent),
                                             "ids": topic_absent[:20],
                                             "distinct_topics": sorted(unknown_topic_names)},
        "heldout_slots_with_unknown_characters": dict(per_slot_unknown),
        "heldout_slots_over_decode_budget": dict(per_slot_over_length),
        "unknown_characters_referenced": {"count": len(unknown_chars),
                                          "characters": sorted(unknown_chars)[:40]},
        "per_case": records,
        "note": ("train-only vocabulary; an unreachable reference is scored as a miss but "
                 "is excluded from ceiling_all_four_exact"),
    }


# --------------------------------------------------------------------------
# read-only feature cache reader
# --------------------------------------------------------------------------
class ReadOnlyFeatureCache:
    """Strictly read-only reader of another run's extracted feature rows.

    Deliberately NOT `train_rwkv_slots.FeatureCache`: that class DISCARDS (deletes)
    shards when the identity does not match, which is the right call for a cache the
    run itself owns and a destructive one for a cache belonging to a finished
    training run. Every failure path here raises and touches nothing.
    """

    def __init__(self, root, data_hash, expected_backbone_sha256, n_layer=None,
                 allow_any_backbone=False):
        self.root = Path(root)
        self.data_hash = data_hash
        self.expected_backbone_sha256 = expected_backbone_sha256
        self.expected_n_layer = n_layer
        self.allow_any_backbone = bool(allow_any_backbone)
        self.manifest = None
        self.shards = []
        self.index = {}          # case id -> (shard number, row)
        self.shape = None
        self._open_name = None
        self._open_tensor = None

    def open(self):
        """Validate the manifest and every shard index. Returns a report dict."""
        manifest_path = self.root / FEATURE_CACHE_MANIFEST
        if not manifest_path.is_file():
            raise TrainingError("feature cache manifest not found: %s" % manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("format") != FEATURE_CACHE_FORMAT:
            raise TrainingError("feature cache manifest format mismatch: %s" % manifest_path)
        identity = manifest.get("identity")
        if not isinstance(identity, dict):
            raise TrainingError("feature cache manifest carries no identity block")
        key = manifest.get("cache_key")
        if key != feature_cache_key(identity):
            raise TrainingError("feature cache key does not match its own identity block "
                                "(half-written or edited manifest): %s" % manifest_path)
        if identity.get("data_hash") != self.data_hash:
            raise TrainingError(
                "feature cache was extracted for a different dataset (cache data_hash %s, "
                "this run %s); refusing instead of training on rows that mean something else"
                % (str(identity.get("data_hash"))[:16], str(self.data_hash)[:16]))
        if identity.get("feature_version") != FEATURE_VERSION:
            raise TrainingError("feature cache version %s != %s"
                                % (identity.get("feature_version"), FEATURE_VERSION))
        fingerprint = feature_function_fingerprint()
        if identity.get("feature_function") != fingerprint:
            raise TrainingError(
                "state_features source changed since the cache was written (%s vs %s); the "
                "cached rows no longer mean what this code assumes"
                % (identity.get("feature_function"), fingerprint))
        if identity.get("backbone_mode") != "frozen-state":
            raise TrainingError("feature cache was not built from the frozen backbone: %s"
                                % identity.get("backbone_mode"))
        if identity.get("dtype") != "float32":
            raise TrainingError("feature cache dtype %s is not float32" % identity.get("dtype"))
        backbone = identity.get("backbone") or {}
        stored_sha = backbone.get("sha256")
        if not self.allow_any_backbone and stored_sha != self.expected_backbone_sha256:
            raise TrainingError(
                "feature cache was extracted with backbone sha256 %s, expected %s; pass "
                "--allow-any-backbone only if that is deliberate"
                % (stored_sha, self.expected_backbone_sha256))
        shape = manifest.get("feature_shape")
        if not shape or len(shape) != 2:
            raise TrainingError("feature cache manifest has no usable feature_shape")
        self.shape = (int(shape[0]), int(shape[1]))
        n_layer = identity.get("n_layer")
        if not isinstance(n_layer, int) or self.shape[0] != n_layer * 3:
            raise TrainingError("feature cache shape %s is inconsistent with n_layer %s"
                                % (self.shape, n_layer))
        if self.expected_n_layer is not None and int(n_layer) != int(self.expected_n_layer):
            raise TrainingError("feature cache n_layer %s, expected %s"
                                % (n_layer, self.expected_n_layer))
        self._read_shard_indexes(manifest)
        self.manifest = manifest
        return {"root": str(self.root), "cache_key": str(key)[:16], "cases_on_disk": len(self.index),
                "shards": len(self.shards), "closed": bool(manifest.get("closed")),
                "feature_shape": list(self.shape), "n_layer": int(n_layer),
                "backbone_sha256": stored_sha,
                "identity_verified": True, "read_only": True}

    def _read_shard_indexes(self, manifest):
        shards, seen = [], set()
        for name in manifest.get("shards", []):
            if not isinstance(name, str) or Path(name).name != name:
                raise TrainingError("shard name is not a plain file name: %r" % (name,))
            if not (self.root / name).is_file():
                raise TrainingError("feature cache shard is missing: %s" % name)
            index_path = self.root / (name + ".ids.json")
            if not index_path.is_file():
                raise TrainingError("feature cache shard index is missing: %s" % index_path.name)
            index = json.loads(index_path.read_text(encoding="utf-8"))
            ids = index.get("ids")
            if not isinstance(ids, list):
                raise TrainingError("feature cache shard index is malformed: %s" % index_path.name)
            for case_id in ids:
                if not isinstance(case_id, str) or case_id in seen:
                    raise TrainingError("case %r appears twice in the feature cache" % (case_id,))
                seen.add(case_id)
            shards.append({"name": name, "ids": ids})
        for number, shard in enumerate(shards):
            for row, case_id in enumerate(shard["ids"]):
                self.index[case_id] = (number, row)
        self.shards = shards

    def _open_shard(self, name):
        if self._open_name == name:
            return self._open_tensor
        payload = _torch().load(self.root / name, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != FEATURE_CACHE_FORMAT:
            raise TrainingError("feature cache shard is not loadable: " + name)
        if payload.get("key") != self.manifest.get("cache_key"):
            raise TrainingError("feature cache shard belongs to another cache key: " + name)
        features, ids = payload.get("features"), payload.get("ids")
        if not hasattr(features, "dim") or features.dim() != 3:
            raise TrainingError("shard holds no [cases, rows, width] matrix: " + name)
        if not isinstance(ids, list) or int(features.shape[0]) != len(ids):
            raise TrainingError("shard rows do not match its id list: " + name)
        if (int(features.shape[1]), int(features.shape[2])) != self.shape:
            raise TrainingError("shard feature shape does not match the manifest: " + name)
        self._open_name = name
        self._open_tensor = features.to(_torch().float32)
        return self._open_tensor

    def row(self, case_id):
        entry = self.index.get(case_id)
        if entry is None:
            return None
        number, row = entry
        return self._open_shard(self.shards[number]["name"])[row]

    def require_all(self, case_ids):
        missing = [case_id for case_id in case_ids if case_id not in self.index]
        if missing:
            raise TrainingError(
                "feature cache is missing %d of %d required cases (e.g. %s). This script "
                "never extracts features, so an incomplete cache cannot be used; point "
                "--feature-cache at the run whose extraction actually finished."
                % (len(missing), len(case_ids), ", ".join(missing[:5])))


def _torch():
    """Import torch on demand so --help and argument errors never load it."""
    import torch
    return torch


def pool_rows(row, mode):
    """[rows, width] -> [pooled_dim]. Mean commutes with per-element standardisation."""
    if mode == "mean":
        return row.mean(dim=0)
    if mode == "meanmax":
        return _torch().cat([row.mean(dim=0), row.max(dim=0).values])
    raise TrainingError("unknown pool mode: %r" % (mode,))


def pooled_dim_for(feature_shape, mode):
    width = int(feature_shape[1])
    return width * (2 if mode == "meanmax" else 1)


def build_pooled(cache, cases, mode, log=None):
    """Pool every case's rows on the fly; never hold the full [N, rows, width] matrix."""
    torch = _torch()
    width = pooled_dim_for(cache.shape, mode)
    matrix = torch.empty((len(cases), width), dtype=torch.float32)
    for index, case in enumerate(cases):
        row = cache.row(case["id"])
        if row is None:
            raise TrainingError("case %s is not in the feature cache" % case["id"])
        matrix[index] = pool_rows(row, mode)
        if log is not None and (index + 1) % 2000 == 0:
            log("pooled %d/%d cases" % (index + 1, len(cases)))
    return matrix


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class ClassifierConfig:
    FIELDS = ("pooled_dim", "d_state", "n_topics", "n_chars", "d_char", "d_rnn",
              "dropout")

    def __init__(self, pooled_dim, d_state, n_topics, n_chars, d_char=192,
                 d_rnn=256, dropout=0.1):
        self.pooled_dim = int(pooled_dim)
        self.d_state = int(d_state)
        self.n_topics = int(n_topics)
        self.n_chars = int(n_chars)
        self.d_char = int(d_char)
        self.d_rnn = int(d_rnn)
        self.dropout = float(dropout)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.FIELDS}


def build_classifier(config, torch):
    """State encoder + topic head + presence heads + one shared character decoder."""
    nn = torch.nn
    functional = torch.nn.functional
    value_slots = len(VALUE_SLOTS)

    class ValueDecoder(nn.Module):
        """Character-level GRU producing one slot value, told which slot it is writing.

        One shared decoder plus a learned slot embedding, rather than three separate
        decoders: with ~10k training cases a shared trunk sees three times the
        supervision and cannot overfit one slot independently.

        The state enters twice: as the GRU's initial hidden state (so the whole value
        is conditioned) and through zero-initialised FiLM on every input embedding
        (so the state stays on the gradient path at every step). Zero initialisation
        means the modulation is exactly the identity at step 0 -- the v4 lesson was
        that a weak, optional conditioning path is simply never learned into use.
        """

        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.n_chars + CHAR_OFFSET, cfg.d_char, padding_idx=PAD_ID)
            self.slot = nn.Embedding(value_slots, cfg.d_state)
            self.hidden = nn.Linear(cfg.d_state, cfg.d_rnn)
            self.film = nn.Linear(cfg.d_state, 2 * cfg.d_char)
            self.gru = nn.GRU(cfg.d_char, cfg.d_rnn, batch_first=True)
            self.out = nn.Linear(cfg.d_rnn, cfg.n_chars + 1)
            self.drop = nn.Dropout(cfg.dropout)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            nn.init.normal_(self.slot.weight, std=0.02)

        def slot_state(self, state, slot_index):
            """[B, d_state] -> [B, d_state] for ONE slot. Used by the decoder loop."""
            index = torch.full((state.shape[0],), int(slot_index), dtype=torch.long,
                               device=state.device)
            return state + self.slot(index)

        def all_slot_states(self, state):
            """[B, d_state] -> [B, slots, d_state]: every value slot at once.

            Training scores all three slots in a single GRU call from this, which is
            why the slot is a row of the batch rather than a separate network.
            """
            return state.unsqueeze(1) + self.slot.weight.unsqueeze(0)

        def initial_hidden(self, slot_state):
            return torch.tanh(self.hidden(slot_state)).unsqueeze(0).contiguous()

        def modulate(self, embeddings, slot_state):
            """Zero-initialised FiLM, applied on the last two dimensions.

            `unsqueeze(-2)` makes one implementation serve both the [B, T, d_char]
            single-slot step and the [B, slots, T, d_char] training block, so the two
            cannot drift apart into different conditioning.
            """
            gain, bias = self.film(slot_state).chunk(2, dim=-1)
            gain = FILM_GAIN_LIMIT * torch.tanh(gain).unsqueeze(-2)
            bias = FILM_BIAS_LIMIT * torch.tanh(bias).unsqueeze(-2)
            return embeddings * (1.0 + gain) + bias

        def forward(self, state, inputs):
            """state [B, d_state], inputs [B, slots, T] -> logits [B, slots, T, classes].

            The slot axis is folded into the batch for the recurrent pass and restored
            afterwards, so a slot's value is decoded without the other slots' prefixes
            in its context -- the cross-slot coupling that made whole-text generation
            fail is not present here.
            """
            count, slots, width = inputs.shape
            slot_state = self.all_slot_states(state)
            embeddings = self.modulate(self.embed(inputs), slot_state)
            flat = embeddings.reshape(count * slots, width, self.cfg.d_char)
            hidden = torch.tanh(self.hidden(slot_state.reshape(count * slots, -1)))
            output, _ = self.gru(flat, hidden.unsqueeze(0).contiguous())
            logits = self.out(self.drop(output))
            return logits.reshape(count, slots, width, -1)

        def step(self, slot_state, current, hidden):
            """One character for one slot: slot_state [B, d_state], current [B]."""
            embeddings = self.modulate(self.embed(current).unsqueeze(1), slot_state)
            output, hidden = self.gru(embeddings, hidden)
            return self.out(output[:, 0, :]), hidden

    class SlotClassifier(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.norm = nn.LayerNorm(cfg.pooled_dim)
            self.enc1 = nn.Linear(cfg.pooled_dim, cfg.d_state)
            self.enc2 = nn.Linear(cfg.d_state, cfg.d_state)
            self.drop = nn.Dropout(cfg.dropout)
            self.topic = nn.Linear(cfg.d_state, max(2, cfg.n_topics))
            self.presence = nn.Linear(cfg.d_state, 2 * value_slots)
            self.decoder = ValueDecoder(cfg)

        def encode(self, pooled):
            hidden = self.drop(functional.gelu(self.enc1(self.norm(pooled))))
            return self.drop(functional.gelu(self.enc2(hidden)))

        def forward(self, pooled, inputs):
            state = self.encode(pooled)
            topic_logits = self.topic(state)
            presence_logits = self.presence(state).view(-1, value_slots, 2)
            value_logits = self.decoder(state, inputs)
            return {"state": state, "topic": topic_logits, "presence": presence_logits,
                    "value": value_logits}

    return SlotClassifier(config)


def parameter_count(module):
    return sum(int(p.numel()) for p in module.parameters())


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------
class Targets:
    """Everything the loss and the metrics need, as tensors on one device.

    Targets are stored per slot as a rectangular [N, slots, width] int64 block with
    IGNORE padding, so one GRU call scores all three slots at once. The input block
    is derived from the target block by a right shift, which makes it impossible for
    the two to disagree about what was teacher-forced.
    """

    def __init__(self, topic_ids, presence, values, length):
        self.topic_ids = topic_ids
        self.presence = presence
        self.values = values
        self.length = int(length)

    def to(self, device):
        # `_n_chars` must travel with the tensors: a Target block that lost it would
        # fail deep inside cross_entropy instead of here.
        return Targets(self.topic_ids.to(device), self.presence.to(device),
                       self.values.to(device), self.length).bind(self.n_chars)

    def inputs(self, torch, rate=0.0, seed=0, step=0):
        """Right-shifted input ids, optionally corrupted at `rate`.

        Corruption replaces gold prefix characters with MASK. Only the INPUTS change:
        the targets stay ground truth, so the model is still asked for the true next
        character while being unable to read it off the prefix. Without this the
        character head can score well by copying, which is how the v3/v4 decoders
        reached a low train loss while reading nothing from the state.
        """
        count = self.values.shape[0]
        width = self.values.shape[2]
        inputs = torch.full_like(self.values, PAD_ID)
        inputs[:, :, 0] = START_ID
        if width > 1:
            shifted = self.values[:, :, :-1]
            is_char = (shifted >= 0) & (shifted < self.n_chars)
            inputs[:, :, 1:] = torch.where(is_char, shifted + CHAR_OFFSET,
                                           torch.full_like(shifted, PAD_ID))
        if rate > 0:
            generator = torch.Generator(device="cpu").manual_seed(
                (int(seed) * 1000003 + int(step)) % (2 ** 31))
            draw = torch.rand(inputs.shape, generator=generator).to(inputs.device)
            eligible = torch.zeros_like(inputs, dtype=torch.bool)
            eligible[:, :, 1:] = True
            eligible &= inputs != PAD_ID
            inputs = torch.where(eligible & (draw < float(rate)),
                                 torch.full_like(inputs, MASK_ID), inputs)
        return inputs

    @property
    def n_chars(self):
        return self._n_chars

    def bind(self, n_chars):
        self._n_chars = int(n_chars)
        return self

    @property
    def width(self):
        return int(self.values.shape[2])


def encode_targets(cases, topic_vocab, char_vocab, max_chars):
    """Build the target block. Raises if a TRAIN label is outside the vocabulary.

    The train split defines the vocabulary, so an out-of-alphabet train label would
    mean the vocabularies were built from a different case list -- caught here rather
    than by a silently wrong loss.
    """
    torch = _torch()
    count = len(cases)
    topic_ids = torch.full((count,), -1, dtype=torch.long)
    presence = torch.zeros((count, len(VALUE_SLOTS)), dtype=torch.long)
    values = torch.full((count, len(VALUE_SLOTS), max_chars + 1), IGNORE, dtype=torch.long)
    for row, (case, slots) in enumerate(cases):
        index = topic_vocab.index(slots["topic"])
        if index is None:
            raise TrainingError("case %s has a topic outside the train vocabulary" % case["id"])
        topic_ids[row] = index
        for column, key in enumerate(VALUE_SLOTS):
            value = slots[key]
            presence[row, column] = 1 if value == EMPTY else 0
            if len(value) > max_chars:
                raise TrainingError("case %s slot %s is longer than the decode budget"
                                    % (case["id"], key))
            classes = char_vocab.encode(value)
            values[row, column, :len(classes)] = torch.tensor(classes, dtype=torch.long)
            values[row, column, len(classes)] = char_vocab.eos_class
    return Targets(topic_ids, presence, values, max_chars + 1).bind(len(char_vocab))


def make_batches(count, batch_size, seed, epoch=0):
    torch = _torch()
    generator = torch.Generator().manual_seed(int(seed) + int(epoch) * 7919)
    order = torch.randperm(count, generator=generator)
    return [order[start:start + batch_size] for start in range(0, count, batch_size)]


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------
def loss_terms(outputs, targets, weights, torch):
    """Weighted sum of the three head losses, reported separately."""
    functional = torch.nn.functional
    topic = functional.cross_entropy(outputs["topic"], targets.topic_ids)
    presence = functional.cross_entropy(outputs["presence"].reshape(-1, 2),
                                        targets.presence.reshape(-1))
    value = functional.cross_entropy(outputs["value"].reshape(-1, targets.n_chars + 1),
                                     targets.values.reshape(-1), ignore_index=IGNORE)
    total = (weights["topic"] * topic + weights["presence"] * presence
             + weights["value"] * value)
    return {"total": total, "topic": topic, "presence": presence, "value": value}


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------
def greedy_decode(module, pooled, slot_index, char_vocab, max_chars, torch,
                  chunk=256):
    """Raw greedy character decode for one slot, batched over cases.

    Character-at-a-time, but each slot is decoded on its own: there is no grammar to
    fail at, and a wrong character in one slot cannot derail another.
    """
    functional = torch.nn.functional
    results = []
    module.eval()
    with torch.inference_mode():
        for start in range(0, pooled.shape[0], chunk):
            block = pooled[start:start + chunk]
            slot_state = module.decoder.slot_state(module.encode(block), slot_index)
            hidden = module.decoder.initial_hidden(slot_state)
            current = torch.full((block.shape[0],), START_ID, dtype=torch.long,
                                 device=block.device)
            done = torch.zeros(block.shape[0], dtype=torch.bool, device=block.device)
            collected = [[] for _ in range(block.shape[0])]
            # Exactly `max_chars` emission steps, never one more. The budget is the
            # longest TRAIN value, so every reachable reference fits; running an extra
            # step would let an undeveloped model emit an over-budget string that no
            # reference could ever match, silently turning a format failure into a
            # "prediction".
            for _step in range(max_chars):
                logits, hidden = module.decoder.step(slot_state, current, hidden)
                predicted = logits.argmax(dim=-1)
                for row in range(block.shape[0]):
                    if done[row]:
                        continue
                    value = int(predicted[row])
                    if value == char_vocab.eos_class:
                        done[row] = True
                    else:
                        collected[row].append(value)
                current = torch.where(predicted < char_vocab.n_chars,
                                      predicted + CHAR_OFFSET,
                                      torch.full_like(predicted, PAD_ID))
                hidden = hidden * (~done).view(1, -1, 1).to(hidden.dtype)
                if bool(done.all()):
                    break
            results.extend(char_vocab.decode(row) for row in collected)
    module.train()
    return results


def predict_slots(module, pooled, char_vocab, max_chars, torch):
    """{slot -> (raw decoded values, presence-gated values)} for every value slot."""
    module.eval()
    with torch.inference_mode():
        state = module.encode(pooled)
        presence = module.presence(state).view(-1, len(VALUE_SLOTS), 2).argmax(dim=-1)
    raw, gated = {}, {}
    for column, key in enumerate(VALUE_SLOTS):
        decoded = greedy_decode(module, pooled, column, char_vocab, max_chars, torch)
        raw[key] = decoded
        empty = [bool(presence[row, column] == 1) for row in range(presence.shape[0])]
        gated[key] = [EMPTY if empty[row] else decoded[row] for row in range(len(decoded))]
    return raw, gated


def predict_topics(module, pooled, topic_vocab, torch):
    module.eval()
    with torch.inference_mode():
        logits = module.topic(module.encode(pooled))
        indices = logits.argmax(dim=-1).tolist()
    return [topic_vocab.itos[index] for index in indices]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(references, predictions):
    """Per-slot exactness, content-slot accuracy and the headline four-slot rate.

    Two numbers are deliberately NOT collapsed into one. `content_slot_accuracy`
    counts only slots whose reference is not `无`, because `pending` is `无` in 97.8%
    of cases and a head that always answers `无` would otherwise collect most of the
    score for free. And `_reachable` (set by the caller from the vocabulary report)
    is carried on each reference so an unreachable case can be excluded from the
    reported ceiling without being dropped from the plain rate.
    """
    count = len(references)
    per_slot = {key: 0 for key in DATA.SLOT_KEYS}
    content_hits = content_scored = 0
    content_per_slot = {key: [0, 0] for key in VALUE_SLOTS}
    exact = 0
    exact_reachable = 0
    for index, reference in enumerate(references):
        prediction = predictions[index]
        hit_all = True
        for key in DATA.SLOT_KEYS:
            same = prediction[key] == reference[key]
            per_slot[key] += 1 if same else 0
            hit_all = hit_all and same
        exact += 1 if hit_all else 0
        if reference.get("_reachable"):
            exact_reachable += 1 if hit_all else 0
        for key in VALUE_SLOTS:
            if reference[key] != EMPTY:
                content_scored += 1
                content_per_slot[key][1] += 1
                if prediction[key] == reference[key]:
                    content_hits += 1
                    content_per_slot[key][0] += 1
    reachable_count = sum(1 for reference in references if reference.get("_reachable"))
    return {
        "cases": count,
        "four_slot_exact": exact,
        "four_slot_exact_rate": (exact / count) if count else None,
        "four_slot_exact_rate_on_reachable": (exact_reachable / reachable_count)
                                             if reachable_count else None,
        "per_slot_exact": {key: per_slot[key] for key in DATA.SLOT_KEYS},
        "per_slot_exact_rate": {key: (per_slot[key] / count) if count else None
                                for key in DATA.SLOT_KEYS},
        "content_slot_hits": content_hits,
        "content_slot_scored": content_scored,
        "content_slot_accuracy": (content_hits / content_scored) if content_scored else None,
        "content_per_slot": {key: {"hits": value[0], "scored": value[1],
                                   "accuracy": (value[0] / value[1]) if value[1] else None}
                             for key, value in content_per_slot.items()},
    }


def paired(before, after):
    """Per-case win/loss/tie of `after` over `before` on the four-slot boolean.

    The ablation claim is "the real state beats zero/shuffled". A lower aggregate
    loss is not that claim: it can come from a handful of easy cases. This counts
    per-case outcomes instead, so the claim has to hold case by case.
    """
    win = loss = tie = 0
    for left, right in zip(before, after):
        if right > left:
            win += 1
        elif right < left:
            loss += 1
        else:
            tie += 1
    return {"wins": win, "losses": loss, "ties": tie,
            "win_rate": (win / (win + loss)) if (win + loss) else None}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(module, pooled, references, topic_vocab, char_vocab, max_chars, torch,
             wrong_permutation):
    """Score the same cases under true / zero / wrong conditioning, paired.

    Three conditioning blocks, one model, one set of cases:
      true   the case's own state
      zero   the zero vector -- quality here can only come from label priors
      wrong  another case's state, cyclically shifted -- conditioning is present but
             means nothing, which is what separates "reads the state" from "uses
             some vector as a bias"
    Two decoding policies are reported for each: `decoded` (greedy characters only)
    and `presence_gated` (the presence head may force `无`). Neither is privileged in
    the report; the run's headline uses the gated one and both are always present so
    the choice can be re-judged.
    """
    zero = torch.zeros_like(pooled)
    wrong = pooled[wrong_permutation] if wrong_permutation is not None else zero
    per_case_hits = {}
    conditions = {}
    for name, block in (("true", pooled), ("zero", zero), ("wrong", wrong)):
        topics = predict_topics(module, block, topic_vocab, torch)
        raw, gated = predict_slots(module, block, char_vocab, max_chars, torch)
        variants = {}
        for variant, values in (("decoded", raw), ("presence_gated", gated)):
            predictions = []
            for row in range(len(references)):
                prediction = {"topic": topics[row]}
                for key in VALUE_SLOTS:
                    prediction[key] = values[key][row]
                predictions.append(prediction)
            variants[variant] = {"predictions": predictions,
                                 "score": score(references, predictions)}
        chosen = variants["presence_gated"]
        hits = [1 if all(chosen["predictions"][row][key] == references[row][key]
                         for key in DATA.SLOT_KEYS) else 0
                for row in range(len(references))]
        per_case_hits[name] = hits
        conditions[name] = {"decoded": variants["decoded"]["score"],
                            "presence_gated": chosen["score"],
                            "four_slot_hits": hits,
                            "predictions": chosen["predictions"]}
    comparisons = {}
    for name in ("zero", "wrong"):
        comparisons["true_vs_" + name] = paired(per_case_hits[name], per_case_hits["true"])
    return {"conditions": conditions, "paired": comparisons,
            "headline_condition": "true", "headline_policy": "presence_gated"}


def baseline_report(references, baselines):
    """What a state-free predictor scores on the very same cases.

    Without this, `four_slot_exact_rate` has no scale. `always_empty` is the one
    that matters most: writing `无` in all three value slots plus the most common
    topic is a predictor that reads nothing at all, and it is the thing a real
    result has to beat by a wide margin.
    """
    per_slot_predictions = [dict(baselines["per_slot"]) for _ in references]
    always_empty = [dict(baselines["four_slot"]) for _ in references]
    return {"majority_per_slot": score(references, per_slot_predictions),
            "always_empty_plus_common_topic": score(references, always_empty),
            "per_slot_constant": baselines["per_slot"],
            "per_slot_constant_train_share": baselines["per_slot_share"]}


def constant_baselines(train_cases, references):
    """What "ignore the state, predict the marginal" scores on the same cases."""
    topic_counts, slot_counts = {}, {key: {} for key in DATA.SLOT_KEYS}
    for _, slots in train_cases:
        topic_counts[slots["topic"]] = topic_counts.get(slots["topic"], 0) + 1
        for key in DATA.SLOT_KEYS:
            slot_counts[key][slots[key]] = slot_counts[key].get(slots[key], 0) + 1
    per_slot = {"topic": max(topic_counts, key=topic_counts.get)}
    for key in VALUE_SLOTS:
        per_slot[key] = max(slot_counts[key], key=slot_counts[key].get)
    all_none = {"topic": per_slot["topic"]}
    for key in VALUE_SLOTS:
        all_none[key] = EMPTY
    return {"per_slot": per_slot, "four_slot": all_none,
            "per_slot_share": {key: slot_counts[key][per_slot[key]] / max(1, sum(slot_counts[key].values()))
                               for key in DATA.SLOT_KEYS}}


def wrong_state_shift(count, seed, torch, device=None):
    """The index mapping for the `wrong` ablation: another case's state, always.

    A plain roll by one, deliberately the same definition the trainer uses
    (`heldout_features.roll(1, dims=0)`), so an ablation number here means the same
    thing as an ablation number there. A `randperm` would leave cases that happen to
    receive their OWN state, quietly weakening the ablation for exactly those cases
    and giving a fixed point a free pass; a roll has none by construction.

    The index tensor is built on the CPU generator (reproducible) and then moved to
    the indexed tensor's device, which is what advanced indexing requires.
    """
    if count < 2:
        return None
    order = torch.arange(count, dtype=torch.long).roll(1)
    return order if device is None else order.to(device)


def build_references(cases, reach):
    """Reference slots for the dev split, each carrying its own reachability flag."""
    out = []
    for case, slots in cases:
        record = reach["per_case"][case["id"]]
        reference = {key: slots[key] for key in DATA.SLOT_KEYS}
        reference["_id"] = case["id"]
        reference["_reachable"] = bool(record["reachable"])
        out.append(reference)
    return out


(
    teacher_forcing_dropout_rate,
) = require(TRAINER, ["teacher_forcing_dropout_rate"])


# --------------------------------------------------------------------------
# run plumbing
# --------------------------------------------------------------------------
def refuse_sealed_split(cases_path, items):
    """Keep the sealed final split sealed.

    dev is the tuning split and is allowed. The final split is scored exactly once,
    by the independent evaluator, after dev passes. Letting this script read it
    would silently turn the acceptance number into another tuning number.
    """
    name = Path(cases_path).name.lower()
    if "test" in name:
        raise TrainingError(
            "refusing %s: a path whose name contains 'test' is the sealed final split, "
            "which this script never reads" % Path(cases_path).name)
    leaked = [case["id"] for case in items if str(case["id"]).startswith("RS-te-")]
    if leaked:
        raise TrainingError("refusing %d cases from the sealed test namespace (e.g. %s)"
                            % (len(leaked), leaked[0]))


class RunLog:
    """Append-only log that is also echoed, so a detached run stays observable."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = open(self.path, "a", encoding="utf-8")

    def __call__(self, message):
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message)
        self.handle.write(line + "\n")
        self.handle.flush()
        print(line, flush=True)

    def close(self):
        try:
            self.handle.close()
        except Exception:
            pass


def compact_eval(evaluation):
    """Evaluation without the per-case lists, for the status/history files."""
    return {
        "headline_condition": evaluation["headline_condition"],
        "headline_policy": evaluation["headline_policy"],
        "paired": evaluation["paired"],
        "conditions": {name: {"decoded": entry["decoded"],
                              "presence_gated": entry["presence_gated"]}
                       for name, entry in evaluation["conditions"].items()},
    }


def eval_summary_line(evaluation):
    parts = []
    for name in CONDITIONS:
        entry = evaluation["conditions"][name]["presence_gated"]
        parts.append("%s=%d/%d(%.4f)" % (name, entry["four_slot_exact"], entry["cases"],
                                         entry["four_slot_exact_rate"] or 0.0))
    content = evaluation["conditions"]["true"]["presence_gated"]["content_slot_accuracy"]
    paired_zero = evaluation["paired"]["true_vs_zero"]
    return ("dev four_slot_exact %s content_slot_accuracy(true)=%s true_vs_zero w/l/t=%d/%d/%d"
            % (" ".join(parts), "None" if content is None else "%.4f" % content,
               paired_zero["wins"], paired_zero["losses"], paired_zero["ties"]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Frozen-RWKV-state -> four structured slots by classification.")
    parser.add_argument("--cases", default=DEFAULT_CASES,
                        help="JSONL of labelled cases (train + dev). The sealed test split "
                             "is refused by name and by case id.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-cache", required=False,
                        help="directory holding the extracted feature cache, e.g. "
                             "D:\\AI\\DBA\\training\\risawoz-run-v3\\features\\<key>. "
                             "Read-only: nothing here is ever written or deleted.")
    parser.add_argument("--expect-backbone-sha256", default=KNOWN_BACKBONE_SHA256)
    parser.add_argument("--allow-any-backbone", action="store_true",
                        help="skip the backbone hash check (fixtures and plumbing only)")
    parser.add_argument("--pool", choices=POOL_MODES, default="meanmax")
    parser.add_argument("--state-mode", choices=STATE_MODES, default="true",
                        help="conditioning used for TRAINING (evaluation always reports "
                             "true/zero/wrong on the same cases)")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-state", type=int, default=384)
    parser.add_argument("--d-char", type=int, default=192)
    parser.add_argument("--d-rnn", type=int, default=256)
    parser.add_argument("--value-max-chars", type=int, default=0,
                        help="decode budget; 0 = derive from the longest TRAIN value "
                             "(never truncate a label)")
    parser.add_argument("--tf-dropout-start", type=float, default=0.1)
    parser.add_argument("--tf-dropout-end", type=float, default=0.6)
    parser.add_argument("--tf-dropout-ramp", type=int, default=400)
    parser.add_argument("--weight-topic", type=float, default=1.0)
    parser.add_argument("--weight-presence", type=float, default=0.5)
    parser.add_argument("--weight-value", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--decode-chunk", type=int, default=256)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-heldout", type=int, default=0)
    parser.add_argument("--resume", default=None,
                        help="checkpoint to continue from (default: <out-dir>/%s)" % RESUME_NAME)
    parser.add_argument("--skip-power-check", action="store_true",
                        help="ONLY valid with --self-test; disables the AC/guard gate")
    parser.add_argument("--self-test", action="store_true",
                        help="run the CPU self-tests and exit without touching CUDA")
    parser.add_argument("--preflight", action="store_true",
                        help="check cases, vocabularies and the feature cache, print a "
                             "report and exit; no CUDA, no guard lease, nothing trained")
    args = parser.parse_args(argv)
    if args.skip_power_check and not args.self_test:
        parser.error("--skip-power-check is only accepted together with --self-test")
    if args.steps < 0:
        parser.error("--steps must not be negative")
    return args


def run_training_steps(*, args, torch, module, optimizer, weights, train_block,
                       train_targets, train_count, start_step, history, current,
                       checkpoint_at, evaluate_at, before_step, log, status_update,
                       peak_memory, device):
    """The stepping loop, kept separate so the CPU self-test can drive it.

    Everything device- or power-specific arrives as a callback, so this function
    contains only the parts that are otherwise discovered slowly on a GPU: how the
    teacher-forcing rate is scheduled, which rows are conditioned together, where
    checkpoints and evaluations land, and what a pause does to the step counter.
    `--steps` is deliberately NOT part of the checkpoint identity, so a paused run
    can be resumed with a larger budget without invalidating its weights.
    """
    step = int(start_step)
    epoch = 0
    started = time.perf_counter()
    while step < args.steps:
        for order in make_batches(train_count, args.batch_size, args.seed, epoch):
            before_step(step)
            order = order.to(device)
            batch_block = train_block[order]
            batch_targets = slice_targets(train_targets, order, torch)
            rate = teacher_forcing_dropout_rate(step, args.tf_dropout_start,
                                                args.tf_dropout_end, args.tf_dropout_ramp)
            inputs = batch_targets.inputs(torch, rate=rate, seed=args.seed, step=step)
            outputs = module(batch_block, inputs)
            terms = loss_terms(outputs, batch_targets, weights, torch)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0).item())
            optimizer.step()
            step += 1
            current.update({"step": step, "loss": float(terms["total"].detach().item()),
                            "tf_rate": rate})
            record = {"step": step, "loss": round(current["loss"], 6),
                      "topic": round(float(terms["topic"].detach().item()), 6),
                      "presence": round(float(terms["presence"].detach().item()), 6),
                      "value": round(float(terms["value"].detach().item()), 6),
                      "grad_norm": round(grad_norm, 4)}
            history.append(dict(record))
            if args.log_every and (step % args.log_every == 0 or step == args.steps):
                elapsed = time.perf_counter() - started
                per_step = elapsed / max(1, step - start_step)
                peak = peak_memory()
                record.update({"seconds": round(elapsed, 3),
                               "seconds_per_step": round(per_step, 4), "vram_mib": peak})
                history[-1] = dict(record)
                log("step %d/%d total=%.4f topic=%.4f presence=%.4f value=%.4f grad=%.3f "
                    "%.3fs/step peak=%.1fMiB tf=%.3f"
                    % (step, args.steps, current["loss"], record["topic"], record["presence"],
                       record["value"], grad_norm, per_step, peak, rate))
                status_update(step=step, phase="training", last=record,
                              history=history[-HISTORY_LIMIT:],
                              step_seconds=round(per_step, 4),
                              eta_seconds=round(per_step * (args.steps - step), 1),
                              peak_vram_mib=peak)
            if args.eval_every and step % args.eval_every == 0 and step != args.steps:
                evaluate_at(step, "periodic")
            if step % args.checkpoint_every == 0 or step == args.steps:
                checkpoint_at(step, "training")
            if step >= args.steps:
                break
        epoch += 1
    return step


def _run_inner(args, hooks=None):
    hooks = hooks or {}
    torch = hooks.get("torch") or _torch()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / RESUME_NAME
    log = hooks.get("log") or RunLog(out_dir / LOG_NAME)
    status_path = out_dir / STATUS_NAME
    eval_history_path = out_dir / EVAL_HISTORY_NAME
    status = {}
    result = {"ok": False, "label": LABEL, "backbone_training": BACKBONE_TRAINING,
              "script": SCRIPT_VERSION, "state_mode": args.state_mode,
              "steps_requested": args.steps, "self_test": False}
    force_lease = bool(args.skip_power_check)

    def status_update(**changes):
        status.update(changes)
        status.update({"label": LABEL, "backbone_training": BACKBONE_TRAINING,
                       "script": SCRIPT_VERSION, "state_mode": args.state_mode,
                       "steps_requested": args.steps,
                       "artifact_paths": {"status": str(status_path), "log": str(log.path),
                                          "checkpoint": str(resume_path),
                                          "eval_history": str(eval_history_path)},
                       "updated_at": time.time()})
        write_json(status_path, status)

    def gate(stage):
        if force_lease:
            return {"stage": stage, "guard_lease": None, "ac_connected": None,
                    "power_gate": "disabled-by---skip-power-check"}
        try:
            return power_report(stage, require_lease=True)
        except PowerLoss as exc:
            raise Done(pause_result(stage, exc.report)) from None

    def pause_result(stage, report):
        payload = {"ok": False, "paused": True, "exit_code": EXIT_PAUSED,
                   "phase": "paused", "paused_stage": stage, "power": report,
                   "label": LABEL, "backbone_training": BACKBONE_TRAINING,
                   "state_mode": args.state_mode, "steps_requested": args.steps,
                   "artifacts": {"status": str(status_path), "log": str(log.path)}}
        status_update(phase="paused", paused=True, paused_before=stage, power=report)
        log("power gate stopped the run at %s: %s"
            % (stage, json.dumps(report, ensure_ascii=False)))
        return payload

    # ---- gate 1: before anything touches CUDA -----------------------------
    result["power_before_cuda"] = gate("before-cuda")
    log("power gate before CUDA: " + json.dumps(result["power_before_cuda"], ensure_ascii=False))
    if not torch.cuda.is_available():
        raise TrainingError("CUDA is required; this classifier refuses CPU training")
    if not torch.cuda.is_bf16_supported():
        raise TrainingError("CUDA BF16 is required")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # ---- cases ------------------------------------------------------------
    items = DATA.all_cases(args.cases)
    refuse_sealed_split(args.cases, items)
    if args.limit_train:
        train_cases = [c for c in items if c["group"] == DATA.TRAIN][:args.limit_train]
    else:
        train_cases = [c for c in items if c["group"] == DATA.TRAIN]
    if args.limit_heldout:
        heldout_cases = [c for c in items if c["group"] == DATA.HELDOUT][:args.limit_heldout]
    else:
        heldout_cases = [c for c in items if c["group"] == DATA.HELDOUT]
    if not train_cases or not heldout_cases:
        raise TrainingError("this classifier needs both train and dev cases")
    train_labelled = case_slots(train_cases)
    heldout_labelled = case_slots(heldout_cases)
    data_hash = DATA.data_version(items)
    log("cases %s data_hash=%s" % (json.dumps(DATA.split_counts(items)), data_hash[:16]))

    topic_vocab, char_vocab = build_vocabularies(train_labelled)
    train_max = max_value_length(train_labelled)
    max_chars = int(args.value_max_chars) if args.value_max_chars else train_max
    if max_chars < train_max:
        raise TrainingError("--value-max-chars %d is below the longest TRAIN value (%d); "
                            "truncating a label would silently corrupt the loss"
                            % (max_chars, train_max))
    log("vocabulary: %d topics, %d characters, decode budget %d chars"
        % (len(topic_vocab), len(char_vocab), max_chars))

    reach = reachability(heldout_labelled, topic_vocab, char_vocab, max_chars)
    log("dev reachability: %d/%d cases fully representable (ceiling %.4f); topics absent %d; "
        "slots with unknown characters %s"
        % (reach["heldout_cases_fully_reachable"], reach["cases"],
           reach["ceiling_all_four_exact"], reach["heldout_topics_absent_from_vocab"]["count"],
           json.dumps(reach["heldout_slots_with_unknown_characters"])))
    result["reachability"] = {key: value for key, value in reach.items() if key != "per_case"}

    # ---- features (read-only reuse; the 1.5B backbone is never loaded) -----
    if not args.feature_cache:
        raise TrainingError("--feature-cache is required: this script never extracts features, "
                            "it consumes one that an earlier run already wrote")
    cache = ReadOnlyFeatureCache(args.feature_cache, data_hash,
                                 args.expect_backbone_sha256,
                                 allow_any_backbone=args.allow_any_backbone)
    cache_report = cache.open()
    log("feature cache: %s" % json.dumps(cache_report, ensure_ascii=False))
    all_cases_in_order = train_cases + heldout_cases
    cache.require_all([case["id"] for case in all_cases_in_order])
    pooled = build_pooled(cache, all_cases_in_order, args.pool, log=log)
    train_count = len(train_cases)
    mean = pooled[:train_count].mean(dim=0, keepdim=True)
    std = pooled[:train_count].std(dim=0, keepdim=True).clamp_min(1e-5)
    pooled = (pooled - mean) / std
    pooled_dim = int(pooled.shape[1])
    result["features"] = dict(cache_report, pooled_dim=pooled_dim, pool=args.pool,
                              standardised_with="train-split statistics only",
                              train_cases=train_count, dev_cases=len(heldout_cases),
                              full_matrix_bytes_saved=int(
                                  len(all_cases_in_order) * cache.shape[0] * cache.shape[1] * 4
                                  - pooled.numel() * 4))
    log("pooled %d cases to width %d (%s) using train-only statistics"
        % (pooled.shape[0], pooled_dim, args.pool))

    train_block = pooled[:train_count].to(device)
    heldout_block = pooled[train_count:].to(device)
    if args.state_mode == "none":
        # Zero conditioning for TRAINING only. Evaluation still measures the trained
        # model under true / zero / wrong on the same cases, so this ablation is
        # reported on the same footing as a normal run instead of being a separate
        # experiment with its own numbers.
        train_block = torch.zeros_like(train_block)
    elif args.state_mode == "shuffled":
        order = torch.randperm(train_count, generator=torch.Generator().manual_seed(
            args.seed + 11)).to(device)
        train_block = train_block[order]
    del pooled

    train_targets = encode_targets(train_labelled, topic_vocab, char_vocab,
                                   max_chars).to(device)
    references = build_references(heldout_labelled, reach)
    baselines = constant_baselines(train_labelled, references)
    wrong_order = wrong_state_shift(len(heldout_cases), args.seed, torch,
                                          device=device)

    # ---- model ------------------------------------------------------------
    config = ClassifierConfig(pooled_dim, args.d_state, len(topic_vocab), len(char_vocab),
                              d_char=args.d_char, d_rnn=args.d_rnn, dropout=args.dropout)
    module = build_classifier(config, torch).to(device)
    module.train()
    parameters = parameter_count(module)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    weights = {"topic": args.weight_topic, "presence": args.weight_presence,
               "value": args.weight_value}

    run_identity = {
        "script": SCRIPT_VERSION, "format": FORMAT, "label": LABEL,
        "backbone_training": BACKBONE_TRAINING, "state_mode": args.state_mode,
        "pool": args.pool, "pooled_dim": pooled_dim, "feature_shape": list(cache.shape),
        "cache_key": cache_report["cache_key"], "cache_root": str(cache.root),
        "backbone_sha256": cache_report["backbone_sha256"],
        "data_hash": data_hash, "data_counts": DATA.split_counts(items),
        "classifier": config.to_dict(), "max_value_chars": max_chars,
        "topic_vocab": topic_vocab.identity(), "char_vocab": char_vocab.identity(),
        "loss_weights": weights, "seed": args.seed, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "tf_dropout": [args.tf_dropout_start, args.tf_dropout_end, args.tf_dropout_ramp],
        "case_ids": [case["id"] for case in all_cases_in_order],
        "unknown_policy": "train-only vocabulary; unreachable references counted, never mapped",
    }
    identity_hash = hashlib.sha256(
        json.dumps(run_identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:32]
    log("parameters=%d identity=%s" % (parameters, identity_hash))

    history = []
    current = {"step": 0, "loss": None, "tf_rate": 0.0}
    resumed_from = None
    start_step = 0
    resume_candidate = Path(args.resume) if args.resume else resume_path
    if resume_candidate.is_file():
        payload = accept_checkpoint(load_checkpoint(resume_candidate, torch), identity_hash,
                                    resume_candidate)
        module.load_state_dict(payload["model_state"], strict=True)
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        restore_rng(payload.get("rng"), torch)
        start_step = int(payload.get("step") or 0)
        history = list(payload.get("history") or [])
        resumed_from = str(resume_candidate)
        log("resumed from %s at step %d" % (resume_candidate, start_step))

    def checkpoint_payload(step, phase, evaluation=None):
        return {"format": FORMAT, "script": SCRIPT_VERSION, "identity_hash": identity_hash,
                "run_identity": run_identity, "step": int(step), "phase": phase,
                "model_state": module.state_dict(),
                "optimizer_state": cpu_optimizer_state(optimizer),
                "history": history[-HISTORY_LIMIT:], "loss": current["loss"],
                "dev": compact_eval(evaluation) if evaluation else None,
                "rng": rng_state(torch), "saved_at": time.time(), "device": str(device),
                "parameters": parameters, "state_mode": args.state_mode,
                "config": config.to_dict(), "max_value_chars": max_chars,
                "topics": topic_vocab.itos, "characters": char_vocab.itos,
                "resumed_from": resumed_from, "label": LABEL,
                "backbone_training": BACKBONE_TRAINING,
                "scope": "dev split only; the sealed final split has never been read here"}

    def save_at(path, step, phase, evaluation=None):
        save_checkpoint(path, checkpoint_payload(step, phase, evaluation), torch,
                        pointer=out_dir / POINTER_NAME)
        return str(path)

    def run_eval(step, phase="periodic"):
        evaluation = evaluate(module, heldout_block, references, topic_vocab, char_vocab,
                              max_chars, torch, wrong_order)
        record = {"step": int(step), "phase": phase,
                  "four_slot_exact_rate": evaluation["conditions"]["true"]["presence_gated"]["four_slot_exact_rate"],
                  "paired": evaluation["paired"]}
        with open(eval_history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": int(step), "phase": phase,
                                     "evaluation": compact_eval(evaluation)},
                                    ensure_ascii=False) + "\n")
        return evaluation, record

    # ---- warm-up forward: fail before spending steps ----------------------
    headline_index = torch.arange(1, device=device)
    warm_targets = slice_targets(train_targets, headline_index, torch)
    warm_outputs = module(train_block[:1], warm_targets.inputs(torch))
    initial_loss = float(loss_terms(warm_outputs, warm_targets, weights, torch)["total"].item())
    log("warm-up forward ok: first-case total loss %.4f (uniform topic ln(%d)=%.4f)"
        % (initial_loss, len(topic_vocab), math.log(max(2, len(topic_vocab)))))
    del warm_outputs, warm_targets, headline_index
    torch.cuda.empty_cache()

    # ---- training ---------------------------------------------------------
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    step = int(start_step)
    if step >= args.steps:
        log("nothing to train: checkpoint is already at step %d (requested %d)"
            % (step, args.steps))

    def before_step(current_step):
        """Guard at every optimizer boundary.

        Two independent checks on purpose: the guard's lease is a file another process
        keeps alive, so it can outlive the event it was issued for, and the direct AC
        read is what actually decides.
        """
        if force_lease:
            return
        try:
            pause_if_requested(lambda: save_at(out_dir / PAUSE_NAME, current["step"], "paused"))
        except SystemExit as exc:
            if exc.code == EXIT_PAUSED:
                raise Done(pause_result("before-optimizer-step",
                                        {"guard_lease": False,
                                         "ac_connected": main_connected()})) from None
            raise
        try:
            power_report("before-optimizer-step", require_lease=True)
        except PowerLoss as exc:
            status["power"] = exc.report
            raise Done(pause_result("before-optimizer-step", exc.report)) from None

    def checkpoint_at(step, phase):
        save_at(resume_path, step, phase)
        status_update(step=step, phase="training", checkpoint=str(resume_path),
                      history=history[-HISTORY_LIMIT:])
        log("checkpoint written at step %d" % step)

    def evaluate_at(step, phase):
        evaluation, _ = run_eval(step, phase)
        log("dev @%d: %s" % (step, eval_summary_line(evaluation)))
        status_update(step=step, phase="training", dev=compact_eval(evaluation),
                      history=history[-HISTORY_LIMIT:])
        return evaluation

    step = run_training_steps(
        args=args, torch=torch, module=module, optimizer=optimizer, weights=weights,
        train_block=train_block, train_targets=train_targets, train_count=train_count,
        start_step=step, history=history, current=current, checkpoint_at=checkpoint_at,
        evaluate_at=evaluate_at, before_step=before_step, log=log,
        status_update=status_update, device=device,
        peak_memory=lambda: round(torch.cuda.max_memory_allocated(0) / 1024 ** 2, 1))

    # ---- final dev evaluation --------------------------------------------
    final_eval, _ = run_eval(step, "final")
    save_at(resume_path, step, "complete", final_eval)
    peak = round(torch.cuda.max_memory_allocated(0) / 1024 ** 2, 1)
    headline = final_eval["conditions"]["true"]["presence_gated"]
    qualified, reasons = judge_qualified(headline, final_eval, reach)
    baselines_scored = baseline_report(references, baselines)

    predictions_path = out_dir / PREDICTIONS_NAME
    predictions = final_eval["conditions"]["true"]["predictions"]
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for index, reference in enumerate(references):
            handle.write(json.dumps(
                {"id": reference["_id"], "reachable": reference["_reachable"],
                 "reference": {key: reference[key] for key in DATA.SLOT_KEYS},
                 "predicted": predictions[index],
                 "exact": all(predictions[index][key] == reference[key]
                              for key in DATA.SLOT_KEYS)},
                ensure_ascii=False) + "\n")

    report = {
        "script": SCRIPT_VERSION, "label": LABEL,
        "backbone_training": BACKBONE_TRAINING, "self_test": False,
        "scope": ("dev split only. The sealed final split has never been read by this "
                  "script; qualification here is a dev result, not an acceptance."),
        "identity_hash": identity_hash, "run_identity": run_identity,
        "step": step, "parameters": parameters, "state_mode": args.state_mode,
        "classifier": config.to_dict(), "loss_weights": weights,
        "features": result["features"], "reachability": result["reachability"],
        "baselines": baselines_scored,
        "dev": compact_eval(final_eval),
        "dev_headline": {
            "criterion": "four_slot_exact_rate >= 0.80, dev, true conditioning",
            "four_slot_exact": headline["four_slot_exact"],
            "four_slot_exact_rate": headline["four_slot_exact_rate"],
            "four_slot_exact_rate_on_reachable": headline["four_slot_exact_rate_on_reachable"],
            "per_slot_exact_rate": headline["per_slot_exact_rate"],
            "content_slot_accuracy": headline["content_slot_accuracy"],
            "content_per_slot": headline["content_per_slot"],
            "vocabulary_ceiling": reach["ceiling_all_four_exact"],
        },
        "qualified": bool(qualified), "not_qualified_reasons": reasons,
        "ablations": {name: final_eval["conditions"][name]["presence_gated"]
                      for name in CONDITIONS},
        "paired": final_eval["paired"],
        "history": history[-HISTORY_LIMIT:],
        "peak_vram_mib": peak,
        "artifacts": {"status": str(status_path), "log": str(log.path),
                      "checkpoint": str(resume_path), "report": str(out_dir / REPORT_NAME),
                      "predictions": str(predictions_path),
                      "eval_history": str(eval_history_path)},
    }
    write_json(out_dir / REPORT_NAME, report)
    status_update(step=step, phase="complete", paused=False, qualified=bool(qualified),
                  not_qualified_reasons=reasons,
                  dev_headline=report["dev_headline"], dev=compact_eval(final_eval),
                  duration_seconds=round(time.perf_counter() - started, 3),
                  peak_vram_mib=peak, checkpoint=str(resume_path),
                  history=history[-HISTORY_LIMIT:])
    log("finished step %d: %s" % (step, eval_summary_line(final_eval)))
    log("ceiling %.4f | qualified=%s%s"
        % (reach["ceiling_all_four_exact"], qualified,
           "" if qualified else " | " + "; ".join(reasons)))
    log("dev per-slot exact rate: " + json.dumps(report["dev_headline"]["per_slot_exact_rate"]))
    log("dev baselines: " + json.dumps(
        {"majority_per_slot_all_four": baselines_scored["majority_per_slot"]["four_slot_exact_rate"],
         "always_empty_plus_common_topic": baselines_scored[
             "always_empty_plus_common_topic"]["four_slot_exact_rate"]}, ensure_ascii=False))
    log("report: %s | predictions: %s" % (out_dir / REPORT_NAME, predictions_path))

    result.update({"ok": True, "phase": "complete", "step": step, "qualified": bool(qualified),
                   "not_qualified_reasons": reasons, "identity_hash": identity_hash,
                   "initial_loss": initial_loss, "final_loss": current["loss"],
                   "parameters": parameters, "peak_vram_mib": peak,
                   "checkpoint": str(resume_path),
                   "dev_headline": report["dev_headline"],
                   "baselines": {"majority_per_slot_all_four_rate":
                                 baselines_scored["majority_per_slot"]["four_slot_exact_rate"],
                                 "always_empty_plus_common_topic_rate":
                                 baselines_scored["always_empty_plus_common_topic"]["four_slot_exact_rate"]},
                   "paired": final_eval["paired"],
                   "artifacts": report["artifacts"]})
    return result


# --------------------------------------------------------------------------
# CPU self-test
# --------------------------------------------------------------------------
class SelfTest:
    """Named assertions; every failure is listed, never swallowed."""

    def __init__(self):
        self.passed = 0
        self.failures = []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            return True
        self.failures.append("%s%s" % (name, (": " + str(detail)) if detail else ""))
        return False

    def raises(self, name, call):
        try:
            call()
        except TrainingError:
            self.passed += 1
            return True
        except Exception as exc:  # a non-training exception is a different bug
            self.failures.append("%s: raised %s instead of TrainingError"
                                 % (name, type(exc).__name__))
            return False
        self.failures.append("%s: nothing was raised" % name)
        return False


def _fixture_cases():
    """A tiny split containing a deliberate vocabulary gap on the dev side.

    DV-01 uses a topic and characters the train split never produced, which is the
    case the reachability accounting exists for: it must be counted, not mapped.
    """
    train = []
    for index in range(8):
        train.append(({"id": "TR-%02d" % index, "group": DATA.TRAIN}, {
            "topic": "电脑价格区间",
            "constraint": "本轮条件：价格区间=%d" % index,
            "pending": EMPTY if index % 2 else "核对报价",
            "result": "已告知价格%d" % index,
        }))
    train.append(({"id": "TR-08", "group": DATA.TRAIN},
                  {"topic": "航班改签", "constraint": EMPTY, "pending": EMPTY,
                   "result": "已确认改签"}))
    dev = [
        # every character here is in the train alphabet, so DV-00 is reachable
        ({"id": "DV-00", "group": DATA.HELDOUT},
         {"topic": "电脑价格区间", "constraint": "本轮条件：价格区间=7", "pending": EMPTY,
          "result": "已告知价格7"}),
        # a topic and characters the train split never produced
        ({"id": "DV-01", "group": DATA.HELDOUT},
         {"topic": "酒店房型", "constraint": "本轮条件：房型=套房", "pending": EMPTY,
          "result": "已告知房型"}),
    ]
    return train, dev


def _no_raise(call):
    try:
        call()
        return True
    except Exception:
        return False


def self_test(verbose=False):
    """Exercise every non-GPU code path. Never opens a CUDA context."""
    import shutil
    import tempfile

    torch = _torch()
    suite = SelfTest()
    train, dev = _fixture_cases()

    # ---- label parsing ----------------------------------------------------
    good = "主题：电机调试\n约束：只允许GPU\n未完成：跑低速测试\n结果：尚未确认"
    suite.check("parse_slots reads all four slots",
                parse_slots(good) == {"topic": "电机调试", "constraint": "只允许GPU",
                                      "pending": "跑低速测试", "result": "尚未确认"})
    suite.raises("parse_slots rejects three lines",
                 lambda: parse_slots("主题：a\n约束：b\n结果：c"))
    suite.raises("parse_slots rejects a mislabelled slot",
                 lambda: parse_slots("主题：a\n条件：b\n未完成：c\n结果：d"))
    suite.raises("parse_slots rejects an empty value",
                 lambda: parse_slots("主题：a\n约束：\n未完成：c\n结果：d"))
    suite.raises("parse_slots rejects an extra line",
                 lambda: parse_slots("主题：a\n约束：b\n未完成：c\n结果：d\n备注：e"))
    suite.check("case_slots reads the summary of a case",
                case_slots([{"id": "X", "summary": good}])[0][1]["result"] == "尚未确认")
    suite.raises("case_slots refuses a malformed summary",
                 lambda: case_slots([{"id": "X", "summary": "主题：a"}]))

    # ---- vocabularies are train-only --------------------------------------
    topic_vocab, char_vocab = build_vocabularies(train)
    suite.check("topic vocabulary holds exactly the train topics",
                topic_vocab.itos == ["电脑价格区间", "航班改签"], topic_vocab.itos)
    suite.check("a dev-only topic has no index", topic_vocab.index("酒店房型") is None)
    suite.check("a dev-only character is absent from the alphabet", "套" not in char_vocab.stoi)
    suite.check("the alphabet covers every train value",
                all(char_vocab.covers(slots[key]) for _, slots in train for key in VALUE_SLOTS))
    suite.check("the alphabet is the sorted unique train character set",
                char_vocab.itos == sorted(set(char_vocab.itos)))
    suite.check("unknown characters are reported, not mapped",
                set(char_vocab.missing_chars("已告知房型")) == {"房", "型"},
                char_vocab.missing_chars("已告知房型"))
    suite.raises("encode raises on an out-of-alphabet character",
                 lambda: char_vocab.encode("已告知房型"))
    suite.check("vocabulary identities are stable across two builds",
                build_vocabularies(train)[1].identity() == char_vocab.identity())

    max_chars = max_value_length(train)
    suite.check("decode budget comes from the longest train value", max_chars == 11, max_chars)

    # ---- reachability accounting -----------------------------------------
    reach = reachability(dev, topic_vocab, char_vocab, max_chars)
    suite.check("reachability counts the absent topic",
                reach["heldout_topics_absent_from_vocab"]["count"] == 1, reach)
    suite.check("reachability counts unknown characters per slot",
                reach["heldout_slots_with_unknown_characters"] == {"constraint": 1, "pending": 0,
                                                                   "result": 1},
                reach["heldout_slots_with_unknown_characters"])
    suite.check("reachability ceiling is the reachable share",
                abs(reach["ceiling_all_four_exact"] - 0.5) < 1e-12,
                reach["ceiling_all_four_exact"])
    suite.check("reachability flags cases individually",
                reach["per_case"]["DV-00"]["reachable"]
                and not reach["per_case"]["DV-01"]["reachable"])
    suite.check("an over-budget value is counted as over budget",
                reachability(dev, topic_vocab, char_vocab, 3)["heldout_slots_over_decode_budget"]
                == {"constraint": 2, "pending": 0, "result": 2},
                reachability(dev, topic_vocab, char_vocab, 3)["heldout_slots_over_decode_budget"])

    # ---- target encoding --------------------------------------------------
    targets = encode_targets(train, topic_vocab, char_vocab, max_chars)
    suite.check("topic targets use train indices",
                targets.topic_ids.tolist() == [0] * 8 + [1],
                targets.topic_ids.tolist())
    suite.check("target width is budget + 1 (room for EOS)",
                int(targets.values.shape[2]) == max_chars + 1)
    suite.check("target block width matches its own report",
                targets.width == max_chars + 1)
    first = char_vocab.encode(train[0][1]["constraint"])
    expected = first + [char_vocab.eos_class] + [IGNORE] * (max_chars - len(first))
    suite.check("EOS follows the value and IGNORE fills the rest",
                targets.values[0, 0].tolist() == expected,
                targets.values[0, 0].tolist())
    suite.check("presence marks the empty marker",
                targets.presence[1, 1].item() == 1 and targets.presence[1, 0].item() == 0)
    suite.check("presence is zero for a content value",
                targets.presence[0, 1].item() == 0)
    suite.raises("a topic outside the train vocabulary is refused",
                 lambda: encode_targets([({"id": "X"}, {"topic": "酒店房型", "constraint": EMPTY,
                                                        "pending": EMPTY, "result": EMPTY})],
                                        topic_vocab, char_vocab, max_chars))
    suite.raises("a value longer than the budget is refused, never truncated",
                 lambda: encode_targets(train, topic_vocab, char_vocab, 3))
    suite.check("a sliced block keeps its own class count",
                slice_targets(targets, torch.arange(2), torch).n_chars == len(char_vocab))

    # ---- teacher-forcing inputs ------------------------------------------
    inputs = targets.inputs(torch)
    suite.check("inputs start with START at position 0",
                bool((inputs[:, :, 0] == START_ID).all()))
    suite.check("inputs are the target block shifted right",
                int(inputs[0, 0, 1]) == CHAR_OFFSET + first[0])
    suite.check("a short value is followed by PAD inputs, not stray ids",
                set(inputs[8, 0, 2:].tolist()) <= {PAD_ID},
                sorted(set(inputs[8, 0, 2:].tolist())))
    clean = targets.inputs(torch, rate=0.0, seed=1, step=1)
    suite.check("rate 0 corrupts nothing", bool((clean != MASK_ID).all()))
    eligible = (inputs != PAD_ID).clone()
    eligible[:, :, 0] = False
    full = targets.inputs(torch, rate=1.0, seed=1, step=1)
    suite.check("rate 1 masks every eligible input and never the start",
                bool((full[eligible] == MASK_ID).all())
                and bool((full[:, :, 0] == START_ID).all()))
    suite.check("corruption is deterministic for one step",
                bool((targets.inputs(torch, rate=0.5, seed=1, step=7)
                      == targets.inputs(torch, rate=0.5, seed=1, step=7)).all()))
    suite.check("corruption changes between steps",
                not bool((targets.inputs(torch, rate=0.5, seed=1, step=7)
                          == targets.inputs(torch, rate=0.5, seed=1, step=8)).all()))
    half = targets.inputs(torch, rate=0.5, seed=3, step=5)
    observed = float((half[eligible] == MASK_ID).float().mean())
    suite.check("corruption rate is close to the schedule", abs(observed - 0.5) < 0.15, observed)

    # ---- model, loss, gradients ------------------------------------------
    config = ClassifierConfig(pooled_dim=12, d_state=16, n_topics=len(topic_vocab),
                              n_chars=len(char_vocab), d_char=8, d_rnn=8, dropout=0.0)
    module = build_classifier(config, torch)
    module.train()
    pooled = torch.randn(len(train), 12)
    batch = slice_targets(targets, torch.arange(4), torch)
    outputs = module(pooled[:4], batch.inputs(torch))
    suite.check("topic head shape", tuple(outputs["topic"].shape) == (4, len(topic_vocab)))
    suite.check("presence head shape",
                tuple(outputs["presence"].shape) == (4, len(VALUE_SLOTS), 2))
    suite.check("value head shape (batch, slot, character, class)",
                tuple(outputs["value"].shape)
                == (4, len(VALUE_SLOTS), max_chars + 1, len(char_vocab) + 1),
                tuple(outputs["value"].shape))
    weights = {"topic": 1.0, "presence": 0.5, "value": 1.0}
    terms = loss_terms(outputs, batch, weights, torch)
    suite.check("all loss components are finite",
                all(math.isfinite(float(value.detach())) for value in terms.values()),
                {key: float(value.detach()) for key, value in terms.items()})
    suite.check("the total is the weighted sum of the heads",
                abs(float(terms["total"].detach()) - float(terms["topic"].detach())
                    - 0.5 * float(terms["presence"].detach())
                    - float(terms["value"].detach())) < 1e-6)
    terms["total"].backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    suite.check("gradients reach many parameters", len(grads) > 8, len(grads))
    suite.check("no gradient is entirely zero for every tensor",
                sum(1 for g in grads if float(g.abs().sum()) > 0) > 8)
    suite.check("the FiLM path starts at the identity",
                bool(torch.allclose(module.decoder.film.weight, torch.zeros_like(module.decoder.film.weight))))

    # ---- decoding ---------------------------------------------------------
    decoded = greedy_decode(module, pooled[:4], 0, char_vocab, max_chars, torch)
    suite.check("greedy decode returns one string per case", len(decoded) == 4, decoded)
    suite.check("greedy decode respects the budget",
                all(len(value) <= max_chars for value in decoded), decoded)
    suite.check("decoded text stays inside the train alphabet",
                all(char_vocab.covers(value) for value in decoded), decoded)
    raw, gated = predict_slots(module, pooled[:4], char_vocab, max_chars, torch)
    suite.check("predict_slots covers every value slot",
                sorted(raw) == sorted(VALUE_SLOTS) and sorted(gated) == sorted(VALUE_SLOTS))
    suite.check("a gated prediction is either the decode or the empty marker",
                all(gated[key][row] in (EMPTY, raw[key][row])
                    for key in VALUE_SLOTS for row in range(4)))
    topics = predict_topics(module, pooled[:4], topic_vocab, torch)
    suite.check("topic predictions come from the vocabulary",
                all(name in topic_vocab.stoi for name in topics), topics)

    # ---- the ablation conditions are what they claim to be ---------------
    order = wrong_state_shift(5, 3, torch)
    suite.check("the wrong-state permutation is a derangement of the indices",
                sorted(order.tolist()) == [0, 1, 2, 3, 4]
                and all(int(order[i]) != i for i in range(5)), order.tolist())
    suite.check("the wrong-state permutation is reproducible",
                bool((wrong_state_shift(5, 3, torch) == order).all()))
    suite.check("a single case has no wrong-state permutation",
                wrong_state_shift(1, 3, torch) is None)

    eval_references = build_references(dev, reach)
    suite.check("references carry their reachability flag",
                [reference["_reachable"] for reference in eval_references] == [True, False])
    dev_block = torch.randn(len(dev), 12)
    shift = wrong_state_shift(len(dev), 1, torch)
    evaluation = evaluate(module, dev_block, eval_references, topic_vocab, char_vocab,
                          max_chars, torch, shift)
    suite.check("every conditioning condition is scored",
                sorted(evaluation["conditions"]) == sorted(CONDITIONS))
    suite.check("every condition keeps one hit per case",
                all(len(evaluation["conditions"][name]["four_slot_hits"]) == len(dev)
                    for name in CONDITIONS))
    suite.check("the paired comparison covers every case",
                all(evaluation["paired"][key]["wins"] + evaluation["paired"][key]["losses"]
                    + evaluation["paired"][key]["ties"] == len(dev)
                    for key in evaluation["paired"]))
    zero_topics = predict_topics(module, torch.zeros_like(dev_block), topic_vocab, torch)
    suite.check("the `zero` condition really conditions on the zero vector",
                zero_topics == [evaluation["conditions"]["zero"]["predictions"][row]["topic"]
                                for row in range(len(dev))])
    wrong_topics = predict_topics(module, dev_block[shift], topic_vocab, torch)
    suite.check("the `wrong` condition really conditions on another case's state",
                wrong_topics == [evaluation["conditions"]["wrong"]["predictions"][row]["topic"]
                                 for row in range(len(dev))])
    suite.check("the `true` condition really conditions on the case's own state",
                [evaluation["conditions"]["true"]["predictions"][row]["topic"]
                 for row in range(len(dev))]
                == predict_topics(module, dev_block, topic_vocab, torch))
    compact = compact_eval(evaluation)
    suite.check("compact_eval drops per-case lists but keeps the metrics",
                "predictions" not in compact["conditions"]["true"]
                and "four_slot_exact" in compact["conditions"]["true"]["presence_gated"])
    suite.check("the summary line names all three conditions",
                all(name in eval_summary_line(evaluation) for name in CONDITIONS))

    # ---- metrics ----------------------------------------------------------
    references = [
        {"topic": "A", "constraint": "c1", "pending": EMPTY, "result": "r1", "_reachable": True},
        {"topic": "B", "constraint": "c2", "pending": "p2", "result": "r2", "_reachable": False},
    ]
    predictions = [
        {"topic": "A", "constraint": "c1", "pending": EMPTY, "result": "r1"},
        {"topic": "B", "constraint": "xx", "pending": "p2", "result": "r2"},
    ]
    scored = score(references, predictions)
    suite.check("four-slot exact counts whole cases",
                scored["four_slot_exact"] == 1 and scored["four_slot_exact_rate"] == 0.5)
    suite.check("per-slot exact is reported separately",
                scored["per_slot_exact"] == {"topic": 2, "constraint": 1, "pending": 2,
                                             "result": 2}, scored["per_slot_exact"])
    suite.check("content slots exclude the empty marker",
                (scored["content_slot_scored"], scored["content_slot_hits"]) == (5, 4),
                (scored["content_slot_scored"], scored["content_slot_hits"]))
    suite.check("content accuracy is hit over scored",
                abs(scored["content_slot_accuracy"] - 0.8) < 1e-12)
    suite.check("the reachable subset is reported separately",
                scored["four_slot_exact_rate_on_reachable"] == 1.0)
    trivial = score([{"topic": "A", "constraint": "c", "pending": EMPTY, "result": "r",
                      "_reachable": True}],
                    [{"topic": "A", "constraint": EMPTY, "pending": EMPTY, "result": EMPTY}])
    suite.check("writing 无 everywhere earns nothing on content slots",
                trivial["content_slot_accuracy"] == 0.0 and trivial["four_slot_exact"] == 0,
                trivial)
    comparison = paired([0, 0, 1], [0, 1, 1])
    suite.check("paired counts wins/losses/ties",
                (comparison["wins"], comparison["losses"], comparison["ties"]) == (1, 0, 2),
                comparison)
    suite.check("paired win rate ignores ties", comparison["win_rate"] == 1.0)
    baselines = constant_baselines(train, references)
    suite.check("majority topic is the most frequent train topic",
                baselines["per_slot"]["topic"] == "电脑价格区间")
    suite.check("majority pending is the empty marker",
                baselines["per_slot"]["pending"] == EMPTY)
    baseline_scored = baseline_report(references, baselines)
    suite.check("the always-empty baseline is scored on the same cases",
                baseline_scored["always_empty_plus_common_topic"]["content_slot_accuracy"] == 0.0)

    # ---- qualification rule ----------------------------------------------
    clean_eval = {"conditions": {"zero": {"presence_gated": {"four_slot_exact_rate": 0.1}}},
                  "paired": {"true_vs_zero": {"wins": 50, "losses": 0, "ties": 10},
                             "true_vs_wrong": {"wins": 50, "losses": 0, "ties": 10}}}
    passed, reasons = judge_qualified({"four_slot_exact_rate": 0.9}, clean_eval,
                                      {"ceiling_all_four_exact": 0.98})
    suite.check("a clean pass is accepted", passed, reasons)
    beaten = {"conditions": {"zero": {"presence_gated": {"four_slot_exact_rate": 0.2}}},
              "paired": {"true_vs_zero": {"wins": 0, "losses": 5, "ties": 0},
                         "true_vs_wrong": {"wins": 5, "losses": 0, "ties": 0}}}
    passed, reasons = judge_qualified({"four_slot_exact_rate": 0.9}, beaten,
                                      {"ceiling_all_four_exact": 0.98})
    suite.check("a run the zero state beats is refused", not passed)
    passed, reasons = judge_qualified({"four_slot_exact_rate": 0.2}, clean_eval,
                                      {"ceiling_all_four_exact": 0.10})
    suite.check("a rate above the vocabulary ceiling is refused",
                not passed and any("ceiling" in reason for reason in reasons), reasons)
    passed, reasons = judge_qualified({"four_slot_exact_rate": 0.9}, clean_eval,
                                      {"ceiling_all_four_exact": 0.50})
    suite.check("a target the ceiling cannot produce is refused",
                not passed and any("ceiling" in reason for reason in reasons), reasons)

    # ---- the training loop itself ----------------------------------------
    # Driven here with tiny CPU tensors and stubbed collaborators. Without this the
    # whole stepping path would first run on a GPU, where a wrong checkpoint cadence
    # or a lost step counter costs a 30-minute round to discover.
    suite.check("the teacher-forcing schedule ramps then holds",
                teacher_forcing_dropout_rate(0, 0.2, 0.8, 4) == 0.2
                and teacher_forcing_dropout_rate(2, 0.2, 0.8, 4) == 0.5
                and teacher_forcing_dropout_rate(4, 0.2, 0.8, 4) == 0.8
                and teacher_forcing_dropout_rate(99, 0.2, 0.8, 4) == 0.8)
    loop_args = argparse.Namespace(steps=6, batch_size=2, seed=5, log_every=2,
                                   eval_every=4, checkpoint_every=3,
                                   tf_dropout_start=0.2, tf_dropout_end=0.8,
                                   tf_dropout_ramp=4)
    loop_module = build_classifier(config, torch)
    loop_optimizer = torch.optim.AdamW(loop_module.parameters(), lr=1e-3)
    checkpoints, evaluations, log_lines = [], [], []
    loop_history = []
    loop_current = {"step": 0, "loss": None, "tf_rate": 0.0}
    final_step = run_training_steps(
        args=loop_args, torch=torch, module=loop_module, optimizer=loop_optimizer,
        weights=weights, train_block=pooled, train_targets=targets,
        train_count=len(train), start_step=0, history=loop_history, current=loop_current,
        checkpoint_at=lambda step, phase: checkpoints.append(step),
        evaluate_at=lambda step, phase: evaluations.append(step),
        before_step=lambda step: None, log=log_lines.append,
        status_update=lambda **changes: None, peak_memory=lambda: 0.0,
        device=torch.device("cpu"))
    suite.check("the loop stops exactly at the requested step", final_step == 6, final_step)
    suite.check("the loop records one history entry per step", len(loop_history) == 6,
                len(loop_history))
    suite.check("history carries every head loss and the gradient norm",
                all({"step", "loss", "topic", "presence", "value", "grad_norm"}
                    <= set(entry) for entry in loop_history))
    suite.check("checkpoints land on their cadence and at the final step",
                checkpoints == [3, 6], checkpoints)
    suite.check("periodic evaluation runs on its own cadence, never twice at the end",
                evaluations == [4], evaluations)
    suite.check("the step counter is published for the guard and the checkpoint",
                loop_current["step"] == 6, loop_current)
    suite.check("the recorded loss is finite", math.isfinite(loop_current["loss"]))
    suite.check("the loop passes the scheduled rate into the batch",
                any("tf=0.800" in line for line in log_lines), log_lines[-1:])
    suite.check("the loop logs the ETA and peak memory fields",
                any("s/step" in line and "peak=" in line for line in log_lines))

    paused_current = {"step": 0, "loss": None, "tf_rate": 0.0}
    paused_history = []

    def pause_after_two(step):
        if step >= 2:
            raise Done({"paused": True, "exit_code": EXIT_PAUSED})

    try:
        run_training_steps(
            args=loop_args, torch=torch, module=loop_module, optimizer=loop_optimizer,
            weights=weights, train_block=pooled, train_targets=targets,
            train_count=len(train), start_step=0, history=paused_history,
            current=paused_current,
            checkpoint_at=lambda step, phase: None,
            evaluate_at=lambda step, phase: None, before_step=pause_after_two,
            log=lambda message: None, status_update=lambda **changes: None,
            peak_memory=lambda: 0.0, device=torch.device("cpu"))
        suite.check("a pause stops the loop", False, "the loop ran to completion")
    except Done:
        suite.check("a pause stops the loop at an optimizer boundary",
                    paused_current["step"] == 2, paused_current)
    suite.check("a paused run does not record a step it did not finish",
                len(paused_history) == 2, len(paused_history))

    # ---- pooling and the standardisation claim ---------------------------
    rows = torch.randn(6, 4)
    mean = pool_rows(rows, "mean")
    meanmax = pool_rows(rows, "meanmax")
    suite.check("mean pooling returns one vector", tuple(mean.shape) == (4,))
    suite.check("mean+max pooling returns two concatenated vectors",
                tuple(meanmax.shape) == (8,))
    suite.check("mean+max begins with the mean",
                bool(torch.allclose(meanmax[:4], mean)))
    matrix = torch.randn(5, 6, 4)
    raw_mean = matrix.mean(dim=1)
    stats_mean = raw_mean.mean(dim=0, keepdim=True)
    stats_std = raw_mean.std(dim=0, keepdim=True).clamp_min(1e-5)
    suite.check("standardisation commutes with mean pooling",
                bool(torch.allclose(((matrix - stats_mean) / stats_std).mean(dim=1),
                                    (raw_mean - stats_mean) / stats_std, atol=1e-5)))
    suite.check("pooled_dim_for matches the pooling it describes",
                pooled_dim_for((72, 2048), "meanmax") == 4096
                and pooled_dim_for((72, 2048), "mean") == 2048)

    # ---- sealed-split guard ----------------------------------------------
    suite.check("a normal cases path is accepted",
                _no_raise(lambda: refuse_sealed_split(DEFAULT_CASES, [{"id": "RS-tr-0001"}])))
    suite.raises("a path named like the final split is refused",
                 lambda: refuse_sealed_split(r"D:\x\memory_test.jsonl", [{"id": "RS-tr-0001"}]))
    suite.raises("a case id from the test namespace is refused",
                 lambda: refuse_sealed_split(DEFAULT_CASES, [{"id": "RS-te-0001"}]))

    # ---- checkpoints ------------------------------------------------------
    temporary = Path(tempfile.mkdtemp(prefix="slot-classifier-selftest-"))
    try:
        checkpoint = temporary / "checkpoint-last.pt"
        # `TRAINER.rng_state` is deliberately NOT called here: on a machine with CUDA
        # it reaches torch.cuda.get_rng_state_all(), which OPENS a CUDA context. The
        # last check in this suite asserts that nothing here created one, so a stub
        # stands in -- and this is why the CPU path must not call that helper.
        stub_rng = {"torch": torch.get_rng_state(), "python": random.getstate()}
        payload = {"format": FORMAT, "identity_hash": "abc", "step": 3, "phase": "self-test",
                   "model_state": {"w": torch.zeros(2)}, "optimizer_state": {},
                   "rng": stub_rng}
        suite.check("a checkpoint saves",
                    bool(save_checkpoint(checkpoint, payload, torch,
                                          pointer=temporary / "latest.json")))
        suite.check("a checkpoint round-trips",
                    load_checkpoint(checkpoint, torch)["step"] == 3)
        suite.raises("a checkpoint from another run is refused",
                     lambda: accept_checkpoint(load_checkpoint(checkpoint, torch), "other",
                                               checkpoint))
        suite.check("a checkpoint from the same run is accepted",
                    accept_checkpoint(load_checkpoint(checkpoint, torch), "abc",
                                      checkpoint)["step"] == 3)
        torch.save({"format": 0, "step": 1}, temporary / "bad.pt")
        suite.raises("a malformed checkpoint is refused",
                     lambda: load_checkpoint(temporary / "bad.pt", torch))

        # ---- read-only feature cache -------------------------------------
        n_layer = 2
        identity = feature_cache_identity(
            {"mode": "frozen-state", "identity": {"sha256": "fixture"}}, "data-hash-1",
            n_layer, None)
        key = feature_cache_key(identity)
        directory = temporary / "features" / key[:16]
        directory.mkdir(parents=True)
        features = torch.randn(2, n_layer * 3, 4)
        shard_ids = ["A", "B"]
        atomic_torch_save(directory / "shard-000000.pt",
                          {"format": FEATURE_CACHE_FORMAT, "key": key, "ids": shard_ids,
                           "observation_tokens": [3, 4], "features": features}, torch)
        write_json(directory / "shard-000000.pt.ids.json",
                   {"format": FEATURE_CACHE_FORMAT, "key": key, "ids": shard_ids,
                    "observation_tokens": [3, 4]})
        write_json(directory / "manifest.json",
                   {"format": FEATURE_CACHE_FORMAT, "cache_key": key, "identity": identity,
                    "shards": ["shard-000000.pt"], "case_count": 2,
                    "feature_shape": [n_layer * 3, 4], "closed": True})
        before = sorted(path.name for path in directory.iterdir())

        def reader(**overrides):
            options = {"data_hash": "data-hash-1", "expected_backbone_sha256": "fixture",
                       "n_layer": n_layer}
            options.update(overrides)
            return ReadOnlyFeatureCache(directory, **options)

        report = reader().open()
        suite.check("the reader reports the cases on disk",
                    report["cases_on_disk"] == 2 and report["closed"], report)
        suite.check("the reader reports itself as read-only", report["read_only"] is True)
        cache = reader()
        cache.open()
        row = cache.row("A")
        suite.check("a row comes back as [rows, width]",
                    tuple(row.shape) == (n_layer * 3, 4))
        suite.check("the row is the one that was stored",
                    bool(torch.allclose(row, features[0])))
        suite.check("an absent case returns None", cache.row("ZZ") is None)
        suite.raises("an incomplete cache is refused, not partially used",
                     lambda: reader().require_all(["A", "Z"]))
        suite.raises("a cache for another dataset is refused",
                     lambda: reader(data_hash="other").open())
        suite.raises("a cache from another backbone is refused",
                     lambda: reader(expected_backbone_sha256="nope").open())
        tampered = temporary / "features" / "tampered"
        tampered.mkdir(parents=True)
        write_json(tampered / "manifest.json",
                   {"format": FEATURE_CACHE_FORMAT, "cache_key": "deadbeef",
                    "identity": identity, "shards": [], "case_count": 0,
                    "feature_shape": [n_layer * 3, 4], "closed": False})
        suite.raises("a manifest whose key contradicts its identity is refused",
                     lambda: ReadOnlyFeatureCache(tampered, "data-hash-1", "fixture").open())
        after = sorted(path.name for path in directory.iterdir())
        suite.check("every refusal left the cache untouched", before == after,
                    (before, after))
        suite.check("the reader refuses to be pointed at a missing cache",
                    not _no_raise(lambda: ReadOnlyFeatureCache(temporary / "nope",
                                                               "data-hash-1", "fixture").open()))
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    # ---- the whole point of a self-test ----------------------------------
    suite.check("no CUDA context was created", not torch.cuda.is_initialized())

    total = suite.passed + len(suite.failures)
    print("self-test: %d/%d checks passed" % (suite.passed, total), flush=True)
    for failure in suite.failures:
        print("  FAIL " + failure, flush=True)
    if verbose and not suite.failures:
        print("  (CPU only: no CUDA context, no GPU work, nothing trained)", flush=True)
    return 0 if not suite.failures else 1


def preflight(args):
    """Check everything the training run depends on, before it can be launched.

    The expensive failure this prevents: a run that passes the AC gate, loads the
    cache, and only then discovers the cache belongs to another dataset -- or that
    the 80% target was unreachable from a train-only vocabulary all along. Every
    check here is one `_run_inner` performs anyway, in the same order, using the same
    helpers; doing them first costs seconds and opens no CUDA context.
    """
    torch = _torch()
    if not args.feature_cache:
        raise TrainingError("--preflight needs --feature-cache")
    items = DATA.all_cases(args.cases)
    refuse_sealed_split(args.cases, items)
    train = [case for case in items if case["group"] == DATA.TRAIN]
    held = [case for case in items if case["group"] == DATA.HELDOUT]
    if not train or not held:
        raise TrainingError("--preflight needs both train and dev cases")
    train_labelled = case_slots(train)
    held_labelled = case_slots(held)
    data_hash = DATA.data_version(items)
    topic_vocab, char_vocab = build_vocabularies(train_labelled)
    train_max = max_value_length(train_labelled)
    max_chars = int(args.value_max_chars) if args.value_max_chars else train_max
    if max_chars < train_max:
        raise TrainingError("--value-max-chars %d is below the longest TRAIN value (%d); "
                            "truncating a label would silently corrupt the loss"
                            % (max_chars, train_max))
    reach = reachability(held_labelled, topic_vocab, char_vocab, max_chars)
    cache = ReadOnlyFeatureCache(args.feature_cache, data_hash,
                                 args.expect_backbone_sha256,
                                 allow_any_backbone=args.allow_any_backbone)
    cache_report = cache.open()
    cache.require_all([case["id"] for case in train + held])
    sample = build_pooled(cache, (train + held)[:4], args.pool)
    targets = encode_targets(train_labelled, topic_vocab, char_vocab, max_chars)
    report = {
        "ok": True, "script": SCRIPT_VERSION, "preflight": True, "self_test": False,
        "label": LABEL, "backbone_training": BACKBONE_TRAINING,
        "cases": DATA.split_counts(items), "data_hash": data_hash,
        "feature_cache": cache_report, "pool": args.pool,
        "pooled_dim": pooled_dim_for(cache.shape, args.pool),
        "topic_classes": len(topic_vocab), "characters": len(char_vocab),
        "decode_budget": max_chars, "longest_train_value": train_max,
        "target_shape": list(targets.values.shape),
        "reachability": {key: value for key, value in reach.items() if key != "per_case"},
        "dev_reachable_baseline": {
            "four_slot_exact_rate_if_perfect": reach["ceiling_all_four_exact"],
            "criterion": "four_slot_exact_rate >= 0.80",
            "criterion_is_reachable": bool((reach["ceiling_all_four_exact"] or 0) >= 0.80),
        },
        "sample_pooled_finite": bool(torch.isfinite(sample).all()),
        "cuda_context_created": bool(torch.cuda.is_initialized()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def accept_checkpoint(payload, identity_hash, path):
    """Take a checkpoint only from the SAME run.

    The identity hash covers the cache key, the data hash, the vocabularies, the
    pooling and the hyperparameters. Continuing a run whose features or labels
    changed would produce a model that is half one experiment and half another,
    with no error anywhere -- so this refuses instead.
    """
    if payload.get("identity_hash") != identity_hash:
        raise TrainingError(
            "checkpoint %s belongs to a different run (identity %s vs %s); resuming would "
            "silently mix two experiments" % (path, payload.get("identity_hash"), identity_hash))
    return payload


def slice_targets(targets, index, torch):
    """A row subset that keeps the target block's own `n_chars` binding."""
    return Targets(targets.topic_ids[index], targets.presence[index],
                   targets.values[index], targets.length).bind(targets.n_chars)


def judge_qualified(headline, evaluation, reach):
    """The pre-registered standard, applied on dev only.

    Not "did it improve". All four have to hold:
      * four-slot exact at or above the bar
      * the train-only vocabulary can actually produce that rate (the ceiling)
      * the real state beats BOTH ablations with no per-case loss
      * zero-state conditioning is strictly worse
    A run that fails any of them is reported as not qualified, whatever the train
    loss did -- the v3 and v4 decoders both drove train loss down while sitting at
    0/1779, which is exactly the outcome this list exists to refuse to call progress.
    """
    reasons = []
    rate = headline["four_slot_exact_rate"] or 0.0
    if rate < 0.80:
        reasons.append("four_slot_exact_rate %.4f < 0.80" % rate)
    ceiling = reach["ceiling_all_four_exact"] or 0.0
    if ceiling < 0.80:
        reasons.append("train-only vocabulary ceiling %.4f < 0.80: 80%% cannot be reached "
                       "by construction" % ceiling)
    for name in ("zero", "wrong"):
        comparison = evaluation["paired"]["true_vs_" + name]
        if comparison["losses"] > 0 or comparison["wins"] == 0:
            reasons.append("true state does not strictly beat %s (w=%d l=%d t=%d)"
                           % (name, comparison["wins"], comparison["losses"], comparison["ties"]))
    if (evaluation["conditions"]["zero"]["presence_gated"]["four_slot_exact_rate"] or 0.0) >= rate:
        reasons.append("zero-state conditioning scores at least as well as the real state")
    return (not reasons), reasons


def run(args, hooks=None):
    """Run the classifier, turning an AC pause into a return value, not a traceback."""
    try:
        return _run_inner(args, hooks or {})
    except Done as done:
        payload = dict(done.result)
        payload["exit_code"] = EXIT_PAUSED
        return payload


def main(argv=None):
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
