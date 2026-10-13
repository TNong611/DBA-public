"""FINAL-ONLY evaluation of a trained RWKV-state memory decoder.

WHAT THIS IS FOR
----------------
`train_rwkv_slots.py` reports a heldout number every few hundred steps, and those
numbers are used to steer training. This script exists so the *qualification*
claim is made somewhere else, once, on a case file that training never opened:

    --checkpoint  a saved trainer checkpoint (opened READ-ONLY, never written)
    --cases       a case file training never saw (e.g. memory_test.jsonl)
    --out-dir     where this evaluation writes its own artifacts

It loads the original checkpoint itself. It does NOT resume training, does not
write to the checkpoint or to the training run directory, and does not select
anything (no best-checkpoint search, no threshold tuning on this split).

THREE ISSUES FOUND IN THE TRAINING CONTRACT (read before trusting a result)
---------------------------------------------------------------------------
1. THE CHECKPOINT DOES NOT CONTAIN THE NORMALISATION STATISTICS. The trainer
   standardises the feature matrix with train-split mean/std computed in memory
   (train_rwkv_slots.py: `matrix = (matrix - mean) / std`) and stores only
   `run_identity.feature_hash`, a SHA-256 prefix of the STANDARDISED train
   matrix. Mean and std themselves are nowhere in the checkpoint.
   Consequences, handled explicitly below:
     * The standardised space is what the model was trained in, so test features
       must be standardised with the TRAIN statistics -- using test statistics
       would silently evaluate a different function. Test statistics are never
       used here.
     * The statistics are therefore RECOVERED from the training run's immutable
       feature cache (rows are stored RAW) and then PROVED correct by recomputing
       the standardised train matrix and matching `run_identity.feature_hash`.
       That equality is a hard gate: mismatch aborts the evaluation.
     * If that cache is gone (no manifest under the training run) the statistics
       are unrecoverable and this script FAILS. It does not guess, and it does
       not substitute test statistics. Known case: the 22-case pilot checkpoint
       `training/rwkv-slots-v2/checkpoint-last.pt` has no feature cache on disk,
       so it cannot be finally evaluated by this script at all.
2. THE TRAINER'S `no_state` DROPS THE STATE'S INFORMATION ENTIRELY: zeros in the
   standardised space. Note that a zero vector in standardised space equals the
   train mean in raw space, so `zero_state` here is "the average state", not
   "no conditioning at all". Mirrored deliberately so the two are comparable.
3. `wrong_state` is a cyclic roll inside the split (each case gets the previous
   case's state). It is a real but WEAK corruption: both cases come from the
   same file and domain mix. A model can beat it while still not reading the
   state in any useful way. Passing this contrast is necessary, not sufficient.

NORMALISATION CONTRACT (self-contained; two sources, in this order)
-------------------------------------------------------------------
The decoder was trained in a standardised feature space, so test features must be
standardised with the TRAINING statistics. Those are not in the checkpoint, so the
evaluator resolves them through one of exactly two paths -- never from the test
split, and never by guessing:

  PATH 1  an exported `normalization.pt` (see export_rwkv_normalization.py)
          Verified against the checkpoint: `identity_hash`, `data_hash`,
          `feature_hash`, `feature_shape`, `train_count`, `feature_version` and the
          live `state_features` fingerprint must all match, and the file's own
          payload digest is recomputed. When the training cache is still on disk the
          statistics are ALSO re-derived from its raw rows and must be bit-identical,
          so path 1 normally carries the same proof as path 2.
          Searched at `--normalization`, else `<checkpoint dir>/normalization.pt`,
          else `<out-dir>/normalization.pt`.

  PATH 2  recovery from the training run's immutable feature cache
          Raw train rows are assembled in `run_identity.case_ids[:train_count]`
          order, the statistics are recomputed with the trainer's own operators, and
          the result must hash to `run_identity.feature_hash`.

  Neither available -> the run FAILS with a message naming what is missing and how
  to produce it. `--require-normalization-file` forbids path 2;
  `--require-cache-cross-check` forbids accepting a file that could not be
  re-derived from the cache. Optional `--train-cases` additionally proves that the
  file handed in really is the training set (data_version == the checkpoint's
  data_hash, and case_ids equal in order); without it that check is reported as
  not performed.

SCORING (four-slot exact rate and nonempty accuracy are reported SEPARATELY)
---------------------------------------------------------------------------
* four_slot_exact_rate: generated text whose 主题/约束/未完成/结果 all equal the
  reference, over every test case.
* nonempty_slot_accuracy: per-slot accuracy restricted to cases whose reference
  slot is NOT the degenerate `无`. Degenerate slots are reported on their own so
  a model that emits `无` everywhere cannot look good: `无`-expected accuracy is
  shown next to it and is never mixed into the nonempty figure.
* paired evidence for state dependence: per-case win/loss/tie counts and a
  bootstrap confidence interval of the paired difference, for true vs zero_state
  and true vs wrong_state, on slots-exact, all-slots-exact and per-case token
  loss. A lower loss alone is NOT accepted as proof; the win rate and the CI
  lower bound must clear their thresholds too, and every criterion is printed
  individually with its observed value.

EXIT CODES (kept distinguishable on purpose -- "not qualified" is not "crashed")
    0  evaluation completed, qualification thresholds met
    1  technical/setup failure (nothing qualified was measured)
    2  evaluation completed, thresholds NOT met
    75 paused by the AC/lease power gate; partial artifacts are not results

AC SAFETY: fail-closed. A direct Windows AC check and a valid short-lived guard
lease (`DBA_TRAIN_CONTROL`) are required before CUDA, during extraction, before
every evaluation batch, and before generation. The guard lease and the AC check
are two independent signals; "cannot read the AC state" is treated as failure,
never as "power is fine".

Run with the CUDA venv and an active guard lease:
  D:\\AI\\DBA\\.venv-rwkv\\Scripts\\python.exe eval_rwkv_slots_checkpoint.py ^
      --checkpoint D:\\AI\\DBA\\training\\risawoz-run-v2\\checkpoint-last.pt ^
      --cases D:\\AI\\DBA\\training\\web-sources\\risawoz\\memory_test.jsonl ^
      --out-dir D:\\AI\\DBA\\training\\final-eval

Recommended beforehand, while the run's feature cache still exists (CPU only), so the
checkpoint stays evaluable even if that cache is later deleted:
  py -3.14 export_rwkv_normalization.py ^
      --checkpoint D:\\AI\\DBA\\training\\risawoz-run-v2\\checkpoint-last.pt ^
      --train-cases D:\\AI\\DBA\\training\\web-sources\\risawoz\\memory_cases.jsonl ^
      --out D:\\AI\\DBA\\training\\risawoz-run-v2\\normalization.pt

CPU-only self tests (no GPU, no checkpoint, no data):
  py -3.14 eval_rwkv_slots_checkpoint.py --self-test
  py -3.14 export_rwkv_normalization.py --self-test
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
POWER_SCRIPTS = Path(r"D:\AI\双脑Agent\scripts")
if str(POWER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(POWER_SCRIPTS))

import rwkv_slots_data as DATA

# The trainer module IS the contract for the decoder, the feature function and
# the cache layout. Re-implementing any of them would let the evaluator drift
# away from what was actually trained, which is the one failure this script
# cannot afford. Imported for definitions only: importing has no side effects.
import train_rwkv_slots as T

try:
    from training_power_control import training_allowed
except Exception:  # pragma: no cover - only where the guard is not installed
    def training_allowed():
        return False

SCRIPT_VERSION = "eval_rwkv_slots_checkpoint.py/1"

EXIT_OK_QUALIFIED = 0
EXIT_FAILED = 1
EXIT_NOT_QUALIFIED = 2
EXIT_PAUSED = 75

# The normalisation contract (see NORMALISATION CONTRACT below).
NORMALIZATION_NAME = "normalization.pt"
NORMALIZATION_FORMAT = 1
NORMALIZATION_TOOL = "export_rwkv_normalization.py/1"

CONDITIONS = ("true", "zero_state", "wrong_state")

SLOTS = DATA.SLOT_KEYS                      # ("topic", "constraint", "pending", "result")
SLOT_LABELS = DATA.SLOT_LABELS
EMPTY_SLOT = DATA.EMPTY_SLOT

RESULT_NAME = "final-eval.json"
PERCASE_NAME = "final-eval-percase.jsonl"
LOG_NAME = "final-eval.log"


class EvalError(Exception):
    """Anything that stops the evaluation before it can report a number."""


class PowerLoss(EvalError):
    def __init__(self, report):
        self.report = report
        super().__init__("%s: guard_lease=%s ac_state=%s reason=%s"
                         % (report.get("stage"), report.get("guard_lease"),
                            report.get("ac_state"), report.get("reason")))


# --------------------------------------------------------------------------
# atomic output
# --------------------------------------------------------------------------
def write_json(path, value):
    return T.write_json(path, value)


# Shared atomic-write helpers re-exported so the exporter beside this script does
# not need to know which module they live in.
atomic_torch_save = T.atomic_torch_save
replace_with_retry = T.replace_with_retry


def write_jsonl(path, rows):
    """Full rewrite, atomic via replace: a reader never sees a torn results file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            import os
            os.fsync(handle.fileno())
        T.replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# power gate (fail-closed)
# --------------------------------------------------------------------------
class _PowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]


def ac_state():
    """One of "ac" / "battery" / "unknown".

    Three values, not a boolean: "the API failed" and "on battery" must not
    arrive through the same channel, or an unreadable power state silently reads
    as "power is fine".
    """
    try:
        status = _PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return "unknown"
    except Exception:
        return "unknown"
    if status.ACLineStatus == 1:
        return "ac"
    if status.ACLineStatus == 0:
        return "battery"
    return "unknown"   # 255 = unknown to Windows itself


def power_report(stage, require_lease=True):
    lease = training_allowed() if require_lease else None
    ac = ac_state()
    report = {"stage": stage, "guard_lease": lease, "ac_state": ac,
              "lease_source": "DBA_TRAIN_CONTROL (training_power_control.training_allowed)"}
    if ac != "ac":
        raise PowerLoss({**report, "reason": "AC state is %r, not connected" % ac})
    if require_lease and not lease:
        raise PowerLoss({**report, "reason": "no valid guard lease"})
    return report


# --------------------------------------------------------------------------
# checkpoint contract verification
# --------------------------------------------------------------------------
def load_and_verify_checkpoint(path, torch, allow_ablated=False):
    """Load the trainer checkpoint and check every field this script relies on."""
    path = Path(path)
    if not path.is_file():
        raise EvalError("checkpoint not found: " + str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != T.FORMAT:
        raise EvalError("checkpoint format invalid (want %r): %s" % (T.FORMAT, path))
    for key in ("identity_hash", "step", "model_state", "run_identity", "state_mode"):
        if key not in payload:
            raise EvalError("checkpoint is missing %s: %s" % (key, path))
    identity = payload["run_identity"]
    if not isinstance(identity, dict):
        raise EvalError("checkpoint run_identity is not an object")

    if identity.get("label") != DATA.LABEL or payload.get("label") != DATA.LABEL:
        raise EvalError("checkpoint label is %r, expected %r (is this a decoder pilot?)"
                        % (identity.get("label"), DATA.LABEL))
    if identity.get("backbone_training") is not DATA.BACKBONE_TRAINING:
        raise EvalError("checkpoint reports backbone_training=%r; this evaluator only "
                        "scores the frozen-backbone pilot" % (identity.get("backbone_training"),))
    if identity.get("backbone_mode") != "frozen-state":
        raise EvalError("checkpoint backbone_mode is %r; a synthetic/dry-run checkpoint "
                        "cannot be finally evaluated" % (identity.get("backbone_mode"),))
    state_mode = payload.get("state_mode") or identity.get("state_mode")
    if state_mode != "true" and not allow_ablated:
        raise EvalError(
            "checkpoint was trained with --state-mode %r, so it is an ablation baseline, "
            "not the model under test; pass --allow-ablated-checkpoint to score it anyway"
            % (state_mode,))
    if identity.get("feature_version") != T.FEATURE_VERSION:
        raise EvalError("checkpoint feature_version=%r but this trainer code is %r; the "
                        "feature rows would mean something different"
                        % (identity.get("feature_version"), T.FEATURE_VERSION))
    decoder = identity.get("decoder")
    if not isinstance(decoder, dict):
        raise EvalError("checkpoint run_identity.decoder is missing")
    missing = [name for name in T.DecoderConfig.FIELDS if name not in decoder]
    if missing:
        raise EvalError("decoder config is missing fields: " + ", ".join(missing))
    case_ids = identity.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids:
        raise EvalError("checkpoint run_identity.case_ids is missing or empty")
    counts = identity.get("data_counts") or {}
    train_count = counts.get("train")
    if not isinstance(train_count, int) or train_count < 1 or train_count > len(case_ids):
        raise EvalError("checkpoint data_counts.train=%r is not usable" % (counts.get("train"),))
    if not identity.get("feature_hash"):
        raise EvalError("checkpoint run_identity.feature_hash is missing; without it the "
                        "recovered normalisation cannot be verified")
    return {"path": str(path), "payload": payload, "identity": identity,
            "identity_hash": payload["identity_hash"], "step": int(payload["step"]),
            "state_mode": state_mode, "decoder_config": decoder, "case_ids": case_ids,
            "train_count": train_count, "data_counts": counts,
            "feature_hash": identity["feature_hash"], "data_hash": identity.get("data_hash"),
            "script": identity.get("script"), "tokenizer": identity.get("tokenizer") or {},
            "backbone_identity": identity.get("backbone") or {}}


# --------------------------------------------------------------------------
# training feature cache: READ-ONLY reader
# --------------------------------------------------------------------------
class ReadOnlyFeatureCache:
    """Same on-disk layout as `T.FeatureCache`, but it cannot write or delete.

    The training cache is the only surviving copy of the raw rows the training
    statistics were computed from, so it is opened strictly read-only. In
    particular `T.FeatureCache._discard()` deletes shards on a validation
    failure and must never run against a training artifact.
    """

    def __init__(self, root, key, identity, torch):
        self.root = Path(root)
        self.key = key
        self.identity = identity
        self.torch = torch
        self.index = {}          # case id -> (shard number, row, tokens)
        self.shards = []
        self.shape = None
        self.manifest = None
        self._open_name = None
        self._open_tensor = None

    def load(self):
        path = self.root / T.FEATURE_CACHE_MANIFEST
        if not self.root.is_dir():
            raise EvalError("training feature cache directory not found: " + str(self.root))
        if not path.is_file():
            raise EvalError(
                "training feature cache has no %s under %s; the train-split "
                "normalisation statistics are unrecoverable from this checkpoint"
                % (T.FEATURE_CACHE_MANIFEST, self.root))
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EvalError("training cache manifest is unreadable: %s" % exc)
        if not isinstance(manifest, dict) or manifest.get("format") != T.FEATURE_CACHE_FORMAT:
            raise EvalError("training cache manifest format mismatch")
        if manifest.get("cache_key") != self.key:
            raise EvalError(
                "training cache key mismatch: this code expects %s but the cache holds %s. "
                "The backbone identity, FEATURE_VERSION, `state_features` source or data hash "
                "changed, so those cached rows mean something different now.\n"
                "  expected identity: %s\n  stored identity:   %s"
                % (self.key[:16], str(manifest.get("cache_key"))[:16],
                   json.dumps(self.identity, ensure_ascii=False, sort_keys=True),
                   json.dumps(manifest.get("identity"), ensure_ascii=False, sort_keys=True)))
        if manifest.get("identity") != self.identity:
            raise EvalError("training cache identity differs from the reconstructed identity")
        seen = set()
        for number, name in enumerate(manifest.get("shards") or []):
            if not isinstance(name, str) or Path(name).name != name:
                raise EvalError("shard name is not a plain file name: %r" % (name,))
            if not (self.root / name).is_file():
                raise EvalError("training cache shard is missing: " + name)
            index = json.loads((self.root / (name + ".ids.json")).read_text(encoding="utf-8"))
            if index.get("key") != self.key:
                raise EvalError("shard %s belongs to another cache key" % name)
            ids = index.get("ids")
            tokens = index.get("observation_tokens")
            if not isinstance(ids, list) or not isinstance(tokens, list) or len(ids) != len(tokens):
                raise EvalError("shard %s index is malformed" % name)
            for row, case_id in enumerate(ids):
                if case_id in seen:
                    raise EvalError("case %r appears twice in the training cache" % (case_id,))
                seen.add(case_id)
                self.index[case_id] = (number, row, int(tokens[row]))
            self.shards.append({"name": name, "ids": ids, "tokens": tokens})
        shape = manifest.get("feature_shape")
        self.shape = None if not shape else (int(shape[0]), int(shape[1]))
        self.manifest = manifest
        return sorted(seen)

    def _open_shard(self, number):
        shard = self.shards[number]
        if self._open_name != shard["name"]:
            payload = self.torch.load(self.root / shard["name"], map_location="cpu",
                                      weights_only=False)
            if not isinstance(payload, dict) or payload.get("format") != T.FEATURE_CACHE_FORMAT:
                raise EvalError("training cache shard is not loadable: " + shard["name"])
            if payload.get("key") != self.key:
                raise EvalError("training cache shard key mismatch: " + shard["name"])
            features = payload.get("features")
            if not hasattr(features, "dim") or features.dim() != 3:
                raise EvalError("training cache shard holds no [cases, rows, width] matrix: "
                                + shard["name"])
            if int(features.shape[0]) != len(shard["ids"]):
                raise EvalError("training cache shard rows do not match its id list: "
                                + shard["name"])
            self._open_name = shard["name"]
            self._open_tensor = features.to(self.torch.float32)
        return self._open_tensor

    def row(self, case_id):
        entry = self.index.get(case_id)
        if entry is None:
            return None, None
        number, row, tokens = entry
        return self._open_shard(number)[row], tokens

    def describe(self):
        return {"root": str(self.root), "cache_key": self.key[:16],
                "shards": len(self.shards),
                "cases_on_disk": sum(len(shard["ids"]) for shard in self.shards),
                "feature_shape": list(self.shape) if self.shape else None,
                "closed_manifest": bool((self.manifest or {}).get("closed")),
                "access": "read-only (never flushed, never discarded)"}


def recover_train_statistics(reader, case_ids, train_count, torch, expected_hash):
    """Rebuild train mean/std from raw cached rows and PROVE they are the ones used.

    The trainer computed, in this order:
        mean = matrix[:train_count].mean(dim=(0, 1), keepdim=True)
        std  = matrix[:train_count].std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
        matrix = (matrix - mean) / std
        feature_hash = sha256(matrix.float().numpy().tobytes())[:32]
    Reproducing the same rows in the same order with the same operators makes the
    final hash an equality test rather than a hope: if the recomputed
    standardised train matrix hashes to the checkpoint's `feature_hash`, the
    statistics in hand ARE the training statistics.
    """
    wanted = list(case_ids[:train_count])
    rows = []
    missing = []
    for case_id in wanted:
        row, _ = reader.row(case_id)
        if row is None:
            missing.append(case_id)
        else:
            rows.append(row)
    if missing:
        raise EvalError(
            "training feature cache is incomplete: %d of %d training cases are missing "
            "(first: %s); the train statistics cannot be reconstructed from a partial cache"
            % (len(missing), len(wanted), ", ".join(missing[:5])))
    matrix = torch.stack(rows, dim=0).to(torch.float32)
    del rows
    if reader.shape is not None and tuple(matrix.shape[1:]) != tuple(reader.shape):
        raise EvalError("recovered matrix shape %s does not match cache shape %s"
                        % (tuple(matrix.shape[1:]), tuple(reader.shape)))
    mean = matrix.mean(dim=(0, 1), keepdim=True)
    std = matrix.std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
    standardised = (matrix - mean) / std
    digest = hashlib.sha256(
        standardised.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:32]
    ok = (digest == expected_hash)
    report = {"rows": len(wanted), "expected_feature_hash": expected_hash,
              "recomputed_feature_hash": digest, "verified": ok,
              "shape": [int(value) for value in matrix.shape],
              "std_min": float(std.min().item()), "std_max": float(std.max().item())}
    del matrix, standardised
    if not ok:
        raise EvalError(
            "recovered train-split statistics do NOT reproduce the checkpoint's "
            "feature_hash (%s != %s). Refusing to evaluate: the features would be "
            "standardised differently from training, so every number below would be "
            "about a different model." % (digest, expected_hash))
    return mean, std, report


# --------------------------------------------------------------------------
# test features
# --------------------------------------------------------------------------
def feature_key_for(engine_identity, data_hash, n_layer):
    """Feature-cache identity/key for one case set.

    Two different data hashes are needed in a run: the TRAINING one (to find and
    verify the training run's own cache directory) and the TEST one (to key this
    evaluation's cache). Sharing one key would mean an edited test file silently
    reusing another test file's cached rows.
    """
    identity = T.feature_cache_identity({"mode": "frozen-state", "identity": engine_identity},
                                        data_hash, n_layer, None)
    return identity, T.feature_cache_key(identity)


# Older name kept so the self-test and any existing caller keep working.
test_feature_key = feature_key_for


def extract_test_features(items, engine, n_layer, torch, cache, gate, every=100):
    """[N, rows, width] raw (unstandardised) test features, extracted on GPU."""
    rows = []
    tokens = {}
    reused = 0
    extracted = 0
    gate("before-test-feature-extraction")
    started = time.perf_counter()
    try:
        for index, case in enumerate(items):
            if index and index % every == 0:
                gate("during-test-feature-extraction")
            case_id = case["id"]
            row, count = cache.lookup(case_id)
            if row is None:
                state, count = engine.observe(None, DATA.observation_text(case))
                row = T.state_features(state, n_layer, torch).to("cpu")
                cache.add(case_id, row, count)
                del state
                extracted += 1
            else:
                reused += 1
            rows.append(row)
            tokens[case_id] = int(count)
        gate("after-test-feature-extraction")
    finally:
        # An AC pause leaves through this frame: the rows still in memory are
        # flushed first, so a restart re-uses them instead of paying for the
        # frozen backbone again. At most `every` rows can be lost to a hard kill.
        cache.close()
    matrix = torch.stack(rows, dim=0).to(torch.float32)
    del rows
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return matrix, tokens, {"seconds": round(time.perf_counter() - started, 3),
                            "reused_from_cache": reused, "newly_extracted": extracted}


def verify_training_cases(path, checkpoint):
    """Prove that a case file really is the set the checkpoint was trained on.

    `all_cases` is what the trainer loaded, so it is what the comparison must use:
    it includes the contract module's embedded cases and applies the same sort. The
    check is exact in both directions -- data_version (content) and the ordered id
    list (composition and order), because the training statistics are a reduction
    over that exact ordered slice.
    """
    items = DATA.all_cases(str(path))
    ids = [case["id"] for case in items]
    version = DATA.data_version(items)
    expected_ids = [str(value) for value in checkpoint["case_ids"]]
    return {"path": str(path), "sha256": sha256_file(path), "cases": len(items),
            "split_counts": DATA.split_counts(items),
            "data_version": version,
            "matches_checkpoint_data_hash": version == checkpoint["data_hash"],
            "case_ids_match_in_order": ids == expected_ids,
            "case_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:32],
            "matches_checkpoint_counts": DATA.split_counts(items) == checkpoint["data_counts"]}


def normalization_payload_digest(mean, std, torch):
    """Integrity digest of the two tensors the file exists to carry.

    Separate from provenance: this catches a corrupted or hand-edited file, not a
    file that belongs to another run (the identity checks do that).
    """
    digest = hashlib.sha256()
    for tensor in (mean, std):
        digest.update(tensor.detach().to(torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()[:32]


def load_normalization_file(path, checkpoint, torch, expected_shape):
    """Load `normalization.pt` and check every identity field against the checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise EvalError("normalization file not found: " + str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise EvalError("normalization file is not a mapping: " + str(path))
    if payload.get("format") != NORMALIZATION_FORMAT:
        raise EvalError("normalization format mismatch: %r != %r"
                        % (payload.get("format"), NORMALIZATION_FORMAT))
    mean = payload.get("mean")
    std = payload.get("std")
    for name, tensor in (("mean", mean), ("std", std)):
        if not hasattr(tensor, "shape"):
            raise EvalError("normalization file %s is not a tensor" % name)
    # The statistics are a reduction over the case and row axes, so the stored
    # tensors are [1, 1, feature_dim] and broadcast over [cases, rows, feature_dim].
    # `expected_shape` is the checkpoint's full (feature_count, feature_dim).
    want = (1, 1, int(expected_shape[1]))
    if tuple(mean.shape) != want or tuple(std.shape) != want:
        raise EvalError(
            "normalization tensor shape %s/%s is not %s (a per-feature-channel mean/std "
            "broadcast over [cases, rows, %d]); the checkpoint's feature shape is %s"
            % (tuple(mean.shape), tuple(std.shape), want, int(expected_shape[1]),
               list(expected_shape)))
    mean = mean.to(torch.float32)
    std = std.to(torch.float32)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise EvalError("normalization file holds non-finite statistics")
    if bool((std <= 0).any()):
        raise EvalError("normalization file holds a non-positive std")

    digest = normalization_payload_digest(mean, std, torch)
    stored_digest = payload.get("payload_digest")
    if stored_digest is not None and stored_digest != digest:
        raise EvalError("normalization payload digest mismatch (%s != %s): the file was "
                        "edited or corrupted after export" % (digest, stored_digest))
    identity = payload.get("identity") or {}
    checks = {
        "identity_hash": (identity.get("identity_hash"), checkpoint["identity_hash"]),
        "data_hash": (identity.get("data_hash"), checkpoint["data_hash"]),
        "feature_hash": (identity.get("feature_hash"), checkpoint["feature_hash"]),
        "train_count": (identity.get("train_count"), checkpoint["train_count"]),
        "feature_version": (identity.get("feature_version"), T.FEATURE_VERSION),
        "feature_function": (identity.get("feature_function"),
                             T.feature_function_fingerprint()),
        "feature_shape": (list(identity.get("feature_shape") or []), list(expected_shape)),
    }
    failed = {name: pair for name, pair in checks.items()
              if pair[0] is None or pair[0] != pair[1]}
    report = {"path": str(path), "payload_digest": digest,
              "stored_payload_digest": stored_digest,
              "verified_fields": {name: {"file": pair[0], "checkpoint": pair[1]}
                                  for name, pair in checks.items()},
              "failed_fields": sorted(failed), "format": payload.get("format"),
              "tool": payload.get("tool"), "created_at": payload.get("created_at"),
              "exporter_verification": payload.get("verification") or {},
              "checkpoint_sha256_recorded": identity.get("checkpoint_sha256"),
              "train_cases_recorded": identity.get("train_cases_sha256"),
              "train_cases_verified": identity.get("train_cases_verified"),
              "cache_recorded": identity.get("cache")}
    try:
        now_sha256 = sha256_file(checkpoint["path"])
    except OSError:
        now_sha256 = None
    report["checkpoint_sha256_now"] = now_sha256
    # Informational, deliberately not a hard gate: `checkpoint-last.pt` is a rolling
    # file that later steps legitimately overwrite, while the run identity below
    # stays the same for the whole run.
    report["checkpoint_artifact_unchanged"] = bool(
        identity.get("checkpoint_sha256") is not None
        and identity.get("checkpoint_sha256") == now_sha256)
    if failed:
        raise EvalError(
            "normalization file does not belong to this checkpoint: %s differ (%s). "
            "Standardising with it would evaluate a different function."
            % (", ".join(sorted(failed)),
               "; ".join("%s file=%r checkpoint=%r" % (name, pair[0], pair[1])
                         for name, pair in sorted(failed.items()))))
    return mean, std, report


def resolve_normalisation(args, checkpoint, torch, engine_identity, feature_shape, log):
    """Obtain TRAIN-only statistics: exported file first, else the training cache.

    Order and refusals are the documented contract; every branch records how strong
    its evidence is, so a reader never has to guess whether "verified" meant
    "re-derived from raw rows" or "the identity fields matched".
    """
    n_layer = feature_shape[0] // 3
    candidates = []
    if args.normalization:
        candidates.append(Path(args.normalization))
    else:
        candidates.append(Path(args.checkpoint).resolve().parent / NORMALIZATION_NAME)
        candidates.append(Path(args.out_dir) / NORMALIZATION_NAME)
    existing = [path for path in candidates if path.is_file()]

    train_cases_check = None
    if args.train_cases:
        train_cases_check = verify_training_cases(args.train_cases, checkpoint)
        if not train_cases_check["matches_checkpoint_data_hash"]:
            raise EvalError(
                "--train-cases is not the training set of this checkpoint: its data_version "
                "is %s but the checkpoint recorded %s. (For a run whose source file was "
                "edited afterwards, pass --train-cases only if you still hold the original "
                "revision.)" % (train_cases_check["data_version"][:16],
                                str(checkpoint["data_hash"])[:16]))
        if not train_cases_check["case_ids_match_in_order"]:
            raise EvalError("--train-cases has the same content hash but a different case "
                            "order or composition than the checkpoint recorded")
        log("--train-cases verified: data_version and case_ids match the checkpoint")

    cache_identity, cache_key = feature_key_for(engine_identity, checkpoint["data_hash"], n_layer)
    cache_dir = (Path(args.train_cache_root) if args.train_cache_root
                 else Path(args.checkpoint).resolve().parent) / T.FEATURE_CACHE_DIR / cache_key[:16]
    cache_usable = (cache_dir / T.FEATURE_CACHE_MANIFEST).is_file()
    if cache_usable:
        reader = ReadOnlyFeatureCache(cache_dir, cache_key, cache_identity, torch)
        reader.load()
        missing = [case_id for case_id in checkpoint["case_ids"][:checkpoint["train_count"]]
                   if case_id not in reader.index]
        if missing:
            log("training cache at %s is incomplete (%d rows missing); it cannot be used"
                % (cache_dir, len(missing)))
            cache_usable = False
        else:
            cache_report = reader.describe()
    else:
        reader = None
        cache_report = None

    if existing:
        path = existing[0]
        mean, std, report = load_normalization_file(path, checkpoint, torch, feature_shape)
        report["source"] = "exported-normalization-file"
        report["train_cases_check"] = train_cases_check
        report["used_test_split_statistics"] = False
        if cache_usable and not args.no_normalization_cross_check:
            # Cheapest possible doubt removal: re-derive the statistics from the raw
            # rows right now and demand bit-identical agreement. A file that fails
            # this is either from another run or was edited -- both would silently
            # change the function under test.
            cache_mean, cache_std, recovery = recover_train_statistics(
                reader, checkpoint["case_ids"], checkpoint["train_count"], torch,
                checkpoint["feature_hash"])
            same = bool(torch.equal(mean, cache_mean) and torch.equal(std, cache_std))
            report["cache_cross_check"] = "checked"
            report["cache_cross_check_identical"] = same
            report["recovery"] = recovery
            report["train_cache"] = cache_report
            report["evidence"] = ("statistics re-derived from the training cache's raw rows "
                                  "and bit-identical to the file; identity fields verified "
                                  "against the checkpoint")
            log("normalization file %s cross-checked against the training cache: identical=%s"
                % (path, same))
            if not same:
                raise EvalError(
                    "the exported normalization file and the statistics re-derived from the "
                    "training cache are NOT bit-identical; refusing to guess which one the "
                    "training used")
        elif cache_usable:
            report["cache_cross_check"] = "skipped-by-flag"
            report["evidence"] = ("exporter verified the statistics against the cache's raw "
                                  "rows; identity fields re-checked now, but the raw rows were "
                                  "not re-derived for this evaluation (--no-normalization-"
                                  "cross-check)")
        else:
            report["cache_cross_check"] = "unavailable"
            report["evidence"] = ("exporter verified the statistics against the cache's raw "
                                  "rows at export time; that cache is gone now, so the file "
                                  "rests on its identity fields and payload digest")
            log("normalization file %s used without a cache cross-check (cache unavailable)"
                % path)
            if args.require_cache_cross_check:
                raise EvalError(
                    "--require-cache-cross-check was given but the training cache is unusable "
                    "at %s, so the file at %s could not be re-derived from raw rows"
                    % (cache_dir, path))
        log("normalization loaded from %s (%s)" % (path, report["cache_cross_check"]))
        return mean, std, report

    if cache_usable:
        mean, std, recovery = recover_train_statistics(
            reader, checkpoint["case_ids"], checkpoint["train_count"], torch,
            checkpoint["feature_hash"])
        report = {"source": "training-feature-cache-recovery",
                  "evidence": "raw train rows assembled in the checkpoint's own case order; "
                              "statistics recomputed with the trainer's operators; the "
                              "standardised matrix hashes to the checkpoint's feature_hash",
                  "train_cache": cache_report, "recovery": recovery,
                  "train_cases_check": train_cases_check,
                  "used_test_split_statistics": False,
                  "normalization_file_seen": [str(path) for path in existing],
                  "why": "the standardised space is what the decoder was trained in; test "
                         "statistics would silently change the function under test"}
        log("normalization recovered from the training cache and VERIFIED against "
            "feature_hash=%s (%d train rows)" % (recovery["recomputed_feature_hash"],
                                                 recovery["rows"]))
        return mean, std, report

    # Neither path: explain precisely what is missing and how to fix it.
    hint = ("export_rwkv_normalization.py --checkpoint %s --train-cases <the ORIGINAL "
            "training cases file> --out <checkpoint dir>/%s"
            % (checkpoint["path"], NORMALIZATION_NAME))
    if args.require_normalization_file:
        raise EvalError("--require-normalization-file was given but no %s was found at %s"
                        % (NORMALIZATION_NAME, ", ".join(str(path) for path in candidates)))
    raise EvalError(
        "the train-split normalisation statistics are unavailable: no %s at %s, and the "
        "training feature cache is unusable at %s. Without them the test features cannot be "
        "standardised the way training standardised them, and using the test split's own "
        "statistics would silently evaluate a different function.\n"
        "  Fix: %s\n"
        "  (the cache must be the one written by THIS run: key %s, i.e. the same backbone, "
        "FEATURE_VERSION, state_features source and data hash %s)"
        % (NORMALIZATION_NAME, ", ".join(str(path) for path in candidates), cache_dir,
           hint, cache_key[:16], str(checkpoint["data_hash"])[:16]))


def standardise(matrix, mean, std, expected_shape):
    if tuple(matrix.shape[1:]) != tuple(expected_shape):
        raise EvalError("test feature shape %s does not match the checkpoint's %s"
                        % (tuple(matrix.shape[1:]), tuple(expected_shape)))
    return (matrix - mean) / std


def build_conditions(matrix, torch):
    """true | zero_state | wrong_state, mirroring the trainer's ablation exactly.

    Applied to the STANDARDISED matrix, as the trainer does for its heldout
    slice. zero_state is therefore the standardised origin, i.e. the train mean
    in raw space; wrong_state is the cyclic roll by one case.
    """
    return {"true": matrix,
            "zero_state": torch.zeros_like(matrix),
            "wrong_state": matrix.roll(1, dims=0)}


# --------------------------------------------------------------------------
# teacher-forced per-case loss (batched, token-weighted)
# --------------------------------------------------------------------------
def teacher_forced_per_case(module, items, features, tokenizer, ids, torch, device,
                           batch_size, gate):
    """Per-case CE sum and scored-token count, plus the aggregate.

    Per-case values exist so the state contrasts can be compared PAIRED (the same
    case under true/zero/wrong), which is what a win/loss count and a bootstrap
    interval need. Aggregating to one number first would throw that away.
    """
    functional = torch.nn.functional
    sequences = [T.encode_sequence(tokenizer, DATA.memory_text(case), ids) for case in items]
    sums = []
    counts = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            gate("during-evaluation")
            chunk = sequences[start:start + batch_size]
            inputs, targets, mask = T.pad_batch(chunk, ids["pad"], torch, device)
            batch_features = features[start:start + batch_size].to(device)
            logits = module(inputs, batch_features, mask)
            losses = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                ignore_index=ids["pad"], reduction="none")
            losses = losses.reshape(targets.shape)
            keep = (targets != ids["pad"])
            row_sums = (losses * keep).sum(dim=1)
            row_counts = keep.sum(dim=1)
            for row in range(row_sums.shape[0]):
                sums.append(float(row_sums[row].item()))
                counts.append(int(row_counts[row].item()))
            del logits, losses, inputs, targets, mask, batch_features, keep
    if sum(counts) < 1:
        raise EvalError("evaluation scored no tokens; refusing to report a fake loss")
    return {"per_case_sum": sums, "per_case_tokens": counts,
            "tokens_scored": sum(counts), "sum": sum(sums),
            "token_loss": sum(sums) / sum(counts)}


# --------------------------------------------------------------------------
# greedy generation
# --------------------------------------------------------------------------
def generate_batch(module, features, tokenizer, ids, max_new_tokens, torch, device,
                   batch_size, gate):
    """Greedy decode for every row of `features`, in width-bounded batches.

    The output head is `vocab_size` wide, so logits for one batch occupy
    [B, width, 65533] floats -- the batch size is a memory limit, not a
    throughput knob. Rows finish independently; finished rows keep receiving the
    pad id and are ignored from then on.
    """
    rows = int(features.shape[0])
    results = [""] * rows
    module.eval()
    with torch.no_grad():
        for start in range(0, rows, batch_size):
            gate("during-generation")
            stop = min(rows, start + batch_size)
            block = features[start:stop].to(device)
            count = stop - start
            sequences = [[ids["bos"]] for _ in range(count)]
            finished = [False] * count
            for _ in range(max_new_tokens):
                width = max(len(sequence) for sequence in sequences)
                inputs = torch.full((count, width), ids["pad"], dtype=torch.long, device=device)
                mask = torch.ones((count, width), dtype=torch.bool, device=device)
                for row, sequence in enumerate(sequences):
                    length = len(sequence)
                    inputs[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
                    mask[row, :length] = False
                logits = module(inputs, block, mask)
                for row in range(count):
                    if finished[row]:
                        continue
                    position = len(sequences[row]) - 1
                    token = int(torch.argmax(logits[row, position]).item())
                    if token in (ids["eos"], T.EOT_ID) or token >= ids["first"]:
                        finished[row] = True     # never emit a special as text
                    else:
                        sequences[row].append(token)
                del logits, inputs, mask
                if all(finished):
                    break
            for row, sequence in enumerate(sequences):
                results[start + row] = tokenizer.decode(sequence[1:])
            del block
    return results


def generate_per_case(module, features, tokenizer, ids, max_new_tokens, torch, device, gate):
    """Reference implementation: one case at a time, no padding at all.

    Kept as the correctness baseline for the batched path (padding changes
    attention numerics), and used when --generation-batch-size 1 is requested.
    """
    module.eval()
    out = []
    with torch.no_grad():
        for index in range(int(features.shape[0])):
            gate("during-generation")
            row = features[index:index + 1].to(device)
            out.append(T.generate_text(module, row, tokenizer, ids, max_new_tokens,
                                       torch, device))
    return out


def resolve_max_tokens(requested, items, tokenizer):
    """Auto: long enough for the longest reference, so a cap never fakes a miss."""
    if requested and requested > 0:
        return int(requested), "explicit"
    longest = max(len(tokenizer.encode(DATA.memory_text(case))) for case in items)
    return max(16, longest + 8), "auto(longest reference + 8)"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def parse_slots(text):
    """Label -> value, tolerant of full/half-width colons (mirrors DATA._tolerant_slots)."""
    out = {}
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        for label, key in DATA.LABEL_TO_SLOT.items():
            if line.startswith(label + "：") or line.startswith(label + ":"):
                out[key] = line[len(label) + 1:].strip()
                break
    return out


def score_case(generated, expected):
    """Hits from DATA.slot_match (the authoritative metric) plus nonempty masks."""
    match = DATA.slot_match(generated, expected)
    want = parse_slots(expected)
    per_slot = match["per_slot"]
    nonempty = {}
    for key in SLOTS:
        reference = want.get(key)
        if reference is None:
            nonempty[key] = {"scored": False, "reason": "reference slot missing"}
        elif reference == EMPTY_SLOT:
            nonempty[key] = {"scored": False, "reason": "reference slot is the degenerate 无"}
        else:
            nonempty[key] = {"scored": True, "hit": int(per_slot[key])}
    return {"slots_exact": int(match["slots_exact"]), "slots_total": int(match["slots_total"]),
            "all_slots_exact": bool(match["all_slots_exact"]),
            "per_slot": {key: int(per_slot[key]) for key in SLOTS},
            "nonempty": nonempty}


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return None
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
            / denominator)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def summarise_condition(per_case):
    """Rates for one condition, over ALL cases, with the degenerate slots split out."""
    total = len(per_case)
    exact = sum(1 for case in per_case if case["all_slots_exact"])
    slots = sum(case["slots_exact"] for case in per_case)
    per_slot_nonempty = {}
    per_slot_degenerate = {}
    malformed = 0
    for key in SLOTS:
        scored = [case for case in per_case if case["nonempty"][key]["scored"]]
        hits = sum(case["nonempty"][key]["hit"] for case in scored)
        per_slot_nonempty[key] = {
            "scored_cases": len(scored), "hits": hits,
            "accuracy": round(hits / len(scored), 4) if scored else None}
        degenerate = [case for case in per_case
                      if not case["nonempty"][key]["scored"]
                      and case["nonempty"][key]["reason"].startswith("reference slot is")]
        degenerate_hits = sum(case["per_slot"][key] for case in degenerate)
        per_slot_degenerate[key] = {
            "scored_cases": len(degenerate), "hits": degenerate_hits,
            "accuracy": round(degenerate_hits / len(degenerate), 4) if degenerate else None,
            "note": "reference is 无; a model that always prints 无 scores here for free"}
        malformed += sum(1 for case in per_case
                         if case["nonempty"][key]["reason"] == "reference slot missing")
    pooled_scored = sum(entry["scored_cases"] for entry in per_slot_nonempty.values())
    pooled_hits = sum(entry["hits"] for entry in per_slot_nonempty.values())
    pooled_degenerate = sum(entry["scored_cases"] for entry in per_slot_degenerate.values())
    pooled_degenerate_hits = sum(entry["hits"] for entry in per_slot_degenerate.values())
    return {
        "cases": total,
        "four_slot_exact_count": exact,
        "four_slot_exact_rate": round(exact / total, 6) if total else None,
        "four_slot_exact_rate_wilson95": wilson_interval(exact, total),
        "mean_slots_exact_of_4": round(slots / total, 6) if total else None,
        "nonempty_slot_accuracy_pooled": round(pooled_hits / pooled_scored, 6)
        if pooled_scored else None,
        "nonempty_slot_cases": pooled_scored,
        "nonempty_slot_hits": pooled_hits,
        "nonempty_slot_accuracy_by_slot": per_slot_nonempty,
        "degenerate_slot_accuracy_pooled": round(pooled_degenerate_hits / pooled_degenerate, 6)
        if pooled_degenerate else None,
        "degenerate_slot_cases": pooled_degenerate,
        "degenerate_slot_hits": pooled_degenerate_hits,
        "degenerate_slot_accuracy_by_slot": per_slot_degenerate,
        "malformed_reference_slot_instances": malformed,
        "counts_add_up": bool(pooled_scored + pooled_degenerate + malformed
                              == total * len(SLOTS)),
    }


def bootstrap_ci(values, samples, confidence, seed):
    """Percentile bootstrap of the mean of `values` (a plain paired difference)."""
    if not values or samples <= 0:
        return None
    count = len(values)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()

    def pick(quantile):
        index = int(round(quantile * (samples - 1)))
        return means[max(0, min(samples - 1, index))]

    low = (1.0 - confidence) / 2.0
    return {"mean": round(statistics.fmean(values), 6),
            "ci_low": round(pick(low), 6), "ci_high": round(pick(1.0 - low), 6),
            "confidence": confidence, "samples": samples, "seed": seed,
            "resampling_unit": "case (paired: the same case under both conditions)"}


def paired_contrast(name, baseline_name, true_cases, other_cases, samples, confidence, seed):
    """Paired per-case comparison of `true` against one altered-state condition."""
    if len(true_cases) != len(other_cases):
        raise EvalError("paired comparison needs equal case counts")
    slots_delta = [a["slots_exact"] - b["slots_exact"] for a, b in zip(true_cases, other_cases)]
    exact_delta = [int(a["all_slots_exact"]) - int(b["all_slots_exact"])
                   for a, b in zip(true_cases, other_cases)]
    loss_delta = [a["token_loss"] - b["token_loss"] for a, b in zip(true_cases, other_cases)]
    wins = sum(1 for value in slots_delta if value > 0)
    losses = sum(1 for value in slots_delta if value < 0)
    ties = sum(1 for value in slots_delta if value == 0)
    decided = wins + losses
    return {
        "comparison": "true vs " + baseline_name,
        "cases": len(true_cases),
        "slots_exact": {"wins": wins, "losses": losses, "ties": ties,
                        "win_rate_all_cases": round(wins / len(true_cases), 6),
                        "win_rate_decided_cases": round(wins / decided, 6) if decided else None,
                        "mean_delta": round(statistics.fmean(slots_delta), 6),
                        "bootstrap": bootstrap_ci(slots_delta, samples, confidence, seed)},
        "all_slots_exact": {"wins": sum(1 for value in exact_delta if value > 0),
                            "losses": sum(1 for value in exact_delta if value < 0),
                            "mean_delta": round(statistics.fmean(exact_delta), 6),
                            "bootstrap": bootstrap_ci(exact_delta, samples, confidence, seed)},
        "per_case_token_loss": {"wins": sum(1 for value in loss_delta if value < 0),
                                "losses": sum(1 for value in loss_delta if value > 0),
                                "mean_delta_true_minus_other": round(statistics.fmean(loss_delta), 6),
                                "bootstrap": bootstrap_ci(loss_delta, samples, confidence, seed),
                                "note": "negative favours true; a lower loss alone is not "
                                        "accepted as evidence of state dependence"},
        "paired_by": name,
    }


# --------------------------------------------------------------------------
# qualification criteria
# --------------------------------------------------------------------------
def judge(conditions, contrasts, thresholds):
    """Every criterion is reported with its observed value, pass or fail.

    Deliberately NOT one blended score: a single number invites "it improved" when
    only the loss moved. The exact-match rate is the primary claim; the paired
    contrasts are the state-dependence claim; each is checked on its own.
    """
    criteria = []

    def add(name, observed, threshold, passed, detail):
        criteria.append({"criterion": name, "observed": observed, "threshold": threshold,
                         "passed": bool(passed), "detail": detail})

    exact = conditions["true"]["four_slot_exact_rate"]
    add("four_slot_exact_rate", exact, thresholds["exact_rate"],
        exact is not None and exact >= thresholds["exact_rate"],
        "share of test cases whose four slots ALL match the reference")

    for contrast in contrasts:
        label = contrast["comparison"]
        slots = contrast["slots_exact"]
        bootstrap = slots["bootstrap"] or {}
        add("%s: mean paired slots-exact delta" % label, slots["mean_delta"],
            thresholds["slots_delta"], slots["mean_delta"] >= thresholds["slots_delta"],
            "mean of (true - other) slots matched per case, out of 4")
        low = bootstrap.get("ci_low")
        add("%s: bootstrap CI lower bound" % label, low, thresholds["ci_lower"],
            low is not None and low > thresholds["ci_lower"],
            "the paired advantage must not be explainable by resampling noise")
        win_rate = slots["win_rate_decided_cases"]
        add("%s: per-case win rate (decided)" % label, win_rate, thresholds["win_rate"],
            win_rate is not None and win_rate >= thresholds["win_rate"],
            "cases where true beat the altered state, among cases where they differ")
        loss = contrast["per_case_token_loss"]
        add("%s: per-case token loss delta" % label, loss["mean_delta_true_minus_other"],
            thresholds["loss_delta"],
            loss["mean_delta_true_minus_other"] <= thresholds["loss_delta"],
            "negative favours true; necessary, and on its own NEVER sufficient")

    passed = all(entry["passed"] for entry in criteria)
    return {"qualified": passed, "criteria": criteria,
            "criteria_failed": [entry["criterion"] for entry in criteria
                                if not entry["passed"]]}


# --------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------
def evaluate_run(args, hooks=None):
    hooks = hooks or {}
    log_lines = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / LOG_NAME

    def log(message):
        line = "%s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message)
        log_lines.append(line)
        print(line, flush=True)

    def flush_log():
        if log_lines:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(log_lines) + "\n")
            log_lines.clear()

    result = {"ok": True, "script": SCRIPT_VERSION, "phase": "started",
              "label": DATA.LABEL, "backbone_training": DATA.BACKBONE_TRAINING,
              "evaluation_only": True, "writes_to_checkpoint": False,
              "training_resumed": False}

    def gate(stage):
        try:
            return power_report(stage, require_lease=True)
        except PowerLoss as exc:
            result.update({"ok": True, "phase": "paused", "paused": True,
                           "exit_code": EXIT_PAUSED, "power": exc.report,
                           "artifacts": {"log": str(log_path), "status": str(out_dir / RESULT_NAME)}})
            log("power gate stopped the evaluation at %s: "
                + json.dumps(exc.report, ensure_ascii=False))
            flush_log()
            raise Done(result) from None

    result["power_before_cuda"] = gate("before-cuda")
    log("power gate before CUDA: " + json.dumps(result["power_before_cuda"], ensure_ascii=False))

    torch = hooks.get("torch") or __import__("torch")
    if not torch.cuda.is_available():
        raise EvalError("CUDA is required; CPU inference is forbidden for this evaluation")
    if not torch.cuda.is_bf16_supported():
        raise EvalError("CUDA BF16 is required")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # ---- checkpoint ------------------------------------------------------
    checkpoint = load_and_verify_checkpoint(args.checkpoint, torch,
                                            allow_ablated=args.allow_ablated_checkpoint)
    result["checkpoint"] = {key: checkpoint[key] for key in
                            ("path", "step", "identity_hash", "state_mode", "script",
                             "feature_hash", "data_hash", "decoder_config", "train_count")}
    result["checkpoint"]["data_counts"] = checkpoint["data_counts"]
    result["checkpoint"]["sha256"] = sha256_file(checkpoint["path"])
    log("checkpoint step=%d state_mode=%s identity=%s sha256=%s"
        % (checkpoint["step"], checkpoint["state_mode"], checkpoint["identity_hash"][:16],
           result["checkpoint"]["sha256"][:16]))
    config = T.DecoderConfig.from_dict(checkpoint["decoder_config"])
    feature_shape = (config.feature_count, config.feature_dim)
    n_layer = config.feature_count // 3
    if config.feature_count % 3:
        raise EvalError("feature_count %d is not 3 x layers" % config.feature_count)

    # ---- test cases ------------------------------------------------------
    cases_path = Path(args.cases)
    if not cases_path.is_file():
        raise EvalError("cases file not found: " + str(cases_path))
    items = DATA.load_events(str(cases_path))
    if not items:
        raise EvalError("no cases loaded from " + str(cases_path))
    DATA.validate(items)
    if args.max_cases:
        items = items[:args.max_cases]
    test_version = DATA.data_version(items)
    test_digest = sha256_file(cases_path)
    case_ids = [case["id"] for case in items]
    overlap = sorted(set(case_ids) & set(checkpoint["case_ids"]))
    if overlap:
        raise EvalError(
            "REFUSING: %d of %d test case ids also appear in the training run "
            "(first: %s). This file is not an independent test split."
            % (len(overlap), len(case_ids), ", ".join(overlap[:5])))
    result["test_set"] = {
        "path": str(cases_path), "sha256": test_digest, "sha256_of_file_only": True,
        "cases": len(items), "data_version": test_version,
        "split_counts": DATA.split_counts(items),
        "case_id_overlap_with_training_run": len(overlap),
        "equals_training_data_hash": bool(checkpoint["data_hash"] == test_version),
        "note": "unsupervised by training: no checkpoint was selected on this file, and "
                "nothing here is written back into training",
    }
    if checkpoint["data_hash"] == test_version:
        raise EvalError("REFUSING: this case file is the training data (data_version equals "
                        "the checkpoint's data_hash)")
    log("test cases %d data_version=%s sha256=%s"
        % (len(items), test_version[:16], test_digest[:16]))

    # ---- backbone ----------------------------------------------------------
    from nb_rwkv import TorchEngine
    engine = hooks.get("engine") or TorchEngine(args.model, chunk_tokens=args.chunk_tokens,
                                                generation_tokens=1)
    if engine.identity != checkpoint["backbone_identity"]:
        raise EvalError(
            "backbone identity mismatch: the checkpoint was trained against a different "
            "weight file.\n  checkpoint: %s\n  loaded:     %s"
            % (json.dumps(checkpoint["backbone_identity"], ensure_ascii=False, sort_keys=True),
               json.dumps(engine.identity, ensure_ascii=False, sort_keys=True)))
    pipeline = engine.pipeline
    table = getattr(pipeline.tokenizer, "idx2token", {})
    vocab_size = max(table.keys()) + 1 if table else 0
    if not vocab_size:
        raise EvalError("RWKV pipeline exposes no token table")
    tokenizer = T.BackboneTokenizer(pipeline, vocab_size)
    ids = T.special_ids(tokenizer)
    stored_tokenizer = checkpoint["tokenizer"] or {}
    if stored_tokenizer.get("name") != tokenizer.name:
        raise EvalError("tokenizer mismatch: checkpoint %r vs loaded %r"
                        % (stored_tokenizer.get("name"), tokenizer.name))
    if int(stored_tokenizer.get("vocab_size") or -1) != tokenizer.vocab_size:
        raise EvalError("tokenizer vocabulary mismatch: checkpoint %r vs loaded %r"
                        % (stored_tokenizer.get("vocab_size"), tokenizer.vocab_size))
    result["tokenizer"] = {"name": tokenizer.name, "vocab_size": tokenizer.vocab_size,
                           "specials": ids, "verified_against_checkpoint": True,
                           "vocab_size_matches_decoder_config":
                               bool(tokenizer.vocab_size == config.vocab_size)}

    # ---- feature function / train cache -----------------------------------
    feature_function = T.feature_function_fingerprint()
    result["feature_function"] = {"fingerprint": feature_function,
                                  "feature_version": T.FEATURE_VERSION,
                                  "source": "train_rwkv_slots.state_features (imported)"}
    # ---- normalisation (TRAIN-only; file or cache, both strictly verified) --
    mean, std, normalisation = resolve_normalisation(
        args, checkpoint, torch, engine.identity, feature_shape, log)
    result["normalisation"] = normalisation
    result["normalisation"]["contract"] = {
        "order": ["--normalization file (verified against the checkpoint)",
                  "<checkpoint dir>/" + NORMALIZATION_NAME,
                  "<out-dir>/" + NORMALIZATION_NAME,
                  "training feature cache recovery (verified by feature_hash)"],
        "never_used": "test-split statistics",
        "require_normalization_file": bool(args.require_normalization_file),
        "require_cache_cross_check": bool(args.require_cache_cross_check),
        "mean_std_shape": [int(value) for value in mean.shape],
    }
    result["normalisation"]["normalization_file_name"] = NORMALIZATION_NAME

    # ---- test features ----------------------------------------------------
    # A separate key: the test cache must change when the TEST file changes, or an
    # edited test split would silently reuse another split's cached rows.
    test_identity, test_key = test_feature_key(engine.identity, test_version, n_layer)
    test_cache_dir = out_dir / T.FEATURE_CACHE_DIR / test_key[:16]
    test_cache = None
    if args.test_feature_cache:
        test_cache = T.FeatureCache(test_cache_dir, test_key, test_identity, torch,
                                    every=args.test_cache_every)
        cached = test_cache.load()
        if test_cache.discarded:
            log("test feature cache discarded: %s" % test_cache.discarded)
        if cached:
            log("test feature cache: %d/%d cases reused" % (len(cached), len(items)))
    feature_matrix, observation_tokens, extraction = extract_test_features(
        items, engine, n_layer, torch, test_cache, gate, every=100)
    if test_cache is not None:
        extraction["cache"] = test_cache.describe()
    standardised = standardise(feature_matrix, mean, std, feature_shape)
    del feature_matrix
    extraction["standardised_with"] = "recovered train-split statistics (verified)"
    extraction["observation_tokens"] = observation_tokens
    extraction["test_cache_key"] = test_key[:16]
    result["feature_extraction"] = extraction
    log("test features %s (reused=%d extracted=%d) test_cache_key=%s"
        % ([int(v) for v in standardised.shape], extraction["reused_from_cache"],
           extraction["newly_extracted"], test_key[:16]))

    # NO BACKBONE RELEASE HERE, deliberately. `tokenizer.pipeline` holds the same
    # PIPELINE object as the engine, and that object holds the RWKV model, so
    # dropping the engine's references (what the trainer logs as "freed the frozen
    # backbone") does not actually return the tensors. Pretending otherwise would
    # make a later OOM look mysterious, so the backbone stays resident and its
    # cost is reported instead.
    result["memory"] = {"backbone_resident_mib": round(
        torch.cuda.memory_allocated(0) / 1024 ** 2, 1)}

    # ---- model ------------------------------------------------------------
    module = T.build_decoder(config, torch)
    state = module.load_state_dict(checkpoint["payload"]["model_state"], strict=True)
    module = module.to(device)
    module.eval()
    parameters = sum(p.numel() for p in module.parameters())
    if "decoder_parameters" in checkpoint["payload"] and \
            int(checkpoint["payload"]["decoder_parameters"]) != parameters:
        raise EvalError("parameter count mismatch: checkpoint %r vs rebuilt %d"
                        % (checkpoint["payload"]["decoder_parameters"], parameters))
    result["model"] = {"parameters": parameters, "load_state_dict": str(state),
                       "strict": True, "device": str(device),
                       "gpu": torch.cuda.get_device_name(0)}
    log("decoder rebuilt and loaded strictly (%d parameters)" % parameters)

    # ---- conditions -------------------------------------------------------
    max_tokens, max_tokens_basis = resolve_max_tokens(args.eval_max_tokens, items, tokenizer)
    conditions = build_conditions(standardised, torch)
    started = time.perf_counter()
    per_condition = {}
    for name in CONDITIONS:
        features = conditions[name]
        losses = teacher_forced_per_case(module, items, features, tokenizer, ids, torch, device,
                                        args.eval_batch_size, gate)
        # Every condition decodes EVERY case: a rate computed on a sample of cases
        # is not a final number, and all three conditions must cover the same set
        # for the pairing to be valid.
        if args.generation_batch_size > 1:
            generated = generate_batch(module, features, tokenizer, ids, max_tokens, torch,
                                       device, args.generation_batch_size, gate)
        else:
            generated = generate_per_case(module, features, tokenizer, ids, max_tokens, torch,
                                          device, gate)
        if len(generated) != len(items):
            raise EvalError("generation produced %d texts for %d cases"
                            % (len(generated), len(items)))
        cases = []
        for index, case in enumerate(items):
            expected = DATA.memory_text(case)
            score = score_case(generated[index], expected)
            case_row = {"case_id": case["id"], "group": case["group"], "expected": expected,
                        "generated": generated[index],
                        "token_loss": round(losses["per_case_sum"][index]
                                            / max(1, losses["per_case_tokens"][index]), 6),
                        "tokens_scored": losses["per_case_tokens"][index],
                        "reference_tokens": len(tokenizer.encode(expected))}
            case_row.update(score)
            # A generation that consumed the whole budget may simply have been cut
            # off, which would score as a miss for a reason that is not the model's
            # fault; both facts are recorded so a suspicious rate can be checked.
            case_row["hit_generation_cap"] = bool(
                len(tokenizer.encode(case_row["generated"])) >= max_tokens)
            cases.append(case_row)
        summary = summarise_condition(cases)
        summary["token_loss"] = round(losses["token_loss"], 6)
        summary["tokens_scored"] = losses["tokens_scored"]
        summary["reference_exceeds_cap"] = sum(1 for case in cases
                                               if case["reference_tokens"] > max_tokens)
        summary["generation_reached_cap"] = sum(1 for case in cases if case["hit_generation_cap"])
        summary["empty_generations"] = sum(1 for case in cases if not case["generated"].strip())
        per_condition[name] = {"summary": summary, "cases": cases}
        log("%-11s four_slot_exact=%.4f nonempty_accuracy=%s token_loss=%.4f"
            % (name, summary["four_slot_exact_rate"], summary["nonempty_slot_accuracy_pooled"],
               summary["token_loss"]))
    preprocess_seconds = round(time.perf_counter() - started, 3)

    # ---- paired contrasts -------------------------------------------------
    true_cases = per_condition["true"]["cases"]
    contrasts = []
    for name in ("zero_state", "wrong_state"):
        contrasts.append(paired_contrast(name, name, true_cases, per_condition[name]["cases"],
                                        args.bootstrap_samples, args.bootstrap_confidence,
                                        args.bootstrap_seed))

    thresholds = {"exact_rate": args.threshold_exact_rate,
                  "slots_delta": args.threshold_slots_delta,
                  "ci_lower": args.threshold_ci_lower,
                  "win_rate": args.threshold_win_rate,
                  "loss_delta": args.threshold_loss_delta,
                  "bootstrap_samples": args.bootstrap_samples,
                  "bootstrap_confidence": args.bootstrap_confidence,
                  "bootstrap_seed": args.bootstrap_seed}
    verdict = judge({name: per_condition[name]["summary"] for name in CONDITIONS},
                    contrasts, thresholds)

    # ---- artifacts --------------------------------------------------------
    percase_path = out_dir / PERCASE_NAME
    rows = []
    for index, case in enumerate(items):
        row = {"case_id": case["id"], "group": case["group"],
               "expected": DATA.memory_text(case), "conditions": {}}
        for name in CONDITIONS:
            entry = per_condition[name]["cases"][index]
            row["conditions"][name] = {key: entry[key] for key in
                                       ("generated", "slots_exact", "per_slot",
                                        "all_slots_exact", "nonempty", "token_loss",
                                        "tokens_scored")}
        row["paired"] = {
            "true_minus_zero_state_slots": (per_condition["true"]["cases"][index]["slots_exact"]
                                            - per_condition["zero_state"]["cases"][index]["slots_exact"]),
            "true_minus_wrong_state_slots": (per_condition["true"]["cases"][index]["slots_exact"]
                                             - per_condition["wrong_state"]["cases"][index]["slots_exact"]),
            "true_minus_zero_state_loss": round(
                per_condition["true"]["cases"][index]["token_loss"]
                - per_condition["zero_state"]["cases"][index]["token_loss"], 6),
            "true_minus_wrong_state_loss": round(
                per_condition["true"]["cases"][index]["token_loss"]
                - per_condition["wrong_state"]["cases"][index]["token_loss"], 6),
        }
        rows.append(row)
    write_jsonl(percase_path, rows)

    percase_digest = sha256_file(percase_path)
    conditions_block = {name: per_condition[name]["summary"] for name in CONDITIONS}
    result.update({
        "ok": True,
        "phase": "complete",
        "verdict": verdict,
        "qualified": verdict["qualified"],
        "thresholds": thresholds,
        "conditions": conditions_block,
        "contrasts": contrasts,
        "generation": {"conditions_generated": list(CONDITIONS), "cases_per_condition": len(items),
                       "max_new_tokens": max_tokens, "max_new_tokens_basis": max_tokens_basis,
                       "decoding": "greedy, deterministic, then scored by exact slot match",
                       "generation_batch_size": args.generation_batch_size},
        "cost": {"autoregressive_seconds": preprocess_seconds,
                 "per_case_seconds": round(preprocess_seconds / max(1, len(items)), 6)},
        "artifacts": {"json": str(out_dir / RESULT_NAME), "percase_jsonl": str(percase_path),
                      "percase_sha256": percase_digest, "log": str(log_path),
                      "test_feature_cache": str(test_cache_dir)},
        "limitations": [
            "Trained on 10016 RiSAWOZ-derived cases whose labels are template-generated "
            "from belief states; this measures that mapping, not general memory ability.",
            "Generation is single-sample greedy: no sampling, no pass@k.",
            "wrong_state is a cyclic roll within the same test file, so it is a weak "
            "corruption (same domain mix); beating it does not prove useful state reading.",
            "zero_state is the standardised origin = train-mean state, not literally "
            "'no state'; it mirrors the trainer's no_state ablation.",
            "Exact slot match counts the degenerate 无 slots; the nonempty-only figures "
            "are reported separately for that reason.",
            "Train mean/std were recovered from the training feature cache and verified by "
            "feature_hash; if that cache is deleted this checkpoint can no longer be "
            "finally evaluated here.",
            "No ablation MODEL was trained: zero/wrong are inference-time alterations of "
            "the same weights, so this is state sensitivity, not a trained baseline.",
        ],
    })
    result["exit_code"] = EXIT_OK_QUALIFIED if verdict["qualified"] else EXIT_NOT_QUALIFIED
    result.setdefault("memory", {})["peak_vram_mib"] = round(
        torch.cuda.max_memory_allocated(0) / 1024 ** 2, 1)
    write_json(out_dir / RESULT_NAME, result)
    log("verdict qualified=%s failed_criteria=%s"
        % (verdict["qualified"], json.dumps(verdict["criteria_failed"], ensure_ascii=False)))
    log("artifacts: %s | %s" % (out_dir / RESULT_NAME, percase_path))
    flush_log()
    return result


class Done(Exception):
    def __init__(self, result):
        self.result = result
        super().__init__("done")


def run(args, hooks=None):
    try:
        return evaluate_run(args, hooks)
    except Done as done:
        return done.result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="FINAL-ONLY evaluation of a trained RWKV-state memory decoder on a "
                    "case file training never opened. Read-only on the checkpoint; "
                    "GPU only; AC/lease fail-closed.")
    # These three are required for a real run, but `required=True` would also
    # reject `--self-test` (which needs no checkpoint, data or GPU), so they are
    # validated explicitly below instead.
    parser.add_argument("--checkpoint", default=None,
                        help="trainer checkpoint to score (opened read-only) [required]")
    parser.add_argument("--cases", default=None,
                        help="independent case file, e.g. memory_test.jsonl [required]")
    parser.add_argument("--out-dir", default=None,
                        help="where THIS evaluation writes its artifacts [required]")
    parser.add_argument("--model", default=T.DEFAULT_MODEL)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--train-cache-root", default=None,
                        help="training run directory that holds features/<key>/ "
                             "(default: the checkpoint's own directory)")
    parser.add_argument("--normalization", default=None,
                        help="exported %s to use (default: search the checkpoint dir, "
                             "then the out dir)" % NORMALIZATION_NAME)
    parser.add_argument("--train-cases", default=None,
                        help="the ORIGINAL training cases file, used only to prove that the "
                             "checkpoint's recorded training set is what you think it is "
                             "(its data_version and case order must match exactly)")
    parser.add_argument("--require-normalization-file", action="store_true",
                        help="refuse to fall back to training-cache recovery")
    parser.add_argument("--require-cache-cross-check", action="store_true",
                        help="accept statistics only if they could be re-derived from the "
                             "training cache's raw rows")
    parser.add_argument("--no-normalization-cross-check", action="store_true",
                        help="skip re-deriving a normalization file's statistics from the "
                             "cache even when the cache is available (faster, weaker)")
    parser.add_argument("--test-feature-cache", dest="test_feature_cache",
                        action="store_true", default=True)
    parser.add_argument("--no-test-feature-cache", dest="test_feature_cache",
                        action="store_false")
    parser.add_argument("--test-cache-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="cases per teacher-forced forward (logits are vocab-wide)")
    parser.add_argument("--generation-batch-size", type=int, default=16,
                        help="cases decoded together; 1 uses the per-case reference path")
    parser.add_argument("--eval-max-tokens", type=int, default=0,
                        help="0 = auto: longest reference + 8, so the cap cannot fake a miss")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = every test case")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--allow-ablated-checkpoint", action="store_true",
                        help="score a --state-mode none/shuffled checkpoint (diagnostics only)")
    parser.add_argument("--bootstrap-samples", type=int, default=2000,
                        help="0 disables the bootstrap (win/loss counts are still reported)")
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20261007)
    parser.add_argument("--threshold-exact-rate", type=float, default=0.80)
    parser.add_argument("--threshold-slots-delta", type=float, default=0.05)
    parser.add_argument("--threshold-ci-lower", type=float, default=0.0)
    parser.add_argument("--threshold-win-rate", type=float, default=0.60)
    parser.add_argument("--threshold-loss-delta", type=float, default=-0.05)
    parser.add_argument("--self-test", action="store_true",
                        help="CPU-only unit tests of the numeric core; no GPU, no data")
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [name for name in ("checkpoint", "cases", "out_dir")
                   if not getattr(args, name)]
        if missing:
            raise EvalError("required for a real evaluation: "
                            + ", ".join("--" + name.replace("_", "-") for name in missing))
    for name in ("eval_batch_size", "generation_batch_size", "test_cache_every", "chunk_tokens"):
        if getattr(args, name) < 1:
            raise EvalError("--%s must be positive" % name.replace("_", "-"))
    if args.max_cases < 0:
        raise EvalError("--max-cases cannot be negative")
    if not 0.5 <= args.bootstrap_confidence < 1.0:
        raise EvalError("--bootstrap-confidence must be in [0.5, 1)")
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.self_test:
            return self_test()
        result = run(args)
    except (EvalError, T.TrainingError) as exc:
        print("FAILED: %s" % exc, file=sys.stderr, flush=True)
        return EXIT_FAILED
    return int(result.get("exit_code", EXIT_FAILED))


# --------------------------------------------------------------------------
# CPU self test
# --------------------------------------------------------------------------
def _self_test_normalisation(tmp_root=None):
    """Exercise `resolve_normalisation` end to end on synthetic artifacts (CPU only).

    Every branch is covered, including the two that matter most: a file whose
    identity fields look right but whose numbers disagree with the raw rows, and a
    run where neither source exists (which must refuse rather than improvise).
    """
    import tempfile
    import torch

    results = []

    def check(name, condition, detail=""):
        results.append((name, bool(condition), detail))

    class Args:
        def __init__(self, checkpoint_path, out_dir, **kwargs):
            self.checkpoint = str(checkpoint_path)
            self.out_dir = str(out_dir)
            self.normalization = None
            self.train_cases = None
            self.train_cache_root = None
            self.require_normalization_file = False
            self.require_cache_cross_check = False
            self.no_normalization_cross_check = False
            for key, value in kwargs.items():
                setattr(self, key, value)

    def quiet(message):
        return None

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        raw = torch.randn(6, 2, 4, generator=torch.Generator().manual_seed(31))
        mean_ref = raw.mean(dim=(0, 1), keepdim=True)
        std_ref = raw.std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
        standardised = (raw - mean_ref) / std_ref
        feature_hash = hashlib.sha256(
            standardised.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:32]
        case_ids = ["c%d" % index for index in range(6)]
        backbone = {"sha256": "b" * 64}
        data_hash = "DH-normalisation-test"
        feature_shape = (2, 4)
        n_layer = feature_shape[0] // 3
        cache_identity, cache_key = feature_key_for(backbone, data_hash, n_layer)
        cache_dir = root / T.FEATURE_CACHE_DIR / cache_key[:16]
        writer = T.FeatureCache(cache_dir, cache_key, cache_identity, torch, every=100)
        for index, case_id in enumerate(case_ids):
            writer.add(case_id, raw[index], index + 1)
        writer.close()

        checkpoint_path = root / "ck.pt"
        checkpoint_path.write_bytes(b"checkpoint stand-in")
        checkpoint = {"identity_hash": "IH", "data_hash": data_hash,
                      "feature_hash": feature_hash, "train_count": 6,
                      "case_ids": list(case_ids), "data_counts": {"train": 6, "heldout": 0,
                                                                  "total": 6},
                      "path": str(checkpoint_path), "step": 1,
                      "feature_version": T.FEATURE_VERSION,
                      "tokenizer": {}}
        out_dir = root / "out"
        out_dir.mkdir()
        empty_root = root / "elsewhere"
        empty_root.mkdir()

        def build_payload(mean_value, std_value, digest_override=None):
            return {"format": NORMALIZATION_FORMAT, "tool": NORMALIZATION_TOOL,
                    "created_at": time.time(), "mean": mean_value.clone(),
                    "std": std_value.clone(),
                    "payload_digest": digest_override or normalization_payload_digest(
                        mean_value, std_value, torch),
                    "identity": {"identity_hash": "IH", "data_hash": data_hash,
                                 "feature_hash": feature_hash,
                                 "feature_version": T.FEATURE_VERSION,
                                 "feature_function": T.feature_function_fingerprint(),
                                 "feature_shape": list(feature_shape), "train_count": 6}}

        file_path = root / NORMALIZATION_NAME
        args = Args(checkpoint_path, out_dir)

        # 1. cache only
        mean, std, report = resolve_normalisation(args, checkpoint, torch, backbone,
                                                 feature_shape, quiet)
        check("cache-only resolution reports the verified recovery",
              report["source"] == "training-feature-cache-recovery"
              and report["recovery"]["verified"] is True, json.dumps(report["source"]))
        check("cache-only resolution returns the trainer's statistics",
              bool(torch.equal(mean, mean_ref) and torch.equal(std, std_ref)))

        # 2. file present + cache present -> cross-checked by default
        torch.save(build_payload(mean_ref, std_ref), file_path)
        mean, std, report = resolve_normalisation(args, checkpoint, torch, backbone,
                                                 feature_shape, quiet)
        check("a file is preferred and reports its source",
              report["source"] == "exported-normalization-file", report["source"])
        check("a file is cross-checked against the cache by default",
              report["cache_cross_check"] == "checked"
              and report["cache_cross_check_identical"] is True,
              json.dumps(report.get("cache_cross_check")))
        check("the cross-checked file gives the trainer's statistics",
              bool(torch.equal(mean, mean_ref) and torch.equal(std, std_ref)))

        # 3. an internally consistent but WRONG file must be refused
        torch.save(build_payload(mean_ref + 0.25, std_ref), file_path)
        try:
            resolve_normalisation(args, checkpoint, torch, backbone, feature_shape, quiet)
            check("a plausible file that disagrees with the raw rows is refused", False,
                  "accepted")
        except EvalError as exc:
            check("a plausible file that disagrees with the raw rows is refused",
                  "NOT bit-identical" in str(exc))
        torch.save(build_payload(mean_ref, std_ref), file_path)

        # 4. file present, cache gone
        gone = Args(checkpoint_path, out_dir, train_cache_root=str(empty_root))
        mean, std, report = resolve_normalisation(gone, checkpoint, torch, backbone,
                                                 feature_shape, quiet)
        check("a file is used when the cache is gone, and says so",
              report["cache_cross_check"] == "unavailable"
              and bool(torch.equal(mean, mean_ref)),
              json.dumps(report.get("cache_cross_check")))
        strict = Args(checkpoint_path, out_dir, train_cache_root=str(empty_root),
                      require_cache_cross_check=True)
        try:
            resolve_normalisation(strict, checkpoint, torch, backbone, feature_shape, quiet)
            check("require-cache-cross-check refuses a file without the cache", False,
                  "accepted")
        except EvalError as exc:
            check("require-cache-cross-check refuses a file without the cache",
                  "could not be re-derived" in str(exc))

        # 5. neither source
        file_path.unlink()
        try:
            resolve_normalisation(gone, checkpoint, torch, backbone, feature_shape, quiet)
            check("neither source refuses instead of improvising", False, "accepted")
        except EvalError as exc:
            check("neither source refuses instead of improvising",
                  "export_rwkv_normalization.py" in str(exc)
                  and "different function" in str(exc))
        missing_flag = Args(checkpoint_path, out_dir, train_cache_root=str(empty_root),
                            require_normalization_file=True)
        try:
            resolve_normalisation(missing_flag, checkpoint, torch, backbone, feature_shape, quiet)
            check("require-normalization-file refuses the cache fallback", False, "accepted")
        except EvalError as exc:
            check("require-normalization-file refuses the cache fallback",
                  "require-normalization-file" in str(exc))

        # 6. --train-cases: the positive check and the refusal
        cases_path = root / "cases.jsonl"
        cases_path.write_text(json.dumps({
            "id": "SELF-T01", "group": "heldout",
            "user": "帮我看下电机驱动板的固件版本",
            "assistant": "当前驱动板固件是 v1.2，我还没更新",
            "unverified": True,
            "summary": "主题：驱动板固件\n约束：无\n未完成：待确认版本\n结果：尚未更新",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        expected = DATA.all_cases(str(cases_path))
        matching = dict(checkpoint, data_hash=DATA.data_version(expected),
                        case_ids=[case["id"] for case in expected],
                        data_counts=DATA.split_counts(expected))
        cases_check = verify_training_cases(str(cases_path), matching)
        check("--train-cases accepts the genuine training file",
              cases_check["matches_checkpoint_data_hash"] is True
              and cases_check["case_ids_match_in_order"] is True
              and cases_check["matches_checkpoint_counts"] is True,
              json.dumps(cases_check, ensure_ascii=False))
        args_bad = Args(checkpoint_path, out_dir, train_cases=str(cases_path))
        try:
            resolve_normalisation(args_bad, checkpoint, torch, backbone, feature_shape, quiet)
            check("--train-cases refuses a file that is not the training set", False, "accepted")
        except EvalError as exc:
            check("--train-cases refuses a file that is not the training set",
                  "not the training set" in str(exc))
        check("the training cache was never written to during resolution",
              len(list(cache_dir.glob("shard-*"))) == 2)
    return results


def self_test():
    """CPU-only tests of everything that can be wrong without a GPU.

    A GPU run is needed to produce a real number; these tests exist so that the
    arithmetic, the cache reader, the batching and the verdict logic are already
    known-good before anyone spends a GPU hour on them.
    """
    import tempfile
    import torch

    checks = []

    def check(name, condition, detail=""):
        checks.append((name, bool(condition), detail))

    # ---- slot parsing agrees with the trainer's own parser -----------------
    sample = "主题：飞控固件烧写\n约束：无\n未完成：待查\n结果：尚未烧写成功"
    mine = parse_slots(sample)
    reference = getattr(DATA, "_tolerant_slots", None)
    if reference is not None:
        theirs = dict(reference(sample))
        check("parse_slots matches DATA._tolerant_slots", mine == theirs,
              "%r vs %r" % (mine, theirs))
    else:
        check("parse_slots mirrors DATA._tolerant_slots", len(mine) == 4,
              "trainer parser unavailable; structure check only")
    score = score_case(sample, sample)
    check("score_case: identical text scores 4/4",
          score["slots_exact"] == 4 and score["all_slots_exact"], str(score["per_slot"]))
    degenerate = score_case(sample, "主题：X\n约束：无\n未完成：无\n结果：Y")
    check("degenerate slots are excluded from the nonempty metric",
          degenerate["nonempty"]["constraint"]["scored"] is False
          and degenerate["nonempty"]["topic"]["scored"] is True, str(degenerate["nonempty"]))

    # ---- condition construction -------------------------------------------
    features = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    built = build_conditions(features, torch)
    check("zero_state is all zeros", bool(torch.count_nonzero(built["zero_state"]) == 0))
    check("wrong_state is the cyclic roll by one case",
          bool(torch.equal(built["wrong_state"][0], features[1])))
    check("true is untouched", bool(torch.equal(built["true"], features)))

    # ---- statistics recovery reproduces the trainer's formula -------------
    raw = torch.randn(6, 2, 4, generator=torch.Generator().manual_seed(7))
    mean_ref = raw.mean(dim=(0, 1), keepdim=True)
    std_ref = raw.std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
    standardised_ref = (raw - mean_ref) / std_ref
    digest_ref = hashlib.sha256(
        standardised_ref.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:32]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        identity, key = test_feature_key({"sha256": "deadbeef"}, "datahash", 2)
        write_cache = T.FeatureCache(root, key, identity, torch, every=100)
        for index in range(6):
            write_cache.add("c%d" % index, raw[index], index + 1)
        write_cache.close()
        reader = ReadOnlyFeatureCache(root, key, identity, torch)
        available = reader.load()
        check("read-only reader loads every case", len(available) == 6, str(available))
        recovered = torch.stack([reader.row("c%d" % index)[0] for index in range(6)], dim=0)
        check("read-only reader returns the stored rows", bool(torch.equal(recovered, raw)))
        mean, std, report = recover_train_statistics(reader, ["c%d" % i for i in range(6)], 6,
                                                    torch, digest_ref)
        check("recovered statistics verify against feature_hash", report["verified"],
              report["recomputed_feature_hash"])
        check("recovered mean equals the trainer's mean",
              bool(torch.allclose(mean, mean_ref, atol=0, rtol=0)))
        check("recovered std equals the trainer's std",
              bool(torch.allclose(std, std_ref, atol=0, rtol=0)))
        check("read-only reader never deleted the shards",
              len(list(root.glob("shard-*"))) == 2, str(sorted(p.name for p in root.glob("*"))))
        wrong_identity, _ = test_feature_key({"sha256": "cafebabe"}, "datahash", 2)
        mismatch = ReadOnlyFeatureCache(root, T.feature_cache_key(wrong_identity),
                                        wrong_identity, torch)
        try:
            mismatch.load()
            check("a foreign cache key is refused", False, "no error raised")
        except EvalError:
            check("a foreign cache key is refused", True)
        try:
            recover_train_statistics(reader, ["c0"], 1, torch, "0" * 32)
            check("a wrong feature_hash aborts recovery", False, "no error raised")
        except EvalError:
            check("a wrong feature_hash aborts recovery", True)
        try:
            recover_train_statistics(reader, ["c0", "missing"], 2, torch, digest_ref)
            check("a partial cache aborts recovery", False, "no error raised")
        except EvalError:
            check("a partial cache aborts recovery", True)

    # ---- batched generation equals the per-case reference ------------------
    tiny = T.DecoderConfig(feature_count=6, feature_dim=4, vocab_size=260, d_model=32,
                           n_heads=2, n_self_layers=1, n_cross_layers=1, dropout=0.0,
                           conditioning_tokens=2)
    module = T.build_decoder(tiny, torch)
    module.eval()
    tokenizer = T.ByteTokenizer()
    ids = T.special_ids(tokenizer)
    probe = torch.randn(5, 6, 4, generator=torch.Generator().manual_seed(11))
    noop = lambda stage: None
    per_case = generate_per_case(module, probe, tokenizer, ids, 12, torch,
                                 torch.device("cpu"), noop)
    batched = generate_batch(module, probe, tokenizer, ids, 12, torch,
                             torch.device("cpu"), 3, noop)
    check("batched greedy generation equals the per-case reference", per_case == batched,
          "%r vs %r" % (per_case, batched))

    # ---- teacher-forced per-case loss matches a direct computation ---------
    cases = [{"id": "a", "group": "heldout", "user": "u1", "assistant": "a1",
              "summary": "主题：甲\n约束：无\n未完成：无\n结果：尚未完成", "unverified": True},
             {"id": "b", "group": "heldout", "user": "u2", "assistant": "a2",
              "summary": "主题：乙\n约束：无\n未完成：无\n结果：尚未完成", "unverified": True}]
    features2 = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(3))
    block = teacher_forced_per_case(module, cases, features2, tokenizer, ids, torch,
                                    torch.device("cpu"), 1, noop)
    one_shot = teacher_forced_per_case(module, cases, features2, tokenizer, ids, torch,
                                       torch.device("cpu"), 2, noop)
    check("per-case losses are identical across batch sizes",
          block["per_case_sum"] == one_shot["per_case_sum"]
          and block["per_case_tokens"] == one_shot["per_case_tokens"])
    check("aggregate loss equals the token-weighted mean of the per-case parts",
          abs(block["token_loss"]
              - sum(block["per_case_sum"]) / sum(block["per_case_tokens"])) < 1e-12)

    # ---- metrics and verdict logic ----------------------------------------
    rate = wilson_interval(80, 100)
    check("wilson interval brackets the observed rate", rate[0] < 0.8 < rate[1], str(rate))
    check("wilson interval refuses an empty set", wilson_interval(0, 0) is None)
    values = [1.0] * 30 + [-1.0] * 10
    first = bootstrap_ci(values, 200, 0.95, 5)
    again = bootstrap_ci(values, 200, 0.95, 5)
    check("bootstrap is deterministic for a fixed seed", first == again, json.dumps(first))
    check("bootstrap mean equals the sample mean",
          abs(first["mean"] - statistics.fmean(values)) < 1e-12)
    check("bootstrap is skipped when samples = 0", bootstrap_ci(values, 0, 0.95, 5) is None)

    condition_block = {"true": {"four_slot_exact_rate": 0.81}}
    contrast_pass = [{"comparison": "true vs zero_state",
                      "slots_exact": {"mean_delta": 0.4, "win_rate_decided_cases": 0.8,
                                      "bootstrap": {"ci_low": 0.2}},
                      "per_case_token_loss": {"mean_delta_true_minus_other": -0.3}}]
    contrast_fail = [{"comparison": "true vs zero_state",
                      "slots_exact": {"mean_delta": 0.4, "win_rate_decided_cases": 0.8,
                                      "bootstrap": {"ci_low": 0.2}},
                      "per_case_token_loss": {"mean_delta_true_minus_other": -0.3}},
                     {"comparison": "true vs wrong_state",
                      "slots_exact": {"mean_delta": 0.0, "win_rate_decided_cases": 0.5,
                                      "bootstrap": {"ci_low": 0.0}},
                      "per_case_token_loss": {"mean_delta_true_minus_other": 0.01}}]
    thresholds = {"exact_rate": 0.80, "slots_delta": 0.05, "ci_lower": 0.0,
                  "win_rate": 0.60, "loss_delta": -0.05}
    check("a passing run qualifies", judge(condition_block, contrast_pass, thresholds)["qualified"])
    verdict = judge(condition_block, contrast_fail, thresholds)
    check("one failing contrast blocks qualification", not verdict["qualified"], )
    check("failing criteria are named", len(verdict["criteria_failed"]) >= 3,
          json.dumps(verdict["criteria_failed"], ensure_ascii=False))
    loss_only = [{"comparison": "true vs wrong_state",
                  "slots_exact": {"mean_delta": 0.0, "win_rate_decided_cases": 0.5,
                                  "bootstrap": {"ci_low": -0.01}},
                  "per_case_token_loss": {"mean_delta_true_minus_other": -0.2}}]
    check("a better loss alone does NOT qualify",
          not judge(condition_block, loss_only, thresholds)["qualified"])
    check("every criterion reports pass/fail explicitly",
          all("passed" in entry and "observed" in entry
              for entry in judge(condition_block, contrast_pass, thresholds)["criteria"]))

    # ---- token cap ---------------------------------------------------------
    cap, basis = resolve_max_tokens(0, cases, tokenizer)
    check("auto token cap clears the longest reference", cap > 8, "%d (%s)" % (cap, basis))
    cap, basis = resolve_max_tokens(5, cases, tokenizer)
    check("an explicit token cap is respected", cap == 5 and basis == "explicit")

    # ---- normalisation contract (both paths and every refusal) -------------
    checks.extend(_self_test_normalisation(tmp_root=None))

    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print("%s %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  <- " + detail) if (detail and not ok) else ""), flush=True)
    print("\n%d/%d self-test checks passed" % (passed, len(checks)), flush=True)
    return EXIT_OK_QUALIFIED if passed == len(checks) else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
