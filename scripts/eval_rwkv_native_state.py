"""STANDALONE NATIVE-STATE BASELINE for the four-slot memory task -- NO TRAINING.

WHAT THIS MEASURES
------------------
The trained pilot (train_rwkv_slots.py) asks whether a *trained* decoder can read
the frozen RWKV-7 recurrent state and write four-slot memory text. This script
measures the floor under that number: can the frozen RWKV-7 model itself, with no
training, no adapter and no changed weights, continue generating the four slots
straight out of its own native recurrent state?

Per case, exactly three things happen:
  1. observe ONLY `case["user"]` through the frozen backbone -> native state S.
     The target assistant turn is never observed, so no answer leaks into S.
  2. continue from S with a Chinese prompt that asks for exactly four slots.
  3. greedy-decode, then parse strictly and score against the case's memory text.

The model is allowed to use its own pretrained language knowledge. Nothing is
trained, no optimizer exists in this file, and every weight stays frozen.

WHY THE ENGINE CAN DO THIS
--------------------------
The installed `rwkv` package exposes the exact primitive required: `RWKV.forward(
tokens, state)` consumes `state` and returns `(logits_for_the_last_token,
next_state)` (site-packages/rwkv/model.py:1443, and with `seq_mode and not
full_output` it slices `x[-1, :]` before the head at line 1665). `nb_rwkv.TorchEngine`
already generates from an observed state this way in `TorchEngine.summarize`, so
"continue from an arbitrary observed state" is a supported operation of this
engine, not a workaround. `TorchEngine.observe` does the state-producing half.

SCOPE / HONESTY RULES
---------------------
* Evaluation set: `memory_test.jsonl` (RiSAWOZ test split) ONLY. That file is the
  final-only split; this script never trains and never selects on anything.
* The four reported quantities are kept separate on purpose: strict four-slot
  parse rate, tolerant per-slot agreement, strict per-slot value accuracy, and
  the constant baselines. A high tolerant score can come from echoing `无`, so a
  constant baseline is computed alongside -- do not quote one without the other.
* A failure here does NOT prove "the state has no content"; it bounds what the
  frozen model will say without any adapter.

SAFETY (fail-closed, same contract as the trainer)
--------------------------------------------------
* AC + guard-lease gate before anything touches CUDA, again every
  `--gate-every` cases, and once before the final metrics.
* Unavailable power status, a missing lease or an expired lease is a PAUSE, not a
  warning: partial metrics are written and the process exits 75.
* `--skip-power-check` exists for plumbing tests only and is recorded verbatim in
  the metrics, so a run that used it can never be mistaken for a gated run.

RESUMABLE OUTPUT
----------------
`outputs.jsonl` is appended one fsynced line per case and re-read at startup, so
an AC pause costs the remaining cases, not the whole split. `metrics.json` is
rewritten (atomically) as the run progresses and on pause.

Run with the CUDA venv:
  D:\\AI\\DBA\\.venv-rwkv\\Scripts\\python.exe eval_rwkv_native_state.py \
      --cases D:\\AI\\DBA\\training\\web-sources\\risawoz\\memory_test.jsonl \
      --out-dir D:\\AI\\DBA\\training\\rwkv-native-baseline

Offline check of the parser, stop policy, metrics, resume and pause paths
(no model, no CUDA):
  ... eval_rwkv_native_state.py --self-test
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
POWER_SCRIPTS = Path(r"D:\AI\双脑Agent\scripts")
if str(POWER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(POWER_SCRIPTS))

import rwkv_slots_data as DATA

try:
    from training_power_control import training_allowed
except Exception:  # pragma: no cover - only where the guard is not installed
    def training_allowed():
        return False

SCRIPT_VERSION = "eval_rwkv_native_state.py/1"
FORMAT = 1
LABEL = "native-state baseline (frozen RWKV-7, no training, no adapter)"
BACKBONE_TRAINING = False
EXIT_PAUSED = 75
DEFAULT_MODEL = r"D:\AI\models\rwkv7\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"
DEFAULT_CASES = r"D:\AI\DBA\training\web-sources\risawoz\memory_test.jsonl"
DEFAULT_OUT = r"D:\AI\DBA\training\rwkv-native-baseline"
OUTPUT_NAME = "outputs.jsonl"
METRICS_NAME = "metrics.json"
LOG_NAME = "eval.log"
LABELS = tuple(DATA.SUMMARY_LABEL_ORDER)          # 主题 / 约束 / 未完成 / 结果
EMPTY_SLOT = DATA.EMPTY_SLOT                      # 无
TABLE_MARGIN = 3                                  # BOS/EOS/PAD sit above real ids

# The prompt is part of the measurement: it is hashed into the metrics so a
# baseline run can never be compared against a different question. Two built-in
# styles exist because the first real run showed the frozen model echoing the
# placeholder characters of style "placeholder" instead of filling them in;
# "labelled" names the four labels without giving it a skeleton to copy.
# `--prompt` still overrides both, and the metrics record whichever was used.
PROMPT = (
    "\nUser: 只根据上面观察到的对话，用中文输出四行记忆，每行一个槽位，顺序固定："
    "主题：… / 约束：… / 未完成：… / 结果：…。"
    "每个槽位必须非空，没有内容就写“无”。只输出这四行，不要编号，不要解释，不要多余文字。"
    "\nAssistant:"
)
PROMPT_LABELLED = (
    "\nUser: 请只输出四行中文记忆，顺序固定为 主题、约束、未完成、结果。"
    "每行以该标签加中文冒号开头，冒号后写具体内容；没有内容的槽位写“无”。"
    "不要编号，不要解释，不要输出这四行以外的任何文字。"
    "\nAssistant:"
)
PROMPTS = {"placeholder": PROMPT, "labelled": PROMPT_LABELLED}
OBSERVATION_MODE = "user-field-only"


class BaselineError(RuntimeError):
    """Refusal or configuration error. Never used for an AC pause."""


class Paused(Exception):
    """No guard lease or no AC power: stop, keep partial results, exit 75."""

    def __init__(self, report):
        super().__init__("%s: guard_lease=%s ac_connected=%s"
                         % (report["stage"], report["guard_lease"], report["ac_connected"]))
        self.report = report


# --------------------------------------------------------------------------
# power / fail-closed
# --------------------------------------------------------------------------
class _PowerStatus(ctypes.Structure):
    # Explicit widths so the struct is 12 bytes on any platform, matching Windows.
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]


def main_connected():
    """Direct AC state. Unavailable power status NEVER authorises a GPU run."""
    if os.name != "nt":
        return False
    status = _PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return False
    return int(status.ACLineStatus) == 1


def power_report(stage, require_lease=True):
    lease = bool(training_allowed())
    ac = bool(main_connected())
    report = {"stage": stage, "guard_lease": lease, "ac_connected": ac}
    if require_lease and not lease:
        raise Paused({**report, "reason": "no valid guard lease"})
    if not ac:
        raise Paused({**report, "reason": "AC power not connected"})
    return report


# --------------------------------------------------------------------------
# strict four-slot parsing and the generation stop policy
# --------------------------------------------------------------------------
def _label_of(line):
    """(slot label, value) when the line opens with one of the four labels."""
    for label in LABELS:
        for colon in ("：", ":"):
            if line.startswith(label + colon):
                return label, line[len(label) + 1:].strip()
    return None, None


def parse_four_slots(text):
    """STRICT parse: exactly the four labelled lines, in order, all non-empty.

    Strictness is the point. A tolerant parser happily returns the two `无` slots
    that a constant answer would also return, which is why the tolerant number is
    reported separately and never alone.
    """
    lines = [line.strip() for line in (text or "").replace("\r", "\n").split("\n")]
    slots, order, problems = {}, [], []
    for line in lines:
        if not line:
            continue
        label, value = _label_of(line)
        if label is None:
            problems.append("unlabelled-line")
            continue
        if label in slots:
            problems.append("duplicate-" + label)
            continue
        if not value:
            problems.append("empty-value-" + label)
            continue
        slots[label] = value
        order.append(label)
    if order != list(LABELS):
        problems.append("label-order-or-missing:" + ",".join(order))
    return {"slots": slots, "order": order, "problems": problems,
            "ok": not problems and len(order) == len(LABELS)}


def four_slot_stop(text):
    """Generation stop policy: the four-line block is finished.

    Stop as soon as the fourth labelled line has been closed by a newline, or as
    soon as any further text follows a complete four-line block. This is about
    *when to stop*, not about correctness: a fourth line with an empty value still
    stops the loop and is then reported as a strict-parse failure.
    """
    lines = (text or "").replace("\r", "\n").split("\n")
    seen = []
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        label, _ = _label_of(line)
        if label is None:
            if len(seen) >= len(LABELS):
                return "text-after-four-slots"
            continue
        if label not in seen:
            seen.append(label)
        if len(seen) == len(LABELS) and index < len(lines) - 1:
            return "four-slots"
    return None


def observation_text(case):
    """The ONLY text the backbone observes: `case["user"]`, verbatim.

    No assistant target, no summary, no template wrapping. The distinction
    matters: `case["user"]` already contains the dialogue history, and feeding
    `case["assistant"]` would hand the model the answer it is being scored on.
    """
    return case["user"]


# --------------------------------------------------------------------------
# engine adapter: continue generation from an arbitrary observed state
# --------------------------------------------------------------------------
class NativeContinuation:
    """Generation from any observed state, over the engine's public forward.

    `TorchEngine.observe(state, text)` -> (state, tokens) is the producer;
    `engine.model.forward(tokens, state)` -> (logits, state) is the consumer, and
    is the same call `TorchEngine.summarize` makes. Nothing here trains.
    """

    def __init__(self, engine):
        self.engine = engine
        self.torch = engine.torch
        self.pipeline = engine.pipeline
        table = getattr(getattr(engine.pipeline, "tokenizer", None), "idx2token", None)
        if not isinstance(table, dict) or not table:
            raise BaselineError("RWKV pipeline exposes no token table; cannot bound real token ids")
        self.special_first = max(int(key) for key in table.keys()) + 1
        self.vocab_size = self.special_first + TABLE_MARGIN

    def encode(self, text):
        tokens = [int(token) for token in self.pipeline.encode(text)]
        if not tokens:
            raise BaselineError("tokenizer produced no tokens")
        return tokens

    def decode(self, tokens):
        return self.pipeline.decode([int(token) for token in tokens])

    def _inference(self):
        mode = getattr(self.torch, "inference_mode", None)
        return mode() if callable(mode) else nullcontext()

    def observe_user(self, text):
        """Frozen backbone pass over the user text; returns (state, token count)."""
        return self.engine.observe(None, text)

    def generate(self, state, prompt, max_new_tokens):
        engine = self.engine
        torch = self.torch
        prompt_tokens = self.encode(prompt)
        generated = []
        stop_reason = "max-tokens"
        with self._inference():
            logits = None
            for start in range(0, len(prompt_tokens), engine.chunk_tokens):
                logits, state = engine.model.forward(
                    prompt_tokens[start:start + engine.chunk_tokens], state)
                engine.assert_gpu(state)
            if logits is None:
                raise BaselineError("prompt produced no logits")
            for _ in range(max_new_tokens):
                # One sync per token here; the full 72-tensor state assertion is
                # done per case instead, because per token it would dominate the
                # run without adding information.
                if getattr(logits.device, "type", None) != "cuda" or not torch.isfinite(logits).all().item():
                    raise BaselineError("invalid GPU generation logits")
                token = int(torch.argmax(logits).item())
                if token == DATA.RWKV_EOT_ID:
                    stop_reason = "eot"
                    break
                if token >= self.special_first:
                    stop_reason = "special-token-id"
                    break
                generated.append(token)
                reason = four_slot_stop(self.decode(generated))
                if reason:
                    stop_reason = reason
                    break
                logits, state = engine.model.forward([token], state)
                engine.assert_gpu(state)
        text = self.decode(generated)
        return {"text": text, "tokens": generated, "stop_reason": stop_reason,
                "prompt_tokens": len(prompt_tokens), "state": state}


# --------------------------------------------------------------------------
# atomic small-file io
# --------------------------------------------------------------------------
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path):
    """Tolerant reader: a paused run can leave a truncated final line behind."""
    path = Path(path)
    if not path.is_file():
        return [], 0
    records, broken = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            broken += 1
    return records, broken


def sha256_file(path, block=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------
def load_cases(path, cap=0):
    """Load and contract-validate the evaluation file, sorted like the loader.

    `DATA.all_cases` is deliberately NOT used: it merges the six embedded heldout
    cases, and this baseline must report on the RiSAWOZ test split only.
    """
    cases_path = Path(path)
    if not cases_path.is_file():
        raise BaselineError("cases file missing: " + str(cases_path))
    items = DATA.validate(DATA.load_events(str(cases_path)))
    if not items:
        raise BaselineError("cases file holds no cases: " + str(cases_path))
    items = sorted(items, key=lambda case: (0 if case["group"] == DATA.TRAIN else 1, case["id"]))
    if cap and cap > 0:
        items = items[:cap]
    return items


def expected_slots(case):
    parsed = parse_four_slots(DATA.memory_text(case))
    if not parsed["ok"]:
        raise BaselineError("reference memory text is not four slots: " + case["id"])
    return parsed


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _rate(hits, total):
    return None if not total else round(hits / float(total), 6)


def _mean(values):
    return None if not values else round(sum(values) / float(len(values)), 6)


def compute_metrics(records, context):
    """Aggregate per-case records. Every denominator is published."""
    total = len(records)
    per_slot_keys = tuple(DATA.SLOT_KEYS)
    tolerant_slots = {key: 0 for key in per_slot_keys}
    strict_values = {key: 0 for key in per_slot_keys}
    content_values = {key: 0 for key in per_slot_keys}
    content_filled = {key: 0 for key in per_slot_keys}
    strict_ok = 0
    labels_present = 0
    tolerant_exact = 0
    tolerant_case_exact = 0
    exact_message = 0
    all_empty = 0
    skipped = 0
    stop_reasons = {}
    generated_lengths, observation_lengths, durations = [], [], []
    per_slot_hits = []

    for record in records:
        if record.get("skipped"):
            skipped += 1
            continue
        stop_reasons[record["stop_reason"]] = stop_reasons.get(record["stop_reason"], 0) + 1
        generated_lengths.append(int(record["generated_tokens"]))
        observation_lengths.append(int(record["observation_tokens"]))
        durations.append(float(record["seconds"]))
        strict = record["strict"]
        tolerant = record["tolerant"]
        reference = record["reference"]["slots"]
        if strict["ok"]:
            strict_ok += 1
            if all(strict["slots"].get(label) == EMPTY_SLOT for label in LABELS):
                all_empty += 1
        if all(label in record["labels_seen"] for label in LABELS):
            labels_present += 1
        per_slot_hits.append(int(tolerant["slots_exact"]))
        tolerant_exact += int(tolerant["slots_exact"])
        if tolerant["all_slots_exact"]:
            tolerant_case_exact += 1
        if record["generated"] == record["reference"]["text"]:
            exact_message += 1
        for key, label in zip(DATA.SLOT_KEYS, LABELS):
            if tolerant["per_slot"][key]:
                tolerant_slots[key] += 1
            if strict["ok"] and strict["slots"].get(label) == reference.get(label):
                strict_values[key] += 1
                if reference.get(label) != EMPTY_SLOT:
                    content_values[key] += 1
            if reference.get(label) != EMPTY_SLOT:
                content_filled[key] += 1

    scored = total - skipped
    return {
        "format": FORMAT,
        "script": SCRIPT_VERSION,
        "label": LABEL,
        "backbone_training": BACKBONE_TRAINING,
        "trained_decoder_used": False,
        "counts": {
            "cases_selected": context["cases_selected"],
            "cases_evaluated": scored,
            "cases_skipped": skipped,
            "cases_resumed_from_previous_output": context["cases_resumed"],
            "broken_output_lines": context["broken_output_lines"],
            "cases_in_file": context["cases_in_file"],
            "cap": context["cap"],
        },
        "rates": {
            "strict_four_slot_rate": _rate(strict_ok, scored),
            "all_four_labels_seen_rate": _rate(labels_present, scored),
            "strict_all_empty_answer_rate": _rate(all_empty, scored),
            "tolerant_slots_exact_mean_of_4": _mean(per_slot_hits),
            "tolerant_all_slots_exact_rate": _rate(tolerant_case_exact, scored),
            "strict_message_exact_rate": _rate(exact_message, scored),
            "strict_slot_value_accuracy": _rate(sum(strict_values.values()), scored * len(LABELS)),
            "content_slot_value_accuracy": _rate(sum(content_values.values()),
                                                 sum(content_filled.values())),
        },
        "per_slot": {
            "tolerant_hits": tolerant_slots,
            "strict_hits_when_parsed": strict_values,
            "content_hits": content_values,
            "content_expected": content_filled,
        },
        "stop_reasons": stop_reasons,
        "timing": {
            "seconds_total": round(sum(durations), 3),
            "seconds_per_case_mean": _mean(durations),
            "generated_tokens_mean": _mean(generated_lengths),
            "generated_tokens_max": max(generated_lengths) if generated_lengths else None,
            "observation_tokens_mean": _mean(observation_lengths),
            "observation_tokens_max": max(observation_lengths) if observation_lengths else None,
        },
        "baselines": {
            "note": ("Constant answers, computed on the same selected cases. A tolerant "
                     "score is only interpretable next to these."),
            "all_empty_slots": context["baseline_all_empty"],
            "most_common_reference_message": context["baseline_constant"],
        },
        "model": context["model"],
        "observation": context["observation"],
        "cases_file": context["cases_file"],
        "power": context["power"],
        "artifacts": context["artifacts"],
    }


def constant_baselines(cases):
    """What a trained-nothing constant answer already scores."""
    references = [expected_slots(case) for case in cases]
    empty_message = "\n".join("%s：%s" % (label, EMPTY_SLOT) for label in LABELS)
    counts = {}
    for case in cases:
        text = DATA.memory_text(case)
        counts[text] = counts.get(text, 0) + 1
    best = max(counts.items(), key=lambda item: item[1])[0] if counts else empty_message

    def score(message):
        hits, cases_exact = [], 0
        for case in cases:
            tolerant = DATA.slot_match(message, DATA.memory_text(case))
            hits.append(int(tolerant["slots_exact"]))
            if tolerant["all_slots_exact"]:
                cases_exact += 1
        return {"message": message, "tolerant_slots_exact_mean_of_4": _mean(hits),
                "tolerant_all_slots_exact_rate": _rate(cases_exact, len(cases)),
                "strict_four_slot_parse": parse_four_slots(message)["ok"]}

    return {
        "all_empty_slots": score(empty_message),
        "most_common_reference_message": {**score(best), "reference_count": counts.get(best, 0)},
    }


# --------------------------------------------------------------------------
# one case
# --------------------------------------------------------------------------
def evaluate_case(engine, continuation, case, args):
    """Observe the user text, continue from that state, score the result."""
    started = time.perf_counter()
    observation = observation_text(case)
    reference = expected_slots(case)
    record = {"id": case["id"], "group": case["group"],
              "observation_mode": OBSERVATION_MODE,
              "observation_chars": len(observation),
              "assistant_in_observation": case["assistant"] in observation,
              "summary_in_observation": DATA.memory_text(case) in observation,
              "reference": {"text": DATA.memory_text(case), "slots": reference["slots"]}}
    observation_tokens = len(continuation.encode(observation))
    record["observation_tokens"] = observation_tokens
    record["prompt_tokens"] = len(continuation.encode(args.prompt))
    if observation_tokens > args.max_observation_tokens:
        # Refuse rather than silently truncate: a truncated history would change
        # the state under test and could quietly drop the constraint's source turn.
        record.update({"skipped": "observation-too-long", "stop_reason": "skipped",
                       "generated": "", "generated_tokens": 0,
                       "strict": {"ok": False, "slots": {}, "order": [], "problems": ["skipped"]},
                       "labels_seen": [],
                       "tolerant": {"per_slot": {key: 0 for key in DATA.SLOT_KEYS},
                                    "slots_exact": 0, "slots_total": len(LABELS),
                                    "all_slots_exact": False},
                       "seconds": round(time.perf_counter() - started, 4)})
        return record

    state, observed_tokens = continuation.observe_user(observation)
    engine.assert_gpu(state)
    rollout = continuation.generate(state, args.prompt, args.max_new_tokens)
    engine.assert_gpu(rollout["state"])

    generated = rollout["text"]
    strict = parse_four_slots(generated)
    tolerant = DATA.slot_match(generated, DATA.memory_text(case))
    record.update({
        "observed_tokens_reported": int(observed_tokens),
        "generated": generated,
        "generated_tokens": len(rollout["tokens"]),
        "stop_reason": rollout["stop_reason"],
        "utf8_incomplete": "\ufffd" in generated,
        "strict": strict,
        "labels_seen": [label for label in LABELS
                        if any(label + colon in generated for colon in ("：", ":"))],
        "tolerant": tolerant,
    })
    record["seconds"] = round(time.perf_counter() - started, 4)
    return record


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def make_gate(args, hooks):
    override = hooks.get("gate")
    disabled = {"stage": None, "guard_lease": None, "ac_connected": None,
                "power_gate": "disabled-by---skip-power-check"}

    def gate(stage):
        if override is not None:
            return override(stage)
        if args.skip_power_check:
            return {**disabled, "stage": stage}
        return power_report(stage, require_lease=True)

    return gate


def run(args, hooks=None):
    hooks = hooks or {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / OUTPUT_NAME
    metrics_path = out_dir / METRICS_NAME
    log_path = out_dir / LOG_NAME
    log_lines = []

    def log(message):
        line = "%s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message)
        log_lines.append(line)
        print(line, flush=True)

    def flush_log():
        if not log_lines:
            return
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(log_lines) + "\n")
        log_lines.clear()

    gate = make_gate(args, hooks)
    power = {"before_cuda": None, "during_evaluation": []}

    context = {
        "cases_selected": 0, "cases_resumed": 0, "broken_output_lines": 0,
        "cases_in_file": 0, "cap": args.cap,
        "baseline_all_empty": None, "baseline_constant": None,
        "model": {}, "observation": {
            "mode": OBSERVATION_MODE,
            "assistant_target_observed": False,
            "summary_observed": False,
            "prompt": args.prompt,
            "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
            "max_new_tokens": args.max_new_tokens,
        },
        "cases_file": {"path": str(Path(args.cases).resolve())},
        "power": power,
        "artifacts": {"outputs": str(output_path), "metrics": str(metrics_path),
                      "log": str(log_path)},
    }
    result = {"ok": False, "label": LABEL, "trained_decoder_used": False,
              "script": SCRIPT_VERSION, "phase": "started", "metrics": str(metrics_path),
              "outputs": str(output_path)}

    def finish(phase, ok, exit_code):
        result.update({"phase": phase, "ok": ok, "exit_code": exit_code,
                       "counts": context.get("counts"), "rates": context.get("rates")})
        flush_log()
        return result

    # ---- gate 1: before the model file is even opened ----------------------
    try:
        power["before_cuda"] = gate("before-cuda")
    except Paused as exc:
        context["power"] = power
        write_json(metrics_path, {"format": FORMAT, "script": SCRIPT_VERSION, "label": LABEL,
                                  "phase": "paused", "power": power, "paused_before": "before-cuda",
                                  "error": str(exc), "artifacts": context["artifacts"]})
        log("power gate stopped the run before CUDA: " + json.dumps(exc.report, ensure_ascii=False))
        result.update({"phase": "paused", "ok": False, "exit_code": EXIT_PAUSED})
        return finish("paused", False, EXIT_PAUSED)
    log("power gate before CUDA: " + json.dumps(power["before_cuda"], ensure_ascii=False))

    # ---- data and engine --------------------------------------------------
    cases = load_cases(args.cases, args.cap)
    context["cases_selected"] = len(cases)
    context["cases_in_file"] = len(DATA.validate(DATA.load_events(str(Path(args.cases)))))
    context["cases_file"].update({
        "sha256": sha256_file(args.cases),
        "data_version": DATA.data_version(cases),
        "cases_in_file": context["cases_in_file"],
        "group_counts": DATA.split_counts(cases),
    })
    log("cases selected %d of %d data_version=%s"
        % (len(cases), context["cases_in_file"], context["cases_file"]["data_version"][:16]))

    baselines = constant_baselines(cases)
    context["baseline_all_empty"] = baselines["all_empty_slots"]
    context["baseline_constant"] = baselines["most_common_reference_message"]
    context["observation"]["user_chars_mean"] = _mean([len(case["user"]) for case in cases])

    engine = hooks.get("engine")
    if engine is None:
        import torch  # noqa: F401 - imported here so --self-test never needs CUDA
        if not torch.cuda.is_available():
            raise BaselineError("CUDA is required; the native baseline refuses CPU inference")
        if not torch.cuda.is_bf16_supported():
            raise BaselineError("CUDA BF16 is required")
        from nb_rwkv import TorchEngine
        engine = TorchEngine(args.model, chunk_tokens=args.chunk_tokens,
                             generation_tokens=args.max_new_tokens)
    continuation = NativeContinuation(engine)
    context["model"] = {
        "path": str(getattr(engine, "path", args.model)),
        "identity": getattr(engine, "identity", None),
        "special_ids_floor": continuation.special_first,
        "vocab_with_specials": continuation.vocab_size,
        "generation": "greedy argmax, stop on EOT id 0, special id, four-slot block or cap",
    }
    log("engine ready special_first=%d vocab=%d" % (continuation.special_first,
                                                    continuation.vocab_size))

    # ---- resume -----------------------------------------------------------
    records, broken = read_jsonl(output_path)
    done = {record.get("id") for record in records if record.get("id")}
    context["broken_output_lines"] = broken
    context["cases_resumed"] = len(records)
    if records:
        log("resume: %d case records already on disk (%d unreadable lines); skipping them"
            % (len(records), broken))

    def write_metrics(phase):
        block = compute_metrics(records, context)
        block["phase"] = phase
        write_json(metrics_path, block)
        context["rates"] = block["rates"]
        context["counts"] = block["counts"]
        return block

    paused_at = None
    try:
        for index, case in enumerate(cases):
            if case["id"] in done:
                continue
            if index and index % args.gate_every == 0:
                report = gate("during-evaluation")
                power["during_evaluation"].append(report)
            record = evaluate_case(engine, continuation, case, args)
            append_jsonl(output_path, record)
            records.append(record)
            if (index + 1) % args.log_every == 0 or (index + 1) == len(cases):
                write_metrics("running")
                log("evaluated %d/%d (%d resumed) last=%s strict=%s tolerant=%s/%s stop=%s"
                    % (len(records), len(cases), context["cases_resumed"], record["id"],
                       record["strict"]["ok"], record["tolerant"]["slots_exact"],
                       record["tolerant"].get("slots_total", len(LABELS)),
                       record["stop_reason"]))
    except Paused as exc:
        paused_at = exc.report
        power["during_evaluation"].append(exc.report)
        log("power gate stopped the run: " + json.dumps(exc.report, ensure_ascii=False))

    block = write_metrics("paused" if paused_at else "complete")
    if paused_at:
        result.update({"paused": True, "paused_report": paused_at, "counts": block["counts"],
                       "rates": block["rates"]})
        return finish("paused", False, EXIT_PAUSED)

    log("done: strict_four_slot_rate=%s tolerant_slots_exact_mean=%s tolerant_all_slots_exact_rate=%s"
        % (block["rates"]["strict_four_slot_rate"], block["rates"]["tolerant_slots_exact_mean_of_4"],
           block["rates"]["tolerant_all_slots_exact_rate"]))
    log("baselines: all-empty=%s most-common=%s"
        % (block["baselines"]["all_empty_slots"]["tolerant_slots_exact_mean_of_4"],
           block["baselines"]["most_common_reference_message"]["tolerant_slots_exact_mean_of_4"]))
    result.update({"counts": block["counts"], "rates": block["rates"]})
    return finish("complete", True, 0)


# --------------------------------------------------------------------------
# offline self-test (no model, no CUDA)
# --------------------------------------------------------------------------
class _FakeDevice:
    type = "cuda"
    index = 0


class _FakeValue:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class _FakeLogits:
    def __init__(self, next_id):
        self.next_id = next_id
        self.device = _FakeDevice()


class FakeTorch:
    class cuda:  # noqa: N801 - mirrors torch.cuda
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported():
            return True

    @staticmethod
    def isfinite(value):
        return _FakeValue(True)

    @staticmethod
    def argmax(value):
        return _FakeValue(value.next_id)


class FakePipeline:
    """One token per character, offset by 1 so id 0 stays the EOT sentinel."""

    def __init__(self):
        self.tokenizer = type("Tok", (), {"idx2token": {i: "x" for i in range(70000)}})()

    def encode(self, text):
        return [ord(char) + 1 for char in text]

    def decode(self, tokens):
        return "".join(chr(int(token) - 1) for token in tokens)


class FakeModel:
    """Models `RWKV.forward(tokens, state)`: returns the NEXT planned token."""

    def __init__(self):
        self.calls = []

    def forward(self, tokens, state):
        plan = list(state["plan"])
        position = int(state["pos"])
        if len(tokens) == 1:
            position += 1
        self.calls.append((len(tokens), position))
        next_id = plan[position] if position < len(plan) else DATA.RWKV_EOT_ID
        return _FakeLogits(next_id), {"plan": plan, "pos": position}


class FakeEngine:
    """Same surface the real TorchEngine exposes to NativeContinuation."""

    def __init__(self, plans):
        self.torch = FakeTorch()
        self.pipeline = FakePipeline()
        self.model = FakeModel()
        self.plans = list(plans)
        self.observed = []
        self.path = Path("<fake-model>")
        self.identity = {"sha256": "fake", "bytes": 0}
        self.chunk_tokens = 8

    def observe(self, state, text):
        if state is not None:
            raise AssertionError("the baseline must observe from a fresh state")
        self.observed.append(text)
        plan = self.plans[len(self.observed) - 1]
        return {"plan": plan, "pos": 0}, len(text)

    def assert_gpu(self, state):
        return None


SELF_TEST_CASES = [
    # Synthetic but contract-valid (DATA.validate passes): the assertion slots
    # never copy an 8-character run out of the event, and the target assistant
    # turn is textually different from the history inside `user`, so the
    # no-leakage checks below are meaningful.
    {"id": "S1", "group": DATA.HELDOUT,
     "user": "用户：帮我查一下明天去南京的高铁\n系统：查到3个车次",
     "assistant": "车次是 G7001。", "unverified": False,
     "summary": "主题：火车车次\n约束：无\n未完成：无\n结果：已确认告知车次：G7001"},
    {"id": "S2", "group": DATA.HELDOUT,
     "user": "用户：南京明天天气如何\n系统：明天多云，18到25度",
     "assistant": "空气质量暂时无法确认。", "unverified": True,
     "summary": "主题：天气温度\n约束：无\n未完成：待查：空气质量\n结果：尚未取得空气质量信息"},
    {"id": "S3", "group": DATA.HELDOUT,
     "user": "用户：酒店有早餐吗\n系统：正在为您查询酒店信息",
     "assistant": "这家酒店含双份早餐。", "unverified": False,
     "summary": "主题：酒店早餐\n约束：无\n未完成：无\n结果：已确认告知酒店早餐"},
]


def _chars(text):
    return [ord(char) + 1 for char in text]


def self_test():
    """Drive the real code paths with a fake engine. Returns a check report."""
    import tempfile

    checks = []

    def check(name, condition, detail=None):
        checks.append({"check": name, "ok": bool(condition), "detail": detail})

    # ---- parser and stop policy ------------------------------------------
    good = "主题：A\n约束：无\n未完成：无\n结果：B"
    check("strict parse accepts four ordered non-empty slots", parse_four_slots(good)["ok"])
    check("strict parse rejects missing slot", not parse_four_slots("主题：A\n约束：无\n结果：B")["ok"])
    check("strict parse rejects wrong order",
          not parse_four_slots("约束：无\n主题：A\n未完成：无\n结果：B")["ok"])
    check("strict parse rejects empty value", not parse_four_slots("主题：\n约束：无\n未完成：无\n结果：B")["ok"])
    check("strict parse rejects extra text",
          not parse_four_slots(good + "\n以上。")["ok"])
    check("stop policy stops after four closed lines", four_slot_stop(good + "\n") == "four-slots")
    check("stop policy does not stop mid third line",
          four_slot_stop("主题：A\n约束：无\n未完成：无") is None)
    check("stop policy stops on text after four slots",
          four_slot_stop(good + "\n以上。") == "four-slots")

    # ---- no leakage ------------------------------------------------------
    case = SELF_TEST_CASES[0]
    check("observation is exactly the user field", observation_text(case) == case["user"])
    check("observation excludes the assistant target",
          case["assistant"] not in observation_text(case))
    check("observation excludes the summary",
          DATA.memory_text(case) not in observation_text(case))

    # ---- end to end through run() with a fake engine ----------------------
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        root = Path(td)
        cases_path = root / "cases.jsonl"
        cases_path.write_text(
            "\n".join(json.dumps(case, ensure_ascii=False) for case in SELF_TEST_CASES) + "\n",
            encoding="utf-8", newline="\n")
        plans = [
            _chars(SELF_TEST_CASES[0]["summary"] + "\n"),
            _chars("主题：火车票价\n约束：无\n未完成：无\n结果：已确认告知票价\n"),
            _chars("这段文字没有槽位"),
        ]
        engine = FakeEngine(plans)
        out_dir = root / "out"
        args = parse_args(["--cases", str(cases_path), "--out-dir", str(out_dir),
                           "--skip-power-check", "--cap", "3", "--max-new-tokens", "400",
                           "--gate-every", "2", "--log-every", "100"])
        result = run(args, hooks={"engine": engine})
        check("run completes without CUDA", result["phase"] == "complete", result.get("phase"))
        check("run used no trained decoder", result["trained_decoder_used"] is False)
        records, broken = read_jsonl(out_dir / OUTPUT_NAME)
        check("one record per case", len(records) == 3, len(records))
        check("no unreadable output lines", broken == 0, broken)
        check("observation passed to the engine is the user field only",
              engine.observed == [case["user"] for case in SELF_TEST_CASES], engine.observed)
        check("assistant target never observed",
              all(case["assistant"] not in text for case, text in
                  zip(SELF_TEST_CASES, engine.observed)))
        by_id = {record["id"]: record for record in records}
        check("exact four-slot answer is parsed strictly", by_id["S1"]["strict"]["ok"])
        check("exact answer stops at the four-slot block",
              by_id["S1"]["stop_reason"] == "four-slots", by_id["S1"]["stop_reason"])
        check("exact answer matches the reference",
              by_id["S1"]["tolerant"]["all_slots_exact"] is True)
        check("partial answer is parsed but only the empty slot matches",
              by_id["S2"]["strict"]["ok"] and by_id["S2"]["tolerant"]["slots_exact"] == 1,
              by_id["S2"]["tolerant"]["slots_exact"])
        check("unlabelled answer fails strict parsing", by_id["S3"]["strict"]["ok"] is False)
        check("unlabelled answer stops on EOT", by_id["S3"]["stop_reason"] == "eot",
              by_id["S3"]["stop_reason"])
        metrics = json.loads((out_dir / METRICS_NAME).read_text(encoding="utf-8"))
        check("strict four-slot rate is 2/3", metrics["rates"]["strict_four_slot_rate"] == round(2 / 3, 6),
              metrics["rates"]["strict_four_slot_rate"])
        check("tolerant all-slot rate is 1/3",
              metrics["rates"]["tolerant_all_slots_exact_rate"] == round(1 / 3, 6),
              metrics["rates"]["tolerant_all_slots_exact_rate"])
        check("rates publish the denominator",
              metrics["counts"]["cases_evaluated"] == 3, metrics["counts"])
        check("constant baselines are computed",
              metrics["baselines"]["most_common_reference_message"] is not None)
        check("power gate state is recorded verbatim",
              metrics["power"]["before_cuda"]["power_gate"] == "disabled-by---skip-power-check")
        check("self-test drew no CUDA", metrics["model"]["identity"] == {"sha256": "fake", "bytes": 0})

        # ---- resume: nothing is re-evaluated -----------------------------
        engine2 = FakeEngine(plans)
        result2 = run(args, hooks={"engine": engine2})
        check("resume re-runs no case", result2["counts"]["cases_resumed_from_previous_output"] == 3,
              result2["counts"])
        check("resume does not touch the backbone", engine2.observed == [], engine2.observed)
        records2, _ = read_jsonl(out_dir / OUTPUT_NAME)
        check("resume does not duplicate output lines", len(records2) == 3, len(records2))
        check("resume metrics still describe all records",
              result2["counts"]["cases_evaluated"] == 3, result2["counts"])

        # ---- pause: partial metrics, exit 75 -----------------------------
        pause_dir = root / "paused"
        calls = {"n": 0}

        def pausing_gate(stage):
            calls["n"] += 1
            if calls["n"] > 1:
                raise Paused({"stage": stage, "guard_lease": False, "ac_connected": True,
                              "reason": "no valid guard lease"})
            return {"stage": stage, "guard_lease": True, "ac_connected": True}

        engine3 = FakeEngine(plans)
        args3 = parse_args(["--cases", str(cases_path), "--out-dir", str(pause_dir),
                            "--cap", "3", "--max-new-tokens", "400", "--gate-every", "2",
                            "--log-every", "100"])
        result3 = run(args3, hooks={"engine": engine3, "gate": pausing_gate})
        check("pause exits 75", result3["exit_code"] == EXIT_PAUSED, result3["exit_code"])
        check("pause reports phase paused", result3["phase"] == "paused", result3["phase"])
        records3, _ = read_jsonl(pause_dir / OUTPUT_NAME)
        check("pause keeps the cases already evaluated", len(records3) == 2, len(records3))
        metrics3 = json.loads((pause_dir / METRICS_NAME).read_text(encoding="utf-8"))
        check("pause writes partial metrics with the real denominator",
              metrics3["phase"] == "paused" and metrics3["counts"]["cases_evaluated"] == 2,
              metrics3["counts"])

        # ---- truncated output line is tolerated ---------------------------
        with (pause_dir / OUTPUT_NAME).open("a", encoding="utf-8") as handle:
            handle.write('{"id": "S2", "trunc')
        records4, broken4 = read_jsonl(pause_dir / OUTPUT_NAME)
        check("truncated output line is skipped, not fatal",
              len(records4) == 2 and broken4 == 1, (len(records4), broken4))

        # ---- cap and observation guard ------------------------------------
        small = load_cases(cases_path, 2)
        check("cap limits the selection", [case["id"] for case in small] == ["S1", "S2"],
              [case["id"] for case in small])
        huge = FakeEngine(plans)
        args5 = parse_args(["--cases", str(cases_path), "--out-dir", str(root / "guard"),
                            "--skip-power-check", "--cap", "1", "--max-observation-tokens", "5"])
        result5 = run(args5, hooks={"engine": huge})
        guarded, _ = read_jsonl(Path(root / "guard") / OUTPUT_NAME)
        check("oversize observation is skipped, not truncated",
              guarded[0].get("skipped") == "observation-too-long" and huge.observed == [],
              guarded[0].get("skipped"))
        check("skipped case is excluded from the denominator",
              result5["counts"]["cases_evaluated"] == 0, result5["counts"])

        # ---- missing lease really is fail-closed --------------------------
        def always_paused(stage):
            raise Paused({"stage": stage, "guard_lease": False, "ac_connected": True,
                          "reason": "no valid guard lease"})

        args6 = parse_args(["--cases", str(cases_path), "--out-dir", str(root / "closed")])
        untouched = FakeEngine(plans)
        result6 = run(args6, hooks={"engine": untouched, "gate": always_paused})
        check("no guard lease pauses before CUDA",
              result6["exit_code"] == EXIT_PAUSED and untouched.observed == [],
              result6["exit_code"])
        check("paused-before-CUDA run writes a metrics file",
              (Path(root / "closed") / METRICS_NAME).is_file())

    source = Path(__file__).read_text(encoding="utf-8")
    # Needles are assembled at runtime so this check cannot match its own text.
    check("script contains no backward pass", ("." + "backward(") not in source)
    check("script constructs no optimizer", ("torch." + "optim") not in source)
    failed = [entry for entry in checks if not entry["ok"]]
    return {"ok": not failed, "checks": len(checks), "failed": len(failed),
            "failures": failed, "results": checks}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Native-state baseline: frozen RWKV-7, no training, observe the user "
                    "text, continue with a four-slot Chinese prompt, score the output.")
    parser.add_argument("--cases", default=DEFAULT_CASES,
                        help="evaluation JSONL; defaults to the RiSAWOZ test split")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--model", default=os.environ.get("DBA_RWKV_MODEL", DEFAULT_MODEL))
    parser.add_argument("--cap", type=int, default=0,
                        help="evaluate only the first N selected cases (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-observation-tokens", type=int, default=4096,
                        help="skip (never truncate) cases whose user text exceeds this")
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--gate-every", type=int, default=5,
                        help="re-check AC + guard lease every N cases")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--prompt-style", default="labelled", choices=tuple(PROMPTS) + ("custom",),
                        help="built-in four-slot prompt; 'custom' requires --prompt. "
                             "'labelled' is the default because the frozen model echoed the "
                             "placeholder skeleton of 'placeholder' in the first real run")
    parser.add_argument("--prompt", default=None,
                        help="explicit prompt text; overrides --prompt-style")
    parser.add_argument("--skip-power-check", action="store_true",
                        help="PLUMBING ONLY: disables the AC/guard gate; recorded in metrics")
    parser.add_argument("--self-test", action="store_true",
                        help="run the offline parser/metrics/resume/pause checks and exit")
    args = parser.parse_args(argv)
    if args.prompt is None:
        if args.prompt_style == "custom":
            raise BaselineError("--prompt-style custom requires --prompt")
        args.prompt = PROMPTS[args.prompt_style]
    for name in ("max_new_tokens", "chunk_tokens", "gate_every", "max_observation_tokens"):
        if getattr(args, name) < 1:
            raise BaselineError("--%s must be positive" % name.replace("_", "-"))
    if args.cap < 0:
        raise BaselineError("--cap cannot be negative")
    if args.log_every < 1:
        raise BaselineError("--log-every must be positive")
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.self_test:
            report = self_test()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 1
        result = run(args)
        return int(result.get("exit_code", 0))
    except BaselineError as exc:
        print(json.dumps({"ok": False, "error": "BaselineError: " + str(exc)}, ensure_ascii=False))
        return 1
    except Exception as exc:  # never claim success on an unexpected failure
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
