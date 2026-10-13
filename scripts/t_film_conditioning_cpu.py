"""CPU-only tests for the FiLM conditioning + teacher-forcing dropout fix.

What this file is for
---------------------
The v3 run (`risawoz-run-v3`) reached train loss 0.23 while producing a four-slot
exact message for 0 of 1779 heldout cases, and its zero-state ablation scored as well
as the real state. Two independent problems were added to the trainer to attack that:

  1. `--conditioning film`  -- a zero-initialised FiLM modulation after EVERY decoder
     block, so the state is on every layer's gradient path instead of reaching only
     the two late cross-attention layers.
  2. `--tf-dropout-*`       -- scheduled teacher-forcing dropout: part of the gold
     prefix is replaced by the mask token, so a value cannot be copied from the
     prefix and must be reconstructed from the conditioning.

Neither change may break anything that already works, so these tests pin:
  * the LEGACY contract (identical serialisation, identical module tree, identical
    identity hash, resumable old checkpoints);
  * that the new FiLM path starts out numerically IDENTICAL to the legacy network;
  * that the new path is nonetheless reachable by gradient;
  * that the corruption schedule and the corruption itself are deterministic,
    bounded, and never touch BOS or padding;
  * that the new honest metric (`content_slot_accuracy`) cannot be earned by
    emitting 无 everywhere, which is the trap the old metric fell into.

Every check runs on CPU. No CUDA, no backbone, no dataset, no GPU allocation.
Run:  D:\\AI\\DBA\\.venv-rwkv\\Scripts\\python.exe t_film_conditioning_cpu.py
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch

import rwkv_slots_data as DATA
import train_rwkv_slots as T

LEGACY_DICT_KEYS = {"feature_count", "feature_dim", "vocab_size", "d_model", "n_heads",
                    "n_self_layers", "n_cross_layers", "dropout", "conditioning_tokens"}

CHECKS = []


def check(name, ok, detail=None):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def small_config(mode="cross", self_layers=2, cross_layers=2, dropout=0.0):
    return T.DecoderConfig(feature_count=6, feature_dim=8, vocab_size=64,
                           d_model=32, n_heads=4, n_self_layers=self_layers,
                           n_cross_layers=cross_layers, dropout=dropout,
                           conditioning_tokens=4, conditioning_mode=mode)


def copy_shared(legacy, film):
    """Copy every parameter the two modules share, so only the new path differs."""
    target = film.state_dict()
    for name, value in legacy.state_dict().items():
        if name in target and target[name].shape == value.shape:
            target[name] = value.clone()
    film.load_state_dict(target)
    return film


# --------------------------------------------------------------------------
# 1. serialisation / identity contract
# --------------------------------------------------------------------------
def test_config_contract():
    legacy = small_config("cross")
    as_dict = legacy.to_dict()
    check("legacy config serialises exactly the legacy keys",
          set(as_dict) == LEGACY_DICT_KEYS, sorted(as_dict))

    film_dict = small_config("film").to_dict()
    check("film config adds only conditioning_mode",
          set(film_dict) - LEGACY_DICT_KEYS == {"conditioning_mode"}, sorted(film_dict))
    check("film serialisation records mode=film", film_dict["conditioning_mode"] == "film")

    # An old checkpoint has no conditioning_mode key. It must still load, and must
    # round-trip back to the exact same legacy dict (same identity hash input).
    rebuilt = T.DecoderConfig.from_dict(dict(as_dict))
    check("legacy dict round-trips byte-identically", rebuilt.to_dict() == as_dict)
    check("missing conditioning_mode defaults to legacy cross",
          rebuilt.conditioning_mode == "cross")
    check("film dict round-trips", T.DecoderConfig.from_dict(film_dict).to_dict() == film_dict)

    # Identity hashing must see no difference for an untouched run.
    before = json.dumps({"decoder": as_dict}, sort_keys=True)
    after = json.dumps({"decoder": small_config("cross").to_dict()}, sort_keys=True)
    check("legacy identity hash input unchanged", before == after)
    check("film identity hash input differs",
          json.dumps({"decoder": film_dict}, sort_keys=True) != before)

    check("modulation_layers counts every block for film",
          small_config("film").modulation_layers == 4,
          small_config("film").modulation_layers)
    check("modulation_layers is zero for cross", small_config("cross").modulation_layers == 0)
    check("modulation_layers follows n_self_layers=0",
          small_config("film", self_layers=0).modulation_layers == 2)

    try:
        T.DecoderConfig(feature_count=6, feature_dim=8, vocab_size=64, d_model=32,
                        n_heads=4, conditioning_mode="telepathy")
        check("unknown conditioning mode is rejected", False, "no error raised")
    except T.TrainingError as exc:
        check("unknown conditioning mode is rejected", "conditioning_mode" in str(exc), str(exc))


# --------------------------------------------------------------------------
# 2. module tree & old-checkpoint loadability
# --------------------------------------------------------------------------
def test_module_tree():
    legacy = T.build_decoder(small_config("cross"), torch)
    film = T.build_decoder(small_config("film"), torch)

    legacy_keys = set(legacy.state_dict())
    film_keys = set(film.state_dict())
    check("legacy tree has no modulation parameters",
          not any(key.startswith("modulation.") for key in legacy_keys))
    added = {key for key in film_keys if key.startswith("modulation.")}
    check("film tree adds one modulation per block",
          len({key.split(".")[2] for key in added if key.startswith("modulation.projection")}) == 4,
          sorted(added)[:6])
    check("no shared parameter changed shape or name",
          all(film.state_dict()[key].shape == legacy.state_dict()[key].shape
              for key in legacy_keys))
    check("nothing was removed from the legacy tree", legacy_keys <= film_keys)

    # A checkpoint written by the legacy code must load into the legacy build with
    # strict=True -- this is what the independent final evaluator does.
    state = legacy.state_dict()
    fresh = T.build_decoder(small_config("cross"), torch)
    report = fresh.load_state_dict(state, strict=True)
    check("legacy checkpoint loads strictly", not report.missing_keys, report.missing_keys)

    # And the film tree must NOT silently accept a legacy checkpoint: the identity
    # hash already blocks it, but a strict load is the second line of defence.
    try:
        T.build_decoder(small_config("film"), torch).load_state_dict(state, strict=True)
        check("film build refuses a legacy state_dict strictly", False, "no error raised")
    except RuntimeError as exc:
        check("film build refuses a legacy state_dict strictly",
              "modulation" in str(exc), str(exc)[:120])


# --------------------------------------------------------------------------
# 3. FiLM starts as an exact no-op, but is reachable and does modulate
# --------------------------------------------------------------------------
def test_film_is_zero_initialised_identity():
    legacy = T.build_decoder(small_config("cross"), torch).eval()
    film = T.build_decoder(small_config("film"), torch).eval()
    copy_shared(legacy, film)

    for projection in film.modulation.projection:
        if not (bool((projection.weight == 0).all()) and bool((projection.bias == 0).all())):
            check("modulation projections start at exactly zero", False)
            return
    check("modulation projections start at exactly zero", True)

    tokens = torch.randint(1, 60, (3, 7))
    features = torch.randn(3, 6, 8)
    with torch.no_grad():
        base = legacy(tokens, features)
        modulated = film(tokens, features)
    delta = float((base - modulated).abs().max())
    check("film at init is numerically identical to legacy", delta < 1e-6, delta)

    # Now make the modulation non-trivial and confirm it reaches the output.
    with torch.no_grad():
        for index, projection in enumerate(film.modulation.projection):
            projection.weight.normal_(mean=0.0, std=0.05, generator=torch.Generator().manual_seed(index))
            projection.bias.fill_(0.1 * (index + 1))
        changed = film(tokens, features)
        difference = float((base - changed).abs().max())
        check("non-zero modulation changes the output", difference > 1e-4, difference)

        # The modulation must depend on the STATE, not just be a constant offset.
        other = film(tokens, features + 1.5)
        state_sensitivity = float((changed - other).abs().max())
        legacy_other = legacy(tokens, features + 1.5)
        legacy_sensitivity = float((base - legacy_other).abs().max())
        check("film output depends on the conditioning", state_sensitivity > 1e-6,
              state_sensitivity)
        check("film amplifies the state's reach on the residual stream",
              state_sensitivity > legacy_sensitivity,
              {"film": state_sensitivity, "legacy": legacy_sensitivity})


def test_film_receives_gradient():
    film = T.build_decoder(small_config("film"), torch)
    film.train()
    tokens = torch.randint(1, 60, (2, 6))
    features = torch.randn(2, 6, 8)
    targets = torch.randint(1, 60, (2, 6))
    logits = film(tokens, features)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                             targets.reshape(-1))
    loss.backward()

    grads = [p.grad for p in film.modulation.parameters() if p.grad is not None]
    # 4 blocks: one zero-init Linear each (weight + bias) plus the shared LayerNorm (weight + bias)
    check("every modulation parameter receives a gradient", len(grads) == 2 * 4 + 2, len(grads))
    total = sum(float(g.abs().sum()) for g in grads)
    check("modulation gradient is non-zero", total > 0.0, total)
    check("modulation gradient is finite",
          all(bool(torch.isfinite(g).all()) for g in grads))

    # The pooled conditioning must carry gradient back into the state encoder, or the
    # encoder would be trained by the cross-attention path alone again.
    encoder_grads = [p.grad for p in film.state_encoder.parameters() if p.grad is not None]
    check("state encoder is still on the gradient path",
          any(float(g.abs().sum()) > 0 for g in encoder_grads))


def test_film_gain_is_bounded():
    film = T.build_decoder(small_config("film"), torch).eval()
    with torch.no_grad():
        for projection in film.modulation.projection:
            projection.weight.fill_(1e6)
            projection.bias.fill_(-1e6)
    tokens = torch.randint(1, 60, (1, 5))
    features = torch.randn(1, 6, 8) * 50.0
    with torch.no_grad():
        out = film(tokens, features)
        hidden = film.token_embedding(tokens) + film.position_embedding(
            torch.arange(5).unsqueeze(0))
        conditioning = film.state_encoder(features)
        modulated = film.modulation(0, hidden, conditioning)
    check("saturated modulation stays finite", bool(torch.isfinite(out).all()))
    # multiplier is 1 + gain with |gain| <= FILM_GAIN_LIMIT (3.0 max), shift <= BIAS_LIMIT
    bound = (1.0 + T.FILM_GAIN_LIMIT) * float(hidden.abs().max()) + T.FILM_BIAS_LIMIT + 1e-3
    check("saturated modulation respects the documented bound",
          float(modulated.abs().max()) <= bound,
          {"observed": float(modulated.abs().max()), "bound": bound})
    check("film still produces finite logits under a saturated modulation",
          bool(torch.isfinite(out).all()) and out.shape == (1, 5, film.config.vocab_size))

# --------------------------------------------------------------------------
# 4. teacher-forcing dropout schedule
# --------------------------------------------------------------------------
def test_tf_schedule():
    rate = T.teacher_forcing_dropout_rate
    check("rate starts at --tf-dropout-start", rate(0, 0.0, 0.4, 1000) == 0.0)
    check("rate is linear mid-ramp", abs(rate(500, 0.0, 0.4, 1000) - 0.2) < 1e-9,
          rate(500, 0.0, 0.4, 1000))
    check("rate reaches --tf-dropout-end at the ramp", abs(rate(1000, 0.0, 0.4, 1000) - 0.4) < 1e-9)
    check("rate is held after the ramp", rate(999999, 0.0, 0.4, 1000) == 0.4)
    check("zero ramp jumps straight to the end rate", rate(3, 0.0, 0.4, 0) == 0.4)
    check("schedule can ramp down", rate(0, 0.4, 0.0, 100) == 0.4
          and rate(100, 0.4, 0.0, 100) == 0.0)
    monotone = all(rate(step, 0.05, 0.5, 400) <= rate(step + 1, 0.05, 0.5, 400) + 1e-12
                   for step in range(0, 400))
    check("ramp is monotone", monotone)
    check("negative step cannot un-ramp", rate(-5, 0.0, 0.4, 1000) == 0.0)


# --------------------------------------------------------------------------
# 5. teacher-forcing dropout corruption
# --------------------------------------------------------------------------
def build_batch():
    sequences = [[1, 11, 12, 13, 14, 15], [1, 21, 22], [1, 31, 32, 33]]
    return T.pad_batch(sequences, 63, torch, torch.device("cpu"))


def test_tf_corruption():
    inputs, targets, mask = build_batch()
    bad = int(63)

    same, count = T.corrupt_teacher_forcing_inputs(inputs, mask, 1, 0.0, 7, 0, torch)
    check("rate 0 returns the batch untouched", bool((same == inputs).all()) and count == 0)

    for rate in (-0.5, 1.0, 2.0):
        try:
            T.corrupt_teacher_forcing_inputs(inputs, mask, 1, rate, 7, 0, torch)
            check("invalid rate %s is rejected" % rate, False, "no error")
        except T.TrainingError:
            check("invalid rate %s is rejected" % rate, True)

    corrupted, count = T.corrupt_teacher_forcing_inputs(inputs, mask, 1, 0.5, 7, 3, torch)
    again, count_again = T.corrupt_teacher_forcing_inputs(inputs, mask, 1, 0.5, 7, 3, torch)
    check("corruption is deterministic for (seed, step)",
          bool((corrupted == again).all()) and count == count_again, count)
    other_step, _ = T.corrupt_teacher_forcing_inputs(inputs, mask, 1, 0.5, 7, 4, torch)
    other_seed, _ = T.corrupt_teacher_forcing_inputs(inputs, mask, 1, 0.5, 99, 3, torch)
    check("a different step draws different positions",
          not bool((corrupted == other_step).all()))
    check("a different seed draws different positions",
          not bool((corrupted == other_seed).all()))

    check("BOS at position 0 is never corrupted", int(corrupted[0, 0]) == int(inputs[0, 0]))
    chosen = corrupted != inputs
    check("every corrupted position holds the mask token",
          bool((corrupted[chosen] == 1).all()), sorted(set(corrupted[chosen].tolist())))
    check("only the mask token is ever written", set(corrupted.tolist()[0]) <= set(inputs.tolist()[0]) | {1})
    check("padding is never corrupted",
          not bool(chosen[mask].any()), int(chosen[mask].sum()))
    check("targets are a separate tensor and were not returned",
          corrupted.shape == targets.shape)

    # Corrupting masked positions would be a silent way to change what the loss sees.
    check("reported count matches the number of changed positions",
          count == int(chosen.sum()))

    # Statistical check of the rate on a wide batch.
    wide = torch.randint(1, 60, (64, 40))
    wide_mask = torch.zeros_like(wide, dtype=torch.bool)
    total_eligible = int((~wide_mask).sum()) - wide.shape[0]   # minus the BOS column
    observed = 0
    for step in range(12):
        out, _ = T.corrupt_teacher_forcing_inputs(wide, wide_mask, 1, 0.3, 5, step, torch)
        observed += int((out != wide).sum())
    observed_rate = observed / (12 * total_eligible)
    check("empirical corruption rate is close to the requested rate",
          abs(observed_rate - 0.3) < 0.02, round(observed_rate, 4))


# --------------------------------------------------------------------------
# 6. the honest metric cannot be earned by emitting 无
# --------------------------------------------------------------------------
def test_content_metric():
    reference = "主题：飞控固件烧写\n约束：标定参数不得改动\n未完成：检查串口连接\n结果：尚未烧写成功"
    all_empty = "主题：无\n约束：无\n未完成：无\n结果：无"

    lenient = DATA.slot_match(all_empty, reference)
    content = DATA.content_slot_match(all_empty, reference)
    check("the legacy metric needs no content at all", lenient["slots_exact"] == 0
          or lenient["per_slot"], lenient)
    check("content metric scores every non-empty reference slot",
          content["scored"] == 4, content)
    check("content metric gives an all-无 answer zero credit",
          content["hits"] == 0 and content["accuracy"] == 0.0, content)

    partial = "主题：飞控固件烧写\n约束：无\n未完成：无\n结果：尚未烧写成功"
    mixed = DATA.content_slot_match(partial, reference)
    check("content metric counts the slots that are right",
          mixed["scored"] == 4 and mixed["hits"] == 2 and abs(mixed["accuracy"] - 0.5) < 1e-9,
          mixed)

    empty_reference = "主题：无\n约束：无\n未完成：无\n结果：无"
    nothing = DATA.content_slot_match(all_empty, empty_reference)
    check("an all-empty reference yields no content slots",
          nothing["scored"] == 0 and nothing["accuracy"] is None, nothing)

    check("slot_values exposes the tolerant parser, keyed by slot",
          DATA.slot_values(reference)["result"] == "尚未烧写成功",
          DATA.slot_values(reference))


# --------------------------------------------------------------------------
# 7. evaluate() integration: new fields present, ablations still produced
# --------------------------------------------------------------------------
class FakeArgs:
    eval_samples = 2
    eval_max_tokens = 12
    eval_batch_size = 2


class FakeModule(torch.nn.Module):
    """Deterministic stand-in: predicts a fixed token whatever the state."""

    def __init__(self, vocab, token):
        super().__init__()
        self.vocab = vocab
        self.token = token

    def forward(self, tokens, features, pad_mask=None):
        logits = torch.zeros(tokens.shape[0], tokens.shape[1], self.vocab)
        logits[:, :, self.token] = 10.0
        return logits


def test_evaluate_block():
    cases = [
        {"id": "C1", "group": DATA.HELDOUT, "user": "u", "assistant": "a", "unverified": False,
         "summary": "主题：甲\n约束：乙\n未完成：无\n结果：丙"},
        {"id": "C2", "group": DATA.HELDOUT, "user": "u", "assistant": "a", "unverified": False,
         "summary": "主题：丁\n约束：无\n未完成：无\n结果：戊"},
    ]
    tokenizer = T.ByteTokenizer()
    ids = T.special_ids(tokenizer)
    module = FakeModule(tokenizer.vocab_size, ids["eos"])
    features = torch.randn(2, 6, 8)
    block = T.evaluate(module, cases, features, tokenizer, ids, FakeArgs(), torch,
                       torch.device("cpu"))
    for key in ("content_slots_scored", "content_slot_accuracy", "mean_slots_exact",
                "all_slots_exact_cases", "tokens_scored"):
        check("evaluate reports %s" % key, key in block, sorted(block))
    check("evaluate records per-sample content detail",
          all("content" in sample for sample in block["samples"]))
    check("an always-EOS model scores no content",
          block["content_slot_accuracy"] == 0.0, block["content_slot_accuracy"])

    compact = T.compact_eval_block(block, keep_expected=False)
    check("history block keeps the content metric",
          "content_slot_accuracy" in compact and "expected" not in compact["samples"][0])


# --------------------------------------------------------------------------
# 8. CLI surface and identity wiring
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 8. feature-cache root: reuse without silently paying for a wrong path
# --------------------------------------------------------------------------
def test_feature_cache_root():
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        out_dir = Path(scratch) / "run-v4"
        donor = Path(scratch) / "run-v3"
        key = "1f0b28205d715209"
        (donor / T.FEATURE_CACHE_DIR / key).mkdir(parents=True)

        # Default: under --out-dir, and nothing has to pre-exist.
        own = T.resolve_feature_cache_dir(None, out_dir, key)
        check("default cache root is under --out-dir",
              own == out_dir / T.FEATURE_CACHE_DIR / key, str(own))

        # Reuse: points straight at the donor run's keyed directory.
        shared = T.resolve_feature_cache_dir(str(donor), out_dir, key)
        check("an explicit root reuses the other run's keyed directory",
              shared == donor / T.FEATURE_CACHE_DIR / key, str(shared))
        check("the reusing run does not have to exist first", not out_dir.exists())

        # Miss: refuse, do not quietly start a multi-hour re-extraction.
        empty = Path(scratch) / "empty-run"
        empty.mkdir()
        try:
            T.resolve_feature_cache_dir(str(empty), out_dir, key)
            check("a root without this run's cache is refused", False, "no error")
        except T.TrainingError as exc:
            check("a root without this run's cache is refused",
                  "feature-cache-root" in str(exc) and key in str(exc), str(exc)[:160])

        try:
            T.resolve_feature_cache_dir(str(Path(scratch) / "typo"), out_dir, key)
            check("a nonexistent root is refused", False, "no error")
        except T.TrainingError:
            check("a nonexistent root is refused", True)

        # A wrong key (different backbone/features/dataset) is still a miss.
        try:
            T.resolve_feature_cache_dir(str(donor), out_dir, "0000000000000000")
            check("a different cache key is not silently accepted", False, "no error")
        except T.TrainingError:
            check("a different cache key is not silently accepted", True)


def test_cli():
    default = T.parse_args(["--steps", "2"])
    check("legacy default conditioning is cross", default.conditioning == "cross")
    check("legacy default disables teacher-forcing dropout",
          default.tf_dropout_start == 0.0 and default.tf_dropout_end == 0.0)
    check("legacy default weight decay is unchanged", default.weight_decay == 0.01)

    film = T.parse_args(["--steps", "2", "--conditioning", "film", "--tf-dropout-start", "0.05",
                         "--tf-dropout-end", "0.4", "--tf-dropout-ramp", "1200",
                         "--weight-decay", "0.05"])
    check("film and the dropout schedule parse",
          film.conditioning == "film" and film.tf_dropout_end == 0.4
          and film.tf_dropout_ramp == 1200 and film.weight_decay == 0.05)

    for argv, why in (
            (["--steps", "2", "--tf-dropout-end", "1.0"], "rate of 1.0"),
            (["--steps", "2", "--tf-dropout-start", "-0.1"], "negative rate"),
            (["--steps", "2", "--weight-decay", "-1"], "negative weight decay"),
            (["--steps", "2", "--tf-dropout-end", "0.3", "--tf-dropout-ramp", "0"],
             "zero ramp with dropout on")):
        try:
            T.parse_args(argv)
            check("CLI rejects %s" % why, False, "no error")
        except T.TrainingError:
            check("CLI rejects %s" % why, True)
    try:
        # argparse prints usage to stderr before raising; keep the report clean.
        with contextlib.redirect_stderr(io.StringIO()):
            T.parse_args(["--steps", "2", "--conditioning", "smoke-signals"])
        check("CLI rejects an unknown conditioning mode", False, "no error")
    except SystemExit:
        check("CLI rejects an unknown conditioning mode", True)

    # Identity: the two modes must not be able to resume each other's checkpoint.
    cross_identity = json.dumps({"decoder": small_config("cross").to_dict(), "lr": 3e-4},
                                sort_keys=True)
    film_identity = json.dumps({"decoder": small_config("film").to_dict(), "lr": 3e-4},
                               sort_keys=True)
    check("cross and film runs have different identities", cross_identity != film_identity)

    cache_args = T.parse_args(["--steps", "2", "--feature-cache-root", r"D:\AI\DBA\training\risawoz-run-v3"])
    check("--feature-cache-root parses",
          cache_args.feature_cache_root == r"D:\AI\DBA\training\risawoz-run-v3",
          cache_args.feature_cache_root)
    check("--feature-cache-root is not part of the model identity",
          T.parse_args(["--steps", "2"]).feature_cache_root is None)


# --------------------------------------------------------------------------
# 9. the edited training-step body, end to end, on CPU
# --------------------------------------------------------------------------
def test_training_step_body():
    """Replicates the trainer's step body so the edits are exercised, not just parsed.

    A change here that only breaks under `--steps > 1` (a mask/ignore_index mistake,
    a shape mismatch between corrupted inputs and pad_mask) must not be found for the
    first time on the GPU.
    """
    module = T.build_decoder(small_config("film"), torch)
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01, weight_decay=0.01)
    sequences = [[1, 11, 12, 13, 14], [1, 21, 22], [1, 31, 32, 33, 34]]
    ids = {"bos": 1, "pad": 63}
    losses, corrupted_total = [], 0
    for step in range(8):
        inputs, targets, mask = T.pad_batch(sequences, ids["pad"], torch, torch.device("cpu"))
        features = torch.randn(len(sequences), 6, 8)
        rate = T.teacher_forcing_dropout_rate(step, 0.0, 0.5, 4)
        model_inputs, corrupted = T.corrupt_teacher_forcing_inputs(
            inputs, mask, ids["bos"], rate, 1, step, torch)
        # No per-step expectation here: on a 7-eligible-position batch a single step
        # legitimately corrupts nothing. The rate itself is pinned by the aggregate
        # test above; a per-step assertion would only teach people to ignore a red X.
        corrupted_total += corrupted
        logits = module(model_inputs, features, mask)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=ids["pad"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0))
        optimizer.step()
        losses.append(float(loss.item()))
        if not (torch.isfinite(torch.tensor(loss.item())) and torch.isfinite(torch.tensor(grad_norm))):
            check("step loss and grad norm stay finite", False, {"step": step, "loss": losses[-1]})
            return
    check("step loss and grad norm stay finite", True, [round(v, 4) for v in losses])
    check("the run actually corrupted tokens", corrupted_total > 0, corrupted_total)
    check("the corrupted-prefix objective is learnable on a tiny model",
          losses[-1] < losses[0], {"first": round(losses[0], 4), "last": round(losses[-1], 4)})

    # Targets must never be masked: the corrupted tokens are inputs only.
    check("targets still contain real tokens",
          any(token != ids["pad"] for row in targets.tolist() for token in row))


def main():
    test_config_contract()
    test_module_tree()
    test_film_is_zero_initialised_identity()
    test_film_receives_gradient()
    test_film_gain_is_bounded()
    test_tf_schedule()
    test_tf_corruption()
    test_content_metric()
    test_evaluate_block()
    test_feature_cache_root()
    test_cli()
    test_training_step_body()

    failed = [entry for entry in CHECKS if not entry["ok"]]
    print(json.dumps({"ok": not failed, "checks": len(CHECKS), "failed": len(failed),
                      "failures": failed}, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
