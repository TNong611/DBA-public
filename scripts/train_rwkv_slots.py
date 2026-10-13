"""PILOT: autoregressive memory-text decoder conditioned on a FROZEN RWKV-7 state.

SCOPE -- read before quoting any number this produces
-----------------------------------------------------
This trains a small decoder-only network that reads the frozen RWKV-7 recurrent
state of one closed event and emits real, readable four-line memory text
(主题/约束/未完成/结果), scored by token-level cross entropy.

It is NOT fine-tuning of RWKV-7. The 1.5B backbone receives no gradients, no
LoRA, no adapters and no changed weights: it is used only as a frozen feature
extractor (`TorchEngine.observe`). Only the decoder below is trained.

Stages (separable, each with its own artifact):
  1. BACKBONE (frozen)  event text -> recurrent state S
  2. FEATURES (frozen)  S -> [3*n_layer, d_model] normalised summary
  3. STATE ENCODER (trained)  features -> a few conditioning tokens (bottleneck,
     so the decoder cannot memorise one giant per-case vector)
  4. DECODER (trained)  BOS + already generated tokens x conditioning -> next token

Ablation (--state-mode) decides whether the decoder can see the real state at
all; it is the point of the pilot:
  true      real state from the frozen backbone (the claim under test)
  none      ZERO conditioning: quality can only come from label priors, not state
  shuffled  real states permuted across cases: conditioning is present but wrong,
            which separates "read the state" from "used some vector"

Safety (AC-only, fail-closed, cooperative with training_power_guard.py):
  * direct AC check before torch/CUDA is touched
  * guard lease + direct AC before and during feature extraction (that pass is
    long and monopolises the GPU)
  * guard lease + direct AC at every optimizer boundary via
    training_power_control.pause_if_requested
  * exit code 75 on AC loss, after an atomic checkpoint
  * --skip-power-check is accepted ONLY with --dry-run (synthetic plumbing)

Resume determinism: the checkpoint carries model, optimizer, RNG
(torch/cuda/numpy/python), step, and run identity (backbone hash, tokenizer, data
hash, feature hash, ablation mode, hyperparameters). Writes are atomic
(temp + fsync + os.replace) with a pointer file, so a partial write is never
loadable. A checkpoint from another identity is refused, not silently used.

Feature cache (--feature-cache, on by default): stage 2 rows are written per case
under `<out_dir>/features/<cache key>/`, one atomically written shard per ~100
cases, keyed on backbone identity + FEATURE_VERSION + the source of
`state_features` + the data hash. A pause therefore costs a reload, not another
pass of the frozen 1.5B backbone over every case. Cached rows are RAW: the
train-only standardisation still happens after assembly. Note the backbone is
still loaded before the cache is consulted, because the cache key contains the
engine's own identity; paying that load is what keeps a stale cache from being
silently accepted.

How much extraction work a pause can lose: a shutdown that runs Python's cleanup
(AC pause, Ctrl-C, an exception) flushes the rows still in memory first, so it
loses nothing. Only a hard kill -- task manager, power cut, a crash inside the
CUDA driver -- can lose rows, and never more than `--feature-cache-every`
(default 100) cases, because a row is durable only once its shard is written.
The reported counters distinguish on-disk rows from rows reused at startup.

Evaluation: the heldout teacher-forced loss runs in batches (--eval-batch-size)
and is aggregated token-weighted, so a large split cannot OOM the run; generated
samples are spread across the whole heldout array, and every evaluation appends
step + true/no_state/wrong_state to `<out_dir>/eval-history.jsonl`.

Run with the CUDA venv:
  D:\\AI\\DBA\\.venv-rwkv\\Scripts\\python.exe train_rwkv_slots.py --steps 300
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
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
    from training_power_control import training_allowed, pause_if_requested
except Exception:  # pragma: no cover - only where the guard is not installed
    def training_allowed():
        return False

    def pause_if_requested(save_checkpoint):
        save_checkpoint()
        raise SystemExit(75)

LABEL = DATA.LABEL
BACKBONE_TRAINING = DATA.BACKBONE_TRAINING
FORMAT = 1
FEATURE_VERSION = 1
SCRIPT_VERSION = "train_rwkv_slots.py/1"
EXIT_PAUSED = 75
STATE_MODES = ("true", "none", "shuffled")
# How the state reaches the decoder.
#   cross : legacy. State becomes `conditioning_tokens` vectors that ONLY the
#           cross-attention layers can see, as a plain residual at the end.
#   film  : the same cross-attention plus a zero-initialised FiLM modulation
#           after EVERY decoder block. Because it is zero-initialised the module
#           starts out numerically identical to `cross`, so this can only add a
#           path for the state -- it cannot damage the starting point.
CONDITIONING_MODES = ("cross", "film")
DEFAULT_CONDITIONING_MODE = "cross"
FILM_GAIN_LIMIT = 2.0   # multiplier (1 + gain) stays inside [-1, 3]
FILM_BIAS_LIMIT = 4.0   # additive shift stays inside [-4, 4]
DEFAULT_MODEL = r"D:\AI\models\rwkv7\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"
DEFAULT_OUT = Path(r"D:\AI\DBA\training\rwkv-slots")
RESUME_NAME = "checkpoint-last.pt"
PAUSE_NAME = "checkpoint-pause.pt"
POINTER_NAME = "latest.json"
HISTORY_LIMIT = 200
EOT_ID = DATA.RWKV_EOT_ID
FEATURE_CACHE_DIR = "features"
FEATURE_CACHE_MANIFEST = "manifest.json"
FEATURE_CACHE_FORMAT = 1
FEATURE_CACHE_EVERY_DEFAULT = 100
FEATURE_PROGRESS_EVERY = 100
EVAL_HISTORY_NAME = "eval-history.jsonl"


class TrainingError(RuntimeError):
    """Refusal or configuration error. Never used for an AC pause."""


class PowerLoss(TrainingError):
    """No guard lease, or the machine left AC power.

    Distinct from TrainingError on purpose: this is the one condition the pilot
    may recover from, so it saves a checkpoint and exits 75 instead of failing.
    """

    def __init__(self, report):
        super().__init__("%s: guard_lease=%s ac_connected=%s"
                         % (report["stage"], report["guard_lease"], report["ac_connected"]))
        self.report = report


class Done(Exception):
    """Internal control flow: the run already produced its final result."""

    def __init__(self, result):
        super().__init__("run finished: " + str(result.get("phase")))
        self.result = result


# --------------------------------------------------------------------------
# power / fail-closed
# --------------------------------------------------------------------------
class _PowerStatus(ctypes.Structure):
    # Explicit widths so the struct is 12 bytes on any platform, matching Windows.
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]


def main_connected():
    """Direct AC state. Unavailable power status NEVER authorises training."""
    if os.name != "nt":
        return False
    status = _PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return False
    return int(status.ACLineStatus) == 1


def power_report(stage, require_lease=True):
    """Fail-closed gate used before CUDA work and at optimizer boundaries."""
    lease = bool(training_allowed())
    ac = bool(main_connected())
    report = {"stage": stage, "guard_lease": lease, "ac_connected": ac}
    if require_lease and not lease:
        raise PowerLoss({**report, "reason": "no valid guard lease"})
    if not ac:
        raise PowerLoss({**report, "reason": "AC power not connected"})
    return report


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------
class BackboneTokenizer:
    """Real RWKV tokenizer. State features and decoder targets share it."""
    name = "rwkv_vocab_v20230424"

    def __init__(self, pipeline, vocab_size):
        self.pipeline = pipeline
        self.vocab_size = int(vocab_size) + 3

    def encode(self, text):
        tokens = [int(t) for t in self.pipeline.encode(text)]
        if not tokens:
            raise TrainingError("tokenizer produced no tokens")
        return tokens

    def decode(self, tokens):
        return self.pipeline.decode([int(t) for t in tokens])


class ByteTokenizer:
    """Synthetic byte fallback for --dry-run plumbing tests, not a real tokenizer.

    Real ids are BASE + 1 .. BASE + SPAN (one per byte value), and BOS/EOS/PAD
    sit above SPAN so the embedding table has room and no gather runs out of
    bounds. Ids are derived from SPAN rather than written twice, because a
    mismatch here is a silent device-side assert, not a Python error.
    """
    name = "synthetic-byte"
    SPAN = 256

    def __init__(self, span=None):
        self.span = int(span or self.SPAN)

    @property
    def real_ids(self):
        return EOT_ID + 1, EOT_ID + self.span

    @property
    def vocab_size(self):
        return EOT_ID + self.span + 1 + 3  # real ids, then BOS/EOS/PAD

    def encode(self, text):
        if self.span != self.SPAN:
            raise TrainingError("narrow synthetic tables cannot encode text")
        return [EOT_ID + 1 + b for b in text.encode("utf-8")]

    def decode(self, tokens):
        raw = bytearray()
        for token in tokens:
            if EOT_ID < token <= EOT_ID + self.span:
                raw.append(token - EOT_ID - 1)
            else:
                break  # specials are never part of generated text
        return bytes(raw).decode("utf-8", errors="replace")


def special_ids(tokenizer):
    """BOS/EOS/PAD live above every real token; validated before use."""
    bos = tokenizer.vocab_size - 3
    return {"bos": bos, "eos": bos + 1, "pad": bos + 2, "first": bos}


def widest_real_id(tokenizer, texts):
    return max(max(tokenizer.encode(text)) for text in texts)


# --------------------------------------------------------------------------
# checkpoint io
# --------------------------------------------------------------------------
def replace_with_retry(temporary, path):
    """os.replace, retried: Windows briefly refuses to replace a file a reader
    (an antivirus scanner, a status polling job) still has open."""
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path, payload, torch):
    """temp + fsync + replace: a torn shard must never be loadable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def rng_state(torch):
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy
        state["numpy"] = numpy.random.get_state()
    except Exception:
        state["numpy"] = None
    return state


def restore_rng(state, torch):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("numpy") is not None:
        try:
            import numpy
            numpy.random.set_state(state["numpy"])
        except Exception:
            pass


def cpu_optimizer_state(optimizer):
    import copy
    state = copy.deepcopy(optimizer.state_dict())
    for entry in state.get("state", {}).values():
        for key, value in list(entry.items()):
            if hasattr(value, "detach"):
                entry[key] = value.detach().to("cpu")
    return state


def save_checkpoint(path, payload, torch, pointer=None):
    """Atomically persist everything a deterministic resume needs."""
    atomic_torch_save(path, payload, torch)
    if pointer is not None:
        write_json(pointer, {"checkpoint": str(path), "step": payload["step"],
                             "phase": payload["phase"], "saved_at": time.time()})
    return str(path)


def load_checkpoint(path, torch):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise TrainingError("checkpoint format invalid: " + str(path))
    for key in ("identity_hash", "step", "model_state", "optimizer_state", "rng"):
        if key not in payload:
            raise TrainingError("checkpoint is missing %s: %s" % (key, path))
    return payload


# --------------------------------------------------------------------------
# decoder
# --------------------------------------------------------------------------
class DecoderConfig:
    # `FIELDS` stays exactly the legacy field set on purpose: the independent
    # final evaluator checks the checkpoint's decoder dict against this list, and
    # `to_dict` omits the newer options while they sit at their legacy defaults.
    # A run that never touches the new flags therefore serialises byte-identically
    # and keeps its original identity hash, so existing checkpoints still resume.
    FIELDS = ("feature_count", "feature_dim", "vocab_size", "d_model", "n_heads",
              "n_self_layers", "n_cross_layers", "dropout", "conditioning_tokens")
    OPTIONAL_FIELDS = {"conditioning_mode": DEFAULT_CONDITIONING_MODE}

    def __init__(self, feature_count, feature_dim, vocab_size, d_model=256, n_heads=8,
                 n_self_layers=2, n_cross_layers=2, dropout=0.1, conditioning_tokens=4,
                 conditioning_mode=DEFAULT_CONDITIONING_MODE):
        self.feature_count = int(feature_count)
        self.feature_dim = int(feature_dim)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_self_layers = int(n_self_layers)
        self.n_cross_layers = int(n_cross_layers)
        self.dropout = float(dropout)
        self.conditioning_tokens = int(conditioning_tokens)
        self.conditioning_mode = str(conditioning_mode)
        self.validate()

    @property
    def conditioning_dim(self):
        return self.feature_count * self.feature_dim

    @property
    def modulation_layers(self):
        """One FiLM modulation per decoder block, in forward order."""
        if self.conditioning_mode == "film":
            return self.n_self_layers + self.n_cross_layers
        return 0

    def validate(self):
        if self.d_model < 16 or self.d_model % self.n_heads:
            raise TrainingError("d_model must be >= 16 and divisible by n_heads")
        if self.vocab_size < 16:
            raise TrainingError("vocab_size too small")
        if self.conditioning_dim < 1:
            raise TrainingError("conditioning dimension must be positive")
        if not 1 <= self.conditioning_tokens <= 64:
            raise TrainingError("conditioning_tokens must be 1..64")
        if not 1 <= self.n_cross_layers <= 8:
            raise TrainingError("n_cross_layers must be 1..8 (the state must reach the decoder)")
        if not 0 <= self.n_self_layers <= 8:
            raise TrainingError("n_self_layers must be 0..8")
        if not 0.0 <= self.dropout < 1.0:
            raise TrainingError("dropout must be in [0,1)")
        if self.feature_count < 1 or self.feature_dim < 1:
            raise TrainingError("state feature shape must be positive")
        if self.conditioning_mode not in CONDITIONING_MODES:
            raise TrainingError("conditioning_mode must be one of %s"
                                % (", ".join(CONDITIONING_MODES),))

    def to_dict(self):
        data = {"feature_count": self.feature_count, "feature_dim": self.feature_dim,
                "vocab_size": self.vocab_size, "d_model": self.d_model,
                "n_heads": self.n_heads, "n_self_layers": self.n_self_layers,
                "n_cross_layers": self.n_cross_layers, "dropout": self.dropout,
                "conditioning_tokens": self.conditioning_tokens}
        # Only serialise an option once it leaves its legacy default: a legacy run
        # must hash to exactly the identity it hashed to before this option existed.
        for name, default in self.OPTIONAL_FIELDS.items():
            value = getattr(self, name)
            if value != default:
                data[name] = value
        return data

    @classmethod
    def from_dict(cls, data):
        args = {k: data[k] for k in cls.FIELDS}
        for name, default in cls.OPTIONAL_FIELDS.items():
            args[name] = data.get(name, default)
        return cls(**args)


def build_decoder(config, torch):
    """Causal self-attention over generated text + cross-attention to the state."""
    nn = torch.nn

    class StateEncoder(nn.Module):
        """Bottleneck: the decoder may only see `conditioning_tokens` vectors.

        Deliberately small. The raw state summary is tens of thousands of floats,
        so handing it over uncondensed would let the decoder memorise one giant
        per-case code instead of reading the state -- which would make the
        ablation meaningless. Here the whole summary is reduced to T vectors
        before any cross-attention sees it.
        """

        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            # Linear applies to the last dimension, so [B, T] x weight[T, d_model]
            # yields exactly [B, T, d_model]: T conditioning tokens of width d_model.
            hidden = max(2 * cfg.d_model, cfg.conditioning_tokens)
            self.hidden = nn.Linear(cfg.feature_dim, hidden)
            self.codes = nn.Linear(hidden, cfg.d_model)
            self.norm = nn.LayerNorm(cfg.d_model)

        def forward(self, features):
            # Preserve the recurrent state's row structure. Mean pooling erased
            # layer/slot identity and made the condition easy to ignore.
            pooled = features
            if pooled.shape[1] != self.cfg.conditioning_tokens:
                pooled = torch.nn.functional.adaptive_avg_pool1d(
                    pooled.transpose(1, 2), self.cfg.conditioning_tokens).transpose(1, 2)
            hidden = torch.relu(self.hidden(pooled))
            expanded = self.codes(hidden)
            return self.norm(expanded)

    class StateModulation(nn.Module):
        """Zero-initialised FiLM, one modulation per decoder block.

        Why this exists
        ---------------
        The legacy path hands the state to the last two cross-attention layers
        only. With templated label text a decoder can reach a low teacher-forced
        loss while never reading that conditioning at all -- which is exactly what
        the v3 run did: 0/1779 four-slot exact, and a zero-state ablation that was
        just as good as the real state.

        FiLM touches the residual stream after EVERY block, so the state is on the
        gradient path of every layer rather than one late side door. Both `gamma`
        and `beta` start at exactly zero, so at step 0 the network is numerically
        the legacy network: the new path can only be *learned into* usefulness, it
        cannot perturb a working starting point. Both are tanh-bounded so a run
        cannot destabilise itself by discovering a huge gain.
        """

        def __init__(self, cfg, layers):
            super().__init__()
            self.cfg = cfg
            self.layers = int(layers)
            self.pool = nn.LayerNorm(cfg.d_model)
            self.projection = nn.ModuleList(
                [nn.Linear(cfg.d_model, 2 * cfg.d_model) for _ in range(self.layers)])
            for projection in self.projection:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

        def forward(self, index, hidden, conditioning):
            pooled = self.pool(conditioning).mean(dim=1)             # [B, d_model]
            gamma, beta = self.projection[index](pooled).chunk(2, dim=-1)
            gain = FILM_GAIN_LIMIT * torch.tanh(gamma).unsqueeze(1)
            shift = FILM_BIAS_LIMIT * torch.tanh(beta).unsqueeze(1)
            return hidden * (1.0 + gain) + shift

    class Decoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.state_encoder = StateEncoder(cfg)
            self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.vocab_size - 1)
            self.position_embedding = nn.Embedding(1024, cfg.d_model)
            self.self_blocks = nn.ModuleList(
                [nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_model * 4,
                                            dropout=cfg.dropout, batch_first=True,
                                            norm_first=True)
                 for _ in range(cfg.n_self_layers)])
            self.cross_blocks = nn.ModuleList(
                [nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout,
                                       batch_first=True)
                 for _ in range(cfg.n_cross_layers)])
            self.cross_norms = nn.ModuleList([nn.LayerNorm(cfg.d_model)
                                              for _ in range(cfg.n_cross_layers)])
            # `None` (legacy) means the attribute does not exist at all, so the
            # state_dict of a legacy checkpoint still loads with strict=True.
            self.modulation = (StateModulation(cfg, cfg.modulation_layers)
                               if cfg.modulation_layers else None)
            self.final_norm = nn.LayerNorm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.apply(self._init)
            if self.modulation is not None:
                # `_init` above overwrites every Linear with a small random draw.
                # Re-zero the modulation so the run provably starts as the legacy
                # network instead of merely "almost" it.
                for projection in self.modulation.projection:
                    nn.init.zeros_(projection.weight)
                    nn.init.zeros_(projection.bias)

        @staticmethod
        def _init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, tokens, features, pad_mask=None):
            length = tokens.shape[1]
            positions = torch.arange(length, device=tokens.device).unsqueeze(0)
            hidden = self.token_embedding(tokens) + self.position_embedding(positions)
            if pad_mask is None:
                pad_mask = torch.zeros(tokens.shape, dtype=torch.bool, device=tokens.device)
            conditioning = self.state_encoder(features)
            causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                           device=tokens.device), diagonal=1)
            index = 0
            for layer in self.self_blocks:
                hidden = layer(hidden, src_mask=causal, src_key_padding_mask=pad_mask)
                if self.modulation is not None:
                    hidden = self.modulation(index, hidden, conditioning)
                    index += 1
            for layer, norm in zip(self.cross_blocks, self.cross_norms):
                query = norm(hidden)
                attended, _ = layer(query, conditioning, conditioning, need_weights=False)
                hidden = hidden + attended
                if self.modulation is not None:
                    hidden = self.modulation(index, hidden, conditioning)
                    index += 1
            return self.lm_head(self.final_norm(hidden))

    return Decoder(config)


def decoder_parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# state features
# --------------------------------------------------------------------------
def state_features(state, n_layer, torch):
    """[3*n_layer, d_model] zero-centred, RMS-normalised summary of a state.

    RWKV-7 keeps 3 tensors per layer: a (n_embd,), a (heads, head_size, head_size)
    attention-like matrix and another (n_embd,). The matrix is row-meaned so every
    row has the same width, which keeps the group count fixed across layers.
    """
    if not isinstance(state, list) or len(state) < n_layer * 3:
        raise TrainingError("recurrent state has fewer tensors than the layer stack")
    tensors = state[len(state) - n_layer * 3:]
    rows = []
    target_width = int(tensors[0].numel())
    for value in tensors:
        if not hasattr(value, "reshape"):
            raise TrainingError("recurrent state contains a non-tensor entry")
        vector = value.float().reshape(-1)
        if vector.numel() != target_width:
            if vector.numel() > target_width:
                vector = torch.nn.functional.adaptive_avg_pool1d(vector.reshape(1, 1, -1), target_width).reshape(-1)
            else:
                repeats = (target_width + vector.numel() - 1) // vector.numel()
                vector = vector.repeat(repeats)[:target_width]
        vector = vector - vector.mean()
        rms = vector.pow(2).mean().sqrt().clamp_min(1e-6)
        rows.append(vector / rms)
    if len(rows) != n_layer * 3:
        raise TrainingError("unexpected recurrent tensor count %d (want %d)"
                            % (len(rows), n_layer * 3))
    widths = {int(row.numel()) for row in rows}
    if len(widths) != 1:
        raise TrainingError("recurrent state rows have inconsistent widths: " + str(sorted(widths)))
    return torch.stack(rows, dim=0)


# --------------------------------------------------------------------------
# state feature cache (durable, resumable, per case)
# --------------------------------------------------------------------------
def feature_function_fingerprint():
    """Fingerprint of the code that turns a recurrent state into feature rows.

    Editing `state_features` silently changes what a cached row MEANS, so the
    cache key has to change with it. Hashing the source makes that automatic;
    bumping FEATURE_VERSION by hand relies on someone remembering to.
    """
    import inspect
    try:
        source = inspect.getsource(state_features)
    except Exception:  # pragma: no cover - source may be unavailable
        return "source-unavailable"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def feature_cache_identity(backbone_report, data_hash, n_layer, synthetic_width):
    """Everything a cached feature row depends on. Any change -> other cache."""
    return {
        "feature_version": FEATURE_VERSION,
        "feature_function": feature_function_fingerprint(),
        "backbone": backbone_report.get("identity") or {"mode": backbone_report["mode"]},
        "backbone_mode": backbone_report["mode"],
        "data_hash": data_hash,
        "n_layer": int(n_layer),
        "synthetic_width": None if synthetic_width is None else int(synthetic_width),
        "dtype": "float32",
    }


def feature_cache_key(identity):
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_feature_cache_dir(cache_root, out_dir, key_prefix):
    """Where this run's durable feature cache lives on disk.

    Why a separate root
    -------------------
    The cache key already covers the backbone identity, the feature function and the
    dataset hash, so two runs with the same key may legitimately share rows. A NEW
    run directory would otherwise re-run the frozen 1.5B backbone over every case --
    hours of GPU -- for rows that are already sitting on disk, byte-identical.

    FAIL CLOSED on a miss. `--feature-cache-root` pointing somewhere without this
    run's cache is far more likely a typo than an intention, and the cost of guessing
    wrong is a silent full re-extraction. The caller is told to fix the path instead.
    """
    if not cache_root:
        return Path(out_dir) / FEATURE_CACHE_DIR / key_prefix
    root = Path(cache_root).expanduser()
    target = root / FEATURE_CACHE_DIR / key_prefix
    if not target.is_dir():
        raise TrainingError(
            "--feature-cache-root %s holds no cache for this run (expected %s). Reusing an "
            "earlier run's rows is the point of this flag, so refusing rather than silently "
            "re-extracting every case; point it at the run directory that has the cache or "
            "drop the flag to build one under --out-dir." % (root, target))
    return target


class FeatureCache:
    """Per-case durable cache of GPU-extracted state features under `out_dir`.

    WHY: extraction runs the frozen backbone over every case. That pass is long,
    and an AC pause restarts the whole process; without a cache every pause pays
    the full cost again.

    STORED RAW, NOT NORMALISED. Standardisation happens after the whole matrix is
    assembled and uses train-split statistics only; caching normalised rows would
    bake in the statistics of whatever split existed when the shard was written.

    LAYOUT (one directory per cache key, so two identities can never mix rows):
        manifest.json          key, identity, shard names, feature shape
        shard-000000.pt        ids + float32 features + observation token counts
        shard-000000.ids.json  small index: ids and token counts, no features

    The separate small index is what makes a resume cheap: the run learns which
    cases are already cached WITHOUT loading every feature tensor back to RAM.
    A shard is written atomically, then the manifest is updated atomically, so a
    crash between the two leaves an unreferenced shard that is simply rewritten
    later; a manifest never points at a half-written shard.

    Durability boundary: rows only survive once their shard is written, so
    `flush()` is called on any orderly exit (see the caller) and at most
    `every` rows can be lost to a hard kill.
    """

    def __init__(self, root, key, identity, torch, every=FEATURE_CACHE_EVERY_DEFAULT):
        self.root = Path(root)
        self.key = key
        self.identity = identity
        self.torch = torch
        self.every = max(1, int(every))
        self.manifest = None
        self.shards = []          # [{"name", "ids", "tokens"}]
        self.index = {}           # case id -> (shard number, row, observation tokens)
        self.shape = None
        self.pending = []
        self.cache_hits = 0
        self.extracted = 0
        self.discarded = None
        self.reused_at_startup = 0
        self._open_name = None
        self._open_tensor = None
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- startup ---------------------------------------------------------
    def load(self):
        """Validate the cache on disk. Returns the case ids it can supply.

        An unusable cache is discarded rather than mixed, and the reason is kept
        in `discarded` so the run can report it instead of degrading silently.
        """
        path = self.root / FEATURE_CACHE_MANIFEST
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            self._discard("manifest unreadable: %s" % exc)
            return []
        if not isinstance(manifest, dict) or manifest.get("format") != FEATURE_CACHE_FORMAT:
            self._discard("manifest format mismatch")
            return []
        if manifest.get("cache_key") != self.key:
            self._discard("cache key mismatch (backbone, features or data changed)")
            return []
        if manifest.get("identity") != self.identity:
            self._discard("stored identity differs from this run")
            return []
        self.index = {}
        self.reused_at_startup = 0
        try:
            shards = self._read_shard_indexes(manifest)
        except (OSError, ValueError) as exc:
            self._discard("shard index unusable: %s" % exc)
            return []
        cases = 0
        for number, shard in enumerate(shards):
            for row, case_id in enumerate(shard["ids"]):
                self.index[case_id] = (number, row, int(shard["tokens"][row]))
                cases += 1
        self.manifest = manifest
        self.shards = shards
        shape = manifest.get("feature_shape")
        self.shape = None if not shape else (int(shape[0]), int(shape[1]))
        self.reused_at_startup = cases
        return [case_id for shard in shards for case_id in shard["ids"]]

    def _read_shard_indexes(self, manifest):
        shards = []
        seen = set()
        for name in manifest.get("shards", []):
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("shard name is not a plain file name: %r" % (name,))
            if not (self.root / name).is_file():
                raise ValueError("shard %s is missing" % name)
            index = json.loads((self.root / (name + ".ids.json")).read_text(encoding="utf-8"))
            if index.get("key") != self.key:
                raise ValueError("shard %s belongs to another cache key" % name)
            ids = index.get("ids")
            tokens = index.get("observation_tokens")
            if not isinstance(ids, list) or not isinstance(tokens, list) or len(ids) != len(tokens):
                raise ValueError("shard %s index is malformed" % name)
            for case_id in ids:
                if not isinstance(case_id, str) or case_id in seen:
                    raise ValueError("case %r appears twice in the cache" % (case_id,))
                seen.add(case_id)
            shards.append({"name": name, "ids": ids, "tokens": tokens})
        return shards

    def _discard(self, reason):
        self.discarded = reason
        self.manifest = None
        self.shards = []
        self.index = {}
        self.shape = None
        self.reused_at_startup = 0
        self._open_name = None
        self._open_tensor = None
        for path in sorted(self.root.glob("shard-*")):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            (self.root / FEATURE_CACHE_MANIFEST).unlink()
        except OSError:
            pass

    # ---- rows ------------------------------------------------------------
    def lookup(self, case_id):
        """(feature row, observation tokens) or (None, None) when not cached."""
        entry = self.index.get(case_id)
        if entry is None:
            return None, None
        shard_number, row, tokens = entry
        tensor = self._open_shard(self.shards[shard_number]["name"])
        self.cache_hits += 1
        return tensor[row], tokens

    def _open_shard(self, name):
        """One shard resident at a time: rows are requested in case order."""
        if self._open_name == name:
            return self._open_tensor
        payload = self.torch.load(self.root / name, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != FEATURE_CACHE_FORMAT:
            raise TrainingError("feature cache shard is not loadable: " + name)
        if payload.get("key") != self.key:
            raise TrainingError("feature cache shard key mismatch: " + name)
        features = payload.get("features")
        ids = payload.get("ids")
        if not hasattr(features, "dim") or features.dim() != 3:
            raise TrainingError("feature cache shard holds no [cases, rows, width] matrix: " + name)
        if not isinstance(ids, list) or int(features.shape[0]) != len(ids):
            raise TrainingError("feature cache shard rows do not match its id list: " + name)
        self._open_name = name
        self._open_tensor = features.to(self.torch.float32)
        return self._open_tensor

    def add(self, case_id, feature, observation_tokens):
        shape = (int(feature.shape[0]), int(feature.shape[1]))
        if self.shape is None:
            self.shape = shape
        elif shape != self.shape:
            raise TrainingError("feature shape changed within one run: %s vs %s"
                                % (shape, self.shape))
        self.pending.append((case_id, feature.to(self.torch.float32).contiguous(),
                             int(observation_tokens)))
        self.extracted += 1
        if len(self.pending) >= self.every:
            self.flush()

    # ---- persistence -----------------------------------------------------
    def flush(self):
        """Write one shard, then update the manifest. Both atomically."""
        if not self.pending:
            return None
        name = "shard-%06d.pt" % len(self.shards)
        ids = [case_id for case_id, _, _ in self.pending]
        tokens = [count for _, _, count in self.pending]
        tensor = self.torch.stack([feature for _, feature, _ in self.pending], dim=0)
        atomic_torch_save(self.root / name,
                          {"format": FEATURE_CACHE_FORMAT, "key": self.key, "ids": ids,
                           "observation_tokens": tokens, "features": tensor},
                          self.torch)
        write_json(self.root / (name + ".ids.json"),
                   {"format": FEATURE_CACHE_FORMAT, "key": self.key, "ids": ids,
                    "observation_tokens": tokens})
        number = len(self.shards)
        self.shards.append({"name": name, "ids": ids, "tokens": tokens})
        for row, case_id in enumerate(ids):
            self.index[case_id] = (number, row, tokens[row])
        self.pending = []
        self.write_manifest()
        return name

    def close(self):
        """Flush the tail cases and mark the manifest as a finished pass."""
        self.flush()
        self.write_manifest(closed=True)

    def write_manifest(self, closed=False):
        self.manifest = {"format": FEATURE_CACHE_FORMAT, "cache_key": self.key,
                         "identity": self.identity,
                         "shards": [shard["name"] for shard in self.shards],
                         "case_count": sum(len(shard["ids"]) for shard in self.shards),
                         "feature_shape": list(self.shape) if self.shape else None,
                         "closed": bool(closed), "updated_at": time.time()}
        write_json(self.root / FEATURE_CACHE_MANIFEST, self.manifest)

    def describe(self):
        """Report the cache as three DIFFERENT counts, never one ambiguous one.

        cases_on_disk          rows that are in shards right now (live truth, from
                              the shard list, so it is correct on a first run too)
        cases_reused_at_startup what load() recovered when this run began
        cache_hits             rows served from disk during this run
        newly_extracted        rows this run pushed through the backbone

        Reporting the startup count as "cases_on_disk" read as "nothing was
        written" on a run that had just written every shard, which is exactly the
        wrong thing to believe about durability.
        """
        return {"enabled": True, "root": str(self.root), "cache_key": self.key[:16],
                "shards_on_disk": len(self.shards),
                "cases_on_disk": sum(len(shard["ids"]) for shard in self.shards),
                "cases_reused_at_startup": self.reused_at_startup,
                "cache_hits": self.cache_hits, "newly_extracted": self.extracted,
                "rows_pending_flush": len(self.pending),
                "flush_every": self.every,
                "feature_shape": list(self.shape) if self.shape else None,
                "discarded_reason": self.discarded,
                "normalisation": "raw rows cached; train-only standardisation after assembly"}


# --------------------------------------------------------------------------
# encoding / batching
# --------------------------------------------------------------------------
def encode_sequence(tokenizer, memory_text, ids):
    return [ids["bos"]] + tokenizer.encode(memory_text) + [ids["eos"]]


def pad_batch(sequences, pad_id, torch, device):
    width = max(len(sequence) for sequence in sequences)
    inputs, targets, mask = [], [], []
    for sequence in sequences:
        padded = sequence + [pad_id] * (width - len(sequence))
        inputs.append(padded[:-1])
        targets.append(padded[1:])
        mask.append([False] * (len(sequence) - 1) + [True] * (width - len(sequence)))
    return (torch.tensor(inputs, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device),
            torch.tensor(mask, dtype=torch.bool, device=device))


def make_batches(count, batch_size, seed):
    """Deterministic batch composition: same seed and position -> same batch."""
    order = list(range(count))
    if batch_size <= 1:
        return [[index] for index in order]
    rng = random.Random(seed)
    batches = []
    for start in range(0, count, batch_size):
        chunk = order[start:start + batch_size]
        rng.shuffle(chunk)
        batches.append(chunk)
    return batches


# --------------------------------------------------------------------------
# scheduled teacher-forcing dropout
# --------------------------------------------------------------------------
def teacher_forcing_dropout_rate(step, start, end, ramp_steps):
    """Linear ramp from `start` at step 0 to `end` at `ramp_steps`, then held.

    Ramping rather than switching on keeps the early steps (where the decoder is
    still learning the four-slot format) from being made unsolvable.
    """
    start, end = float(start), float(end)
    step, ramp_steps = max(0, int(step)), int(ramp_steps)
    if ramp_steps <= 0:
        return end
    fraction = min(1.0, step / float(ramp_steps))
    return start + (end - start) * fraction


def corrupt_teacher_forcing_inputs(inputs, pad_mask, mask_token_id, rate, seed, step, torch):
    """Replace some gold-prefix INPUT tokens with the mask token.

    Why this exists
    ---------------
    Under plain teacher forcing the model always sees the gold prefix, and the
    label text is templated: predicting the label words is easy, and predicting a
    slot value can degrade into copying the previous line. The v3 run reached a
    train loss of 0.23 while producing four-slot-exact text for 0 of 1779 cases,
    because it never had to reconstruct anything from the state.

    Here the TARGETS are untouched -- the model is still scored on the true next
    token -- but it can no longer copy that token out of the prefix. What it needs
    has to come from the conditioning, which is what the ablation was measuring.

    Determinism is required for resume, so the draw is seeded from (seed, step)
    rather than from the global RNG: re-running step N after a pause corrupts
    exactly the same positions. Position 0 (BOS) and padded positions are never
    touched -- the first target must stay predictable, and padding is ignored by
    the loss anyway.
    """
    rate = float(rate)
    if rate == 0.0:
        return inputs, 0
    if not 0.0 < rate < 1.0:
        # A negative rate must NOT be silently read as "off": a bad schedule would
        # then look like a working no-op run instead of a configuration error.
        raise TrainingError("teacher-forcing dropout rate must be in [0,1), got %r" % (rate,))
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed((int(seed) * 1000003 + int(step)) % (2 ** 62))
    draw = torch.rand(inputs.shape, generator=generator, device=inputs.device)
    eligible = ~pad_mask
    eligible[:, 0] = False
    chosen = (draw < rate) & eligible
    replacement = torch.full_like(inputs, int(mask_token_id))
    return torch.where(chosen, replacement, inputs), int(chosen.sum().item())


def generate_text(module, features, tokenizer, ids, max_new_tokens, torch, device):
    """Greedy decode. Deterministic, so resume and ablation comparisons stay valid."""
    tokens = [ids["bos"]]
    generated = []
    module.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inputs = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = module(inputs, features)[0, -1]
            token = int(torch.argmax(logits).item())
            if token in (ids["eos"], EOT_ID) or token >= ids["first"]:
                break  # never emit a special as text
            generated.append(token)
            tokens.append(token)
    return tokenizer.decode(generated)


def sample_indices(count, wanted):
    """Evenly spread sample positions across the WHOLE heldout array.

    `items[:wanted]` looked at one corner of the set: with the cases sorted by
    group then id, the first entries are the embedded human-authored ones, so a
    66-case heldout was only ever judged on its three hardest first cases. Even
    spacing covers every region of the split at no extra cost.
    """
    if count <= 0 or wanted <= 0:
        return []
    if wanted >= count:
        return list(range(count))
    if wanted == 1:
        return [count // 2]
    step = (count - 1) / float(wanted - 1)
    picked = []
    for position in range(wanted):
        index = int(round(position * step))
        if not picked or index != picked[-1]:
            picked.append(index)
    return picked


def evaluate(module, items, features, tokenizer, ids, args, torch, device):
    """Heldout token loss + greedy generation with slot-level reference scoring.

    The teacher-forced loss runs in batches: padding every case to one width and
    pushing the whole split through in a single forward is what turns a large
    heldout set into an OOM instead of a number. Batching keeps the metric
    identical -- cross entropy is aggregated token-weighted (sum of losses over
    the exact number of scored tokens), which is the same value the one-shot
    mean produced, only computed in pieces.
    """
    if not items:
        raise TrainingError("evaluation needs at least one case")
    module.eval()
    sequences = [encode_sequence(tokenizer, DATA.memory_text(case), ids) for case in items]
    batch_size = max(1, int(getattr(args, "eval_batch_size", 16) or 1))
    loss_sum = 0.0
    tokens_scored = 0
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            inputs, targets, mask = pad_batch(chunk, ids["pad"], torch, device)
            batch_features = features[start:start + batch_size].to(device)
            logits = module(inputs, batch_features, mask)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            loss_sum += float(torch.nn.functional.cross_entropy(
                flat_logits, flat_targets, ignore_index=ids["pad"], reduction="sum").item())
            tokens_scored += int((flat_targets != ids["pad"]).sum().item())
            del logits, flat_logits, inputs, targets, mask, batch_features
    if tokens_scored < 1:
        raise TrainingError("evaluation scored no tokens; refusing to report a fake loss")
    token_loss = loss_sum / tokens_scored
    samples = []
    for index in sample_indices(len(items), max(0, args.eval_samples)):
        case = items[index]
        text = generate_text(module, features[index:index + 1].to(device), tokenizer, ids,
                             args.eval_max_tokens, torch, device)
        score = DATA.slot_match(text, DATA.memory_text(case))
        content = DATA.content_slot_match(text, DATA.memory_text(case))
        samples.append({"case": case["id"], "group": case["group"], "generated": text,
                        "expected": DATA.memory_text(case), "slots_exact": score["slots_exact"],
                        "per_slot": score["per_slot"], "content": content,
                        "reference_slot_match": score["all_slots_exact"]})
    filled = [s["slots_exact"] for s in samples]
    content_hits = sum(s["content"]["hits"] for s in samples)
    content_scored = sum(s["content"]["scored"] for s in samples)
    block = {"cases": len(items), "token_loss": round(token_loss, 6),
             "token_perplexity": round(math.exp(min(20.0, token_loss)), 4),
             "tokens_scored": tokens_scored, "eval_batch_size": batch_size,
             "generated_samples": len(samples),
             "mean_slots_exact": round(sum(filled) / len(filled), 4) if filled else None,
             "all_slots_exact_cases": sum(1 for s in samples if s["reference_slot_match"]),
             # Additive, not a replacement: `mean_slots_exact` is a frozen contract and
             # still counts 无-vs-无 as a hit, which is why a 1.0/4 run could look alive.
             # This is the number that cannot be earned by emitting 无 everywhere.
             "content_slots_scored": content_scored,
             "content_slot_accuracy": (round(content_hits / content_scored, 4)
                                       if content_scored else None),
             "samples": samples}
    module.train()
    return block


def compact_eval_block(block, keep_expected):
    """The part of an evaluation block worth keeping in the history file."""
    summary = {key: block.get(key) for key in
               ("cases", "token_loss", "token_perplexity", "tokens_scored",
                "eval_batch_size", "generated_samples", "mean_slots_exact",
                "all_slots_exact_cases", "content_slots_scored", "content_slot_accuracy")}
    samples = []
    for sample in block.get("samples", []):
        entry = {key: sample[key] for key in ("case", "group", "generated", "slots_exact",
                                              "per_slot", "content", "reference_slot_match")
                 if key in sample}
        if keep_expected:
            entry["expected"] = sample.get("expected")
        samples.append(entry)
    summary["samples"] = samples
    return summary


def append_eval_history(path, step, block, phase):
    """Append one JSON object per evaluation: step + the three state conditions.

    Written as a single line with an fsync, appended rather than rewritten, so a
    long run keeps a full evaluation curve instead of only the last block. A torn
    final line is possible if the machine dies mid-append; readers must skip
    unparsable lines rather than trusting the last one.
    """
    conditions = {"true": compact_eval_block(block, keep_expected=True)}
    for name, entry in sorted((block.get("ablations") or {}).items()):
        conditions[name] = compact_eval_block(entry, keep_expected=False)
    record = {"step": int(step), "phase": phase, "saved_at": time.time(),
              "state_mode": block.get("state_mode"),
              "conditions": conditions}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Decoder-only PILOT: frozen RWKV-7 recurrent state -> readable memory "
                    "text. This is not RWKV fine-tuning.")
    parser.add_argument("--steps", type=int, default=300,
                        help="total optimizer steps to reach (cumulative across resumes)")
    parser.add_argument("--resume", default=str(DEFAULT_OUT / RESUME_NAME),
                        help="checkpoint to resume from; also the rolling save target")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model", default=os.environ.get("DBA_RWKV_MODEL", DEFAULT_MODEL))
    parser.add_argument("--state-mode", default="true", choices=STATE_MODES,
                        help="ablation: true | none (zero conditioning) | shuffled (permuted)")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-self-layers", type=int, default=2)
    parser.add_argument("--n-cross-layers", type=int, default=2)
    parser.add_argument("--conditioning-tokens", type=int, default=8,
                        help="size of the state bottleneck fed to cross-attention")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--conditioning", default=DEFAULT_CONDITIONING_MODE,
                        choices=CONDITIONING_MODES,
                        help="cross = legacy cross-attention only (default, keeps old "
                             "checkpoints resumable); film = cross-attention PLUS a "
                             "zero-initialised FiLM modulation after every decoder block")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay; legacy default is 0.01")
    parser.add_argument("--tf-dropout-start", type=float, default=0.0,
                        help="fraction of gold-prefix INPUT tokens replaced by the mask token "
                             "at step 0 (0 disables teacher-forcing dropout)")
    parser.add_argument("--tf-dropout-end", type=float, default=0.0,
                        help="same fraction once the ramp finishes (targets are never changed)")
    parser.add_argument("--tf-dropout-ramp", type=int, default=1000,
                        help="steps over which the teacher-forcing dropout ramps start -> end")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--heldout-every", type=int, default=100,
                        help="0 disables periodic heldout evaluation")
    parser.add_argument("--eval-samples", type=int, default=3,
                        help="generations per evaluation, spread across the whole heldout set")
    parser.add_argument("--eval-max-tokens", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="heldout cases per teacher-forced forward (keeps a large split off OOM)")
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--feature-cache", dest="feature_cache", action="store_true", default=True,
                        help="reuse GPU-extracted state features across pauses (default: on)")
    parser.add_argument("--no-feature-cache", dest="feature_cache", action="store_false",
                        help="ignore and do not write the per-case feature cache")
    parser.add_argument("--feature-cache-every", type=int, default=FEATURE_CACHE_EVERY_DEFAULT,
                        help="cases per atomically written cache shard")
    parser.add_argument("--feature-cache-root", default=None,
                        help="read/write the durable feature cache here instead of --out-dir; "
                             "point a new run at an earlier run to reuse its already-extracted "
                             "rows (the cache key must still match, and a miss is refused "
                             "instead of silently re-extracting every case)")
    parser.add_argument("--cases", default=None, help="extra JSONL cases file")
    parser.add_argument("--max-steps-guard", type=int, default=200000)
    parser.add_argument("--dry-run", action="store_true",
                        help="synthetic features + byte tokenizer: no backbone, GPU still used")
    parser.add_argument("--skip-power-check", action="store_true",
                        help="SYNTHETIC PLUMBING ONLY (requires --dry-run): disables AC/guard gates")
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise TrainingError("--steps must be positive")
    if args.steps > args.max_steps_guard:
        raise TrainingError("--steps exceeds --max-steps-guard")
    for name in ("batch_size", "d_model", "n_heads", "eval_max_tokens", "chunk_tokens",
                 "conditioning_tokens", "eval_batch_size", "feature_cache_every"):
        if getattr(args, name) < 1:
            raise TrainingError("--%s must be positive" % name.replace("_", "-"))
    if args.eval_samples < 0:
        raise TrainingError("--eval-samples cannot be negative")
    if args.checkpoint_every < 1 or args.log_every < 0 or args.heldout_every < 0:
        raise TrainingError("--checkpoint-every must be positive; --log-every/--heldout-every >= 0")
    for name in ("tf_dropout_start", "tf_dropout_end"):
        if not 0.0 <= getattr(args, name) < 1.0:
            raise TrainingError("--%s must be in [0,1)" % name.replace("_", "-"))
    if args.tf_dropout_ramp < 0:
        raise TrainingError("--tf-dropout-ramp cannot be negative")
    if args.weight_decay < 0.0:
        raise TrainingError("--weight-decay cannot be negative")
    if (args.tf_dropout_start or args.tf_dropout_end) and args.tf_dropout_ramp < 1:
        raise TrainingError("--tf-dropout-ramp must be >= 1 when teacher-forcing dropout is on")
    if args.skip_power_check and not args.dry_run:
        raise TrainingError("--skip-power-check is only allowed together with --dry-run")
    return args


# --------------------------------------------------------------------------
# training run
# --------------------------------------------------------------------------
def run(args, hooks=None):
    """One training run. `hooks` are test seams: torch/engine/tokenizer/keep_backbone."""
    try:
        return _run_inner(args, hooks)
    except Done as done:
        return done.result


def _run_inner(args, hooks=None):
    hooks = hooks or {}
    log_lines = []
    force_lease = bool(getattr(args, "skip_power_check", False))

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(args.resume)
    pointer = out_dir / POINTER_NAME
    status_path = out_dir / "status.json"
    log_path = out_dir / "train.log"
    eval_history_path = out_dir / EVAL_HISTORY_NAME

    def power_loss_result(stage, report):
        """No lease or no AC: record it and stop without touching the GPU."""
        result = {"ok": True, "paused": True, "exit_code": EXIT_PAUSED, "phase": "paused",
                  "stage": stage, "power": report, "label": LABEL,
                  "backbone_training": BACKBONE_TRAINING, "state_mode": args.state_mode,
                  "steps_requested": args.steps, "artifacts": {"status": str(status_path),
                                                              "log": str(log_path)}}
        status_update(phase="paused", paused=True, paused_before=stage, power=report)
        log("power gate stopped the run at %s: %s" % (stage, json.dumps(report, ensure_ascii=False)))
        flush_log()
        return result

    def gate(stage):
        if force_lease:
            return {"stage": stage, "guard_lease": None, "ac_connected": None,
                    "power_gate": "disabled-by---skip-power-check"}
        try:
            return power_report(stage, require_lease=True)
        except PowerLoss as exc:
            raise Done(power_loss_result(stage, exc.report)) from None

    result = {"ok": False, "label": LABEL, "backbone_training": BACKBONE_TRAINING,
              "state_mode": args.state_mode, "steps_requested": args.steps}
    status = {}

    def status_update(**changes):
        status.update(changes)
        status.update({"label": LABEL, "backbone_training": BACKBONE_TRAINING,
                       "state_mode": args.state_mode, "steps_requested": args.steps,
                       "updated_at": time.time()})
        write_json(status_path, status)

    # ---- gate 1: before anything touches CUDA -----------------------------
    result["power_before_cuda"] = gate("before-cuda")
    log("power gate before CUDA: " + json.dumps(result["power_before_cuda"], ensure_ascii=False))

    torch = hooks.get("torch") or __import__("torch")
    if not torch.cuda.is_available():
        raise TrainingError("CUDA is required; this pilot refuses CPU training")
    if not torch.cuda.is_bf16_supported():
        raise TrainingError("CUDA BF16 is required")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    items = DATA.all_cases(args.cases)
    train_items = [case for case in items if case["group"] == DATA.TRAIN]
    heldout_items = [case for case in items if case["group"] == DATA.HELDOUT]
    if not train_items or not heldout_items:
        raise TrainingError("the pilot needs both train and heldout cases")
    data_hash = DATA.data_version(items)
    log("cases %s data_hash=%s" % (json.dumps(DATA.split_counts(items)), data_hash[:16]))

    # ---- tokenizer / backbone handle --------------------------------------
    engine = hooks.get("engine")
    if args.dry_run:
        tokenizer = hooks.get("tokenizer") or ByteTokenizer()
        backbone_report = {"mode": "synthetic", "reason": "--dry-run: no backbone was loaded"}
        n_layer = int(hooks.get("synthetic_layers", 12))
        synthetic_width = int(hooks.get("synthetic_width", 128))
    else:
        from nb_rwkv import TorchEngine
        engine = engine or TorchEngine(args.model, chunk_tokens=args.chunk_tokens,
                                       generation_tokens=1)
        pipeline = engine.pipeline
        table = getattr(pipeline.tokenizer, 'idx2token', {})
        vocab_size = max(table.keys()) + 1 if table else 0
        if not vocab_size:
            raise TrainingError("RWKV pipeline exposes no token table")
        tokenizer = BackboneTokenizer(pipeline, vocab_size)
        backbone_report = {"mode": "frozen-state", "identity": engine.identity,
                           "weights_path": str(engine.path)}
        n_layer = int(engine.model.args.n_layer)
        synthetic_width = None
    ids = special_ids(tokenizer)
    widest = max(max(tokenizer.encode(DATA.memory_text(case))) for case in items)
    if widest >= ids["first"]:
        raise TrainingError("special ids collide with the text vocabulary (max real id %d)" % widest)
    log("tokenizer=%s vocab=%d specials=%s" % (tokenizer.name, tokenizer.vocab_size,
                                               json.dumps(ids, sort_keys=True)))

    # ---- feature extraction (long: gate before, during and after) ---------
    #
    # Cache-backed: rows already extracted by an earlier (paused) run of the same
    # backbone, features and dataset are loaded from disk instead of being pushed
    # through the frozen 1.5B backbone again. Rows are cached RAW; standardisation
    # below still uses train-split statistics only.
    gate("before-feature-extraction")
    extraction = {"backbone_mode": backbone_report["mode"], "cases": len(items),
                  "n_layer": n_layer, "feature_version": FEATURE_VERSION,
                  "per_case_observation_tokens": {}}
    cache = None
    if args.feature_cache:
        cache_identity = feature_cache_identity(backbone_report, data_hash, n_layer,
                                                synthetic_width)
        cache_key = feature_cache_key(cache_identity)
        cache = FeatureCache(resolve_feature_cache_dir(args.feature_cache_root, out_dir,
                                                       cache_key[:16]),
                             cache_key, cache_identity, torch,
                             every=args.feature_cache_every)
        cached_ids = cache.load()
        if cache.discarded:
            log("feature cache at %s discarded: %s" % (cache.root, cache.discarded))
        if cached_ids:
            log("feature cache: %d/%d cases reused from %s"
                % (len(cached_ids), len(items), cache.root))
            for case_id in cached_ids:
                entry = cache.index.get(case_id)
                if entry is not None:
                    extraction["per_case_observation_tokens"][case_id] = entry[2]
        extraction["feature_cache"] = cache.describe()
    else:
        extraction["feature_cache"] = {"enabled": False, "reason": "--no-feature-cache"}

    def release_backbone():
        nonlocal engine
        if engine is None:
            return False
        if not args.dry_run:
            try:
                engine.model = None
                engine.pipeline = None
            except Exception:
                pass
        engine = None
        gc.collect()
        torch.cuda.empty_cache()
        return True

    if args.dry_run:
        extraction["synthetic"] = True
    started = time.perf_counter()
    shape = cache.shape if cache is not None else None
    matrix = (torch.empty((len(items), shape[0], shape[1]), dtype=torch.float32)
              if shape is not None else None)
    newly_extracted = 0
    paused_during_extraction = False
    try:
        for index, case in enumerate(items):
            case_id = case["id"]
            row, tokens = (None, None)
            if cache is not None:
                row, tokens = cache.lookup(case_id)
            if row is None:
                # The long GPU part: gate every few cases so an unplugged machine
                # stops here instead of burning hours of backbone time.
                if newly_extracted and newly_extracted % 5 == 0:
                    gate("during-feature-extraction")
                if args.dry_run:
                    seed = int.from_bytes(hashlib.sha256(
                        DATA.source_text(case).encode("utf-8")).digest()[:8], "big") % (2 ** 31)
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    row = torch.randn(n_layer * 3, synthetic_width, generator=generator)
                    tokens = 0
                else:
                    state, tokens = engine.observe(None, DATA.observation_text(case))
                    row = state_features(state, n_layer, torch).to("cpu")
                    del state
                newly_extracted += 1
                if cache is not None:
                    cache.add(case_id, row, tokens)
            row = row.to(torch.float32)
            if matrix is None:
                shape = (int(row.shape[0]), int(row.shape[1]))
                matrix = torch.empty((len(items), shape[0], shape[1]), dtype=torch.float32)
            if (int(row.shape[0]), int(row.shape[1])) != shape:
                raise TrainingError("inconsistent state feature shapes: %s vs %s"
                                    % ((int(row.shape[0]), int(row.shape[1])), shape))
            matrix[index] = row
            del row
            extraction["per_case_observation_tokens"][case_id] = tokens
            if (index + 1) % FEATURE_PROGRESS_EVERY == 0 or index + 1 == len(items):
                log("feature progress %d/%d (reused=%d extracted=%d)"
                    % (index + 1, len(items),
                       cache.cache_hits if cache is not None else 0, newly_extracted))
    except Done:
        # An AC pause is raised from `gate()` inside the loop. Record it so the
        # `finally` below knows to publish the cache state to status.json.
        paused_during_extraction = True
        raise
    finally:
        # Flush rows still held in memory before the process can exit. Without
        # this, a pause throws away up to `--feature-cache-every` already
        # extracted rows (no shard, no manifest entry, so not even recoverable).
        # Rows are still LOST if the process is killed outright: only a written
        # shard is durable, so the hard bound stays `--feature-cache-every`
        # (default 100) cases of backbone work.
        try:
            if cache is not None:
                pending = len(cache.pending)
                try:
                    cache.flush()
                    on_exit = {"rows_flushed": pending, "rows_lost": 0}
                except Exception as exc:
                    on_exit = {"rows_flushed": 0, "rows_lost": pending, "error": str(exc)}
                extraction["seconds"] = round(time.perf_counter() - started, 3)
                extraction["feature_cache_on_exit"] = on_exit
                extraction["feature_cache"] = cache.describe()
                if pending:
                    log("feature cache: %d pending rows flushed on exit (lost=%d)"
                        % (on_exit["rows_flushed"], on_exit["rows_lost"]))
                if paused_during_extraction:
                    # The pause path already wrote status.json; this adds the
                    # extraction/cache state instead of leaving it unrecorded.
                    status_update(feature_extraction=extraction)
        except Exception as exc:  # never mask the pause or the original error
            log("feature cache: exit flush failed: %s" % exc)
        try:
            flush_log()
        except Exception:
            pass
    if cache is not None:
        cache.close()
        extraction["feature_cache"] = cache.describe()
    extraction["seconds"] = round(time.perf_counter() - started, 3)
    if not args.dry_run:
        extraction["backbone_identity"] = engine.identity
    gate("after-feature-extraction")

    if matrix is None:
        raise TrainingError("no state features were produced")
    feature_count, feature_dim = shape
    train_count = len(train_items)
    mean = matrix[:train_count].mean(dim=(0, 1), keepdim=True)
    std = matrix[:train_count].std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
    matrix = (matrix - mean) / std
    feature_hash = hashlib.sha256(matrix.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()[:32]
    extraction.update({"feature_shape": [int(feature_count), int(feature_dim)],
                       "standardised_with": "train-split statistics only",
                       "feature_hash": feature_hash,
                       "reused_from_cache": cache.cache_hits if cache is not None else 0,
                       "newly_extracted": newly_extracted})

    # ---- ablation transforms ---------------------------------------------
    if args.state_mode == "none":
        matrix = torch.zeros_like(matrix)
    elif args.state_mode == "shuffled":
        permutation = torch.randperm(matrix.shape[0],
                                     generator=torch.Generator().manual_seed(args.seed + 7))
        matrix = matrix[permutation]
    if not hooks.get("keep_backbone"):
        extraction["freed_backbone"] = release_backbone()
        if extraction["freed_backbone"]:
            log("freed the frozen backbone after feature extraction")

    # ---- model ------------------------------------------------------------
    config = DecoderConfig(feature_count, feature_dim, tokenizer.vocab_size,
                           d_model=args.d_model, n_heads=args.n_heads,
                           n_self_layers=args.n_self_layers, n_cross_layers=args.n_cross_layers,
                           dropout=args.dropout, conditioning_tokens=args.conditioning_tokens,
                           conditioning_mode=args.conditioning)
    module = build_decoder(config, torch).to(device)
    module.train()
    parameters = decoder_parameter_count(module)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sequences = [encode_sequence(tokenizer, DATA.memory_text(case), ids) for case in items]
    max_length = max(len(sequence) for sequence in sequences)
    train_batches = make_batches(train_count, args.batch_size, args.seed)

    run_identity = {
        "script": SCRIPT_VERSION, "format": FORMAT, "label": LABEL,
        "backbone_training": BACKBONE_TRAINING, "state_mode": args.state_mode,
        "backbone": backbone_report.get("identity", {"mode": backbone_report["mode"]}),
        "backbone_mode": backbone_report["mode"], "feature_version": FEATURE_VERSION,
        "feature_hash": feature_hash, "data_hash": data_hash,
        "data_counts": DATA.split_counts(items), "decoder": config.to_dict(),
        "tokenizer": {"name": tokenizer.name, "vocab_size": tokenizer.vocab_size},
        "seed": args.seed, "batch_size": args.batch_size, "lr": args.lr,
        "max_length": max_length, "case_ids": [case["id"] for case in items],
    }
    # Training-shape options are recorded only when they differ from their legacy
    # defaults, so a legacy run keeps the identity hash (and the checkpoint) it had
    # before these flags existed, while a run that used them can never be silently
    # resumed as if it had not.
    training_options = {}
    if args.tf_dropout_start or args.tf_dropout_end:
        training_options["tf_dropout"] = {"start": args.tf_dropout_start,
                                          "end": args.tf_dropout_end,
                                          "ramp_steps": args.tf_dropout_ramp}
    if args.weight_decay != 0.01:
        training_options["weight_decay"] = args.weight_decay
    if training_options:
        run_identity["training_options"] = training_options
    identity_hash = hashlib.sha256(json.dumps(run_identity, ensure_ascii=False,
                                              sort_keys=True).encode("utf-8")).hexdigest()
    log("decoder parameters=%d conditioning=%dtokens dim=%d max_length=%d identity=%s"
        % (parameters, config.conditioning_tokens, config.conditioning_dim, max_length,
           identity_hash[:16]))
    log("conditioning=%s modulation_layers=%d weight_decay=%s tf_dropout=%s..%s over %d steps"
        % (config.conditioning_mode, config.modulation_layers, args.weight_decay,
           args.tf_dropout_start, args.tf_dropout_end, args.tf_dropout_ramp))

    status.update({"run_identity": run_identity, "identity_hash": identity_hash,
                   "decoder_parameters": parameters, "device": str(device),
                   "gpu": torch.cuda.get_device_name(0), "feature_extraction": extraction,
                   "artifact_paths": {"log": str(log_path), "status": str(status_path),
                                      "checkpoint": str(resume_path), "pointer": str(pointer),
                                      "eval_history": str(eval_history_path)}})

    # ---- resume -----------------------------------------------------------
    start_step = 0
    history = []
    resumed_from = None
    if resume_path.is_file():
        checkpoint = load_checkpoint(resume_path, torch)
        if checkpoint["identity_hash"] != identity_hash:
            raise TrainingError(
                "checkpoint identity mismatch (%s vs %s): refusing to resume across a "
                "different backbone, dataset, ablation or decoder config"
                % (str(checkpoint["identity_hash"])[:16], identity_hash[:16]))
        start_step = int(checkpoint["step"])
        if start_step >= args.steps:
            log("checkpoint already at step %d >= --steps %d; nothing to do"
                % (start_step, args.steps))
            status_update(step=start_step, phase="already-complete",
                          resumed_from=str(resume_path), history=checkpoint.get("history", []))
            result.update({"ok": True, "step": start_step, "phase": "already-complete",
                           "resumed_from": str(resume_path), "identity_hash": identity_hash,
                           "checkpoint": str(resume_path), "artifacts": status["artifact_paths"]})
            return result
        module.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint["rng"], torch)
        history = list(checkpoint.get("history", []))[-HISTORY_LIMIT:]
        resumed_from = {"path": str(resume_path), "step": start_step,
                        "identity_hash": checkpoint["identity_hash"][:16],
                        "saved_at": checkpoint.get("saved_at")}
        log("resumed from %s at step %d" % (resume_path, start_step))
    current = {"step": start_step,
               "loss": float(history[-1]["loss"]) if history else None}

    # ---- checkpoint / pause ----------------------------------------------
    def checkpoint_payload(step, eval_block, phase):
        return {"format": FORMAT, "identity_hash": identity_hash, "run_identity": run_identity,
                "step": step, "phase": phase, "model_state": module.state_dict(),
                "optimizer_state": cpu_optimizer_state(optimizer),
                "history": history[-HISTORY_LIMIT:], "loss": current["loss"],
                "heldout": eval_block, "rng": rng_state(torch), "saved_at": time.time(),
                "device": str(device), "decoder_parameters": parameters,
                "state_mode": args.state_mode, "resumed_from": resumed_from,
                "label": LABEL, "backbone_training": BACKBONE_TRAINING}

    def save_at(path, step, phase, eval_block=None):
        return save_checkpoint(path, checkpoint_payload(step, eval_block, phase), torch,
                               pointer=pointer)

    def pause_and_exit():
        target = out_dir / PAUSE_NAME
        save_at(target, current["step"], "paused")
        status_update(step=current["step"], phase="paused", last_loss=current["loss"],
                      paused=True, paused_checkpoint=str(target), history=history[-HISTORY_LIMIT:])
        log("AC power lost: saved %s at step %d, exiting %d" % (target, current["step"], EXIT_PAUSED))
        flush_log()
        return {"ok": True, "paused": True, "step": current["step"], "phase": "paused",
                "checkpoint": str(target), "exit_code": EXIT_PAUSED,
                "identity_hash": identity_hash, "artifacts": status["artifact_paths"]}

    # ---- warm-up forward: fail before spending steps ----------------------
    warm_inputs, warm_targets, warm_mask = pad_batch([sequences[0]], ids["pad"], torch, device)
    warm_logits = module(warm_inputs, matrix[:1].to(device), warm_mask)
    initial_loss = float(torch.nn.functional.cross_entropy(
        warm_logits.reshape(-1, warm_logits.shape[-1]), warm_targets.reshape(-1),
        ignore_index=ids["pad"]).item())
    log("warm-up forward ok: first-case loss %.4f (uniform ln(vocab)=%.4f)"
        % (initial_loss, math.log(config.vocab_size)))
    del warm_logits, warm_inputs, warm_targets, warm_mask
    torch.cuda.empty_cache()

    train_features = matrix[:train_count]
    heldout_features = matrix[train_count:]

    def run_eval(step, phase="periodic"):
        """Reference condition plus the two state ablations, all three recorded.

        no_state  : all-zero conditioning -> how good can the label priors alone be
        wrong_state: real states rolled by one case -> conditioning present but wrong
        The pair is what separates "reads the state" from "used some vector".
        """
        block = evaluate(module, heldout_items, heldout_features, tokenizer, ids, args, torch, device)
        block["state_mode"] = args.state_mode
        block["ablations"] = {}
        for name, altered in (("no_state", torch.zeros_like(heldout_features)),
                              ("wrong_state", heldout_features.roll(1, dims=0))):
            comparison = evaluate(module, heldout_items, altered, tokenizer, ids, args, torch, device)
            comparison["state_mode"] = args.state_mode
            block["ablations"][name] = comparison
        record = append_eval_history(eval_history_path, step, block, phase)
        status["eval_records"] = int(status.get("eval_records", 0)) + 1
        return block, record

    baseline, _ = run_eval(start_step, "baseline")
    log("heldout reference @%d: token_loss=%.4f mean_slots_exact=%s content_slot_accuracy=%s "
        "(ablations %s)"
        % (start_step, baseline["token_loss"], baseline["mean_slots_exact"],
           baseline["content_slot_accuracy"],
           json.dumps({name: entry["token_loss"]
                       for name, entry in baseline["ablations"].items()})))
    status_update(step=start_step, phase="training", initial_loss=initial_loss,
                  heldout_baseline={k: baseline[k] for k in
                                    ("cases", "token_loss", "token_perplexity", "tokens_scored",
                                     "mean_slots_exact", "all_slots_exact_cases",
                                     "content_slots_scored", "content_slot_accuracy")},
                  history=history[-HISTORY_LIMIT:], resumed_from=resumed_from)

    # ---- training loop ----------------------------------------------------
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    step = start_step

    while step < args.steps:
        try:
            if not force_lease:
                pause_if_requested(lambda: save_at(out_dir / PAUSE_NAME, current["step"], "paused"))
        except SystemExit as exc:
            if exc.code == EXIT_PAUSED:
                raise Done(pause_and_exit()) from None
            raise
        if not force_lease:
            try:
                power_report("before-optimizer-step", require_lease=True)
            except PowerLoss as exc:
                status["power"] = exc.report
                raise Done(pause_and_exit()) from None
        batch = train_batches[step % len(train_batches)]
        inputs, targets, mask = pad_batch([sequences[index] for index in batch], ids["pad"],
                                          torch, device)
        batch_features = train_features[batch].to(device)
        # Scheduled teacher-forcing dropout: hide part of the gold prefix so the value
        # tokens cannot be copied and have to come from the conditioning. Targets stay
        # the ground truth, and evaluation never corrupts anything.
        tf_rate = teacher_forcing_dropout_rate(step, args.tf_dropout_start,
                                               args.tf_dropout_end, args.tf_dropout_ramp)
        model_inputs, corrupted = corrupt_teacher_forcing_inputs(
            inputs, mask, ids["bos"], tf_rate, args.seed, step, torch)
        logits = module(model_inputs, batch_features, mask)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=ids["pad"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0).item())
        optimizer.step()
        step += 1
        current.update({"step": step, "loss": float(loss.item()), "tf_rate": tf_rate})
        del logits, inputs, targets, mask, batch_features, model_inputs, loss
        record = {"step": step, "loss": round(current["loss"], 6),
                  "perplexity": round(math.exp(min(20.0, current["loss"])), 4),
                  "grad_norm": round(grad_norm, 4)}
        if tf_rate > 0.0:
            record.update({"tf_rate": round(tf_rate, 4), "corrupted_tokens": corrupted})
        history.append(dict(record))
        if args.log_every and (step % args.log_every == 0 or step == args.steps):
            elapsed = time.perf_counter() - started
            per_step = elapsed / max(1, step - start_step)
            peak = round(torch.cuda.max_memory_allocated(0) / 1024 ** 2, 1)
            record.update({"seconds": round(elapsed, 3), "seconds_per_step": round(per_step, 4),
                           "vram_mib": peak})
            history[-1] = dict(record)
            log("step %d/%d loss=%.4f ppl=%.2f grad=%.3f %.3fs/step peak=%.1fMiB%s"
                % (step, args.steps, current["loss"], record["perplexity"], grad_norm,
                   per_step, peak,
                   (" tf=%.3f hidden=%d" % (record["tf_rate"], record["corrupted_tokens"]))
                   if "tf_rate" in record else ""))
            status_update(step=step, phase="training", last=record, history=history[-HISTORY_LIMIT:],
                          step_seconds=round(per_step, 4),
                          eta_seconds=round(per_step * (args.steps - step), 1),
                          peak_vram_mib=peak)
        if args.heldout_every and step % args.heldout_every == 0 and step != args.steps:
            block, _ = run_eval(step, "periodic")
            log("heldout @%d: token_loss=%.4f mean_slots_exact=%s content_slot_accuracy=%s "
                "ablations=%s"
                % (step, block["token_loss"], block["mean_slots_exact"],
                   block["content_slot_accuracy"],
                   json.dumps({name: entry["token_loss"]
                               for name, entry in block["ablations"].items()})))
            status_update(step=step, phase="training",
                          heldout={k: block[k] for k in
                                   ("cases", "token_loss", "token_perplexity", "tokens_scored",
                                    "mean_slots_exact", "all_slots_exact_cases",
                                    "content_slots_scored", "content_slot_accuracy")},
                          heldout_ablations={name: entry["token_loss"]
                                             for name, entry in block["ablations"].items()},
                          history=history[-HISTORY_LIMIT:])
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_at(resume_path, step, "training")
            status_update(step=step, phase="training", checkpoint=str(resume_path),
                          history=history[-HISTORY_LIMIT:])
            flush_log()

    final_eval, _ = run_eval(step, "final")
    save_at(resume_path, step, "complete", final_eval)
    peak = round(torch.cuda.max_memory_allocated(0) / 1024 ** 2, 1)
    status_update(step=step, phase="complete", final_heldout=final_eval, paused=False,
                  final_ablations={name: {k: entry[k] for k in
                                          ("cases", "token_loss", "tokens_scored",
                                           "mean_slots_exact", "all_slots_exact_cases",
                                           "content_slots_scored", "content_slot_accuracy")}
                                   for name, entry in final_eval["ablations"].items()},
                  duration_seconds=round(time.perf_counter() - started, 3),
                  peak_vram_mib=peak, history=history[-HISTORY_LIMIT:],
                  checkpoint=str(resume_path))
    log("finished step %d: heldout token_loss=%.4f mean_slots_exact=%s content_slot_accuracy=%s "
        "ablations=%s"
        % (step, final_eval["token_loss"], final_eval["mean_slots_exact"],
           final_eval["content_slot_accuracy"],
           json.dumps({name: entry["token_loss"]
                       for name, entry in final_eval["ablations"].items()})))
    flush_log()
    result.update({"ok": True, "step": step, "phase": "complete", "label": LABEL,
                   "backbone_training": BACKBONE_TRAINING, "initial_loss": initial_loss,
                   "final_loss": current["loss"], "decoder_parameters": parameters,
                   "identity_hash": identity_hash, "checkpoint": str(resume_path),
                   "peak_vram_mib": peak, "feature_extraction": extraction,
                   "heldout": {k: final_eval[k] for k in
                               ("cases", "token_loss", "token_perplexity", "tokens_scored",
                                "mean_slots_exact", "all_slots_exact_cases", "samples")},
                   "heldout_ablations": {name: {k: entry[k] for k in
                                                ("cases", "token_loss", "tokens_scored",
                                                 "mean_slots_exact", "all_slots_exact_cases")}
                                         for name, entry in final_eval["ablations"].items()},
                   "history": history[-HISTORY_LIMIT:],
                   "artifacts": status["artifact_paths"]})
    return result


def main(argv=None):
    try:
        args = parse_args(argv)
        result = run(args)
    except TrainingError as exc:
        print(json.dumps({"ok": False, "label": LABEL, "error": str(exc)}, ensure_ascii=False),
              flush=True)
        return 2
    except SystemExit as exc:
        if exc.code == EXIT_PAUSED:
            print(json.dumps({"ok": True, "paused": True, "exit_code": EXIT_PAUSED},
                             ensure_ascii=False), flush=True)
            return EXIT_PAUSED
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return EXIT_PAUSED if result.get("paused") else 0


if __name__ == "__main__":
    raise SystemExit(main())
