"""What is the achievable ceiling for the constraint slot on FROZEN features?

Evidence so far, all measured on the same dev split:
  * the pair model ends with `triple` loss near zero while 64% of predicted constraints
    are EMPTY (only 32/1773 references are empty) -- the head took the all-negative
    solution;
  * a threshold sweep on the frozen checkpoint cannot fix it: 0.5 and 0.2 both give 61
    exact cases, and every lower threshold is worse;
  * `oracle_topk_recall` is 0.3159 -- handed the exact number of top-scoring triples,
    the model still recovers under a third of the gold positives, so the threshold is
    not the problem;
  * independently trained per-triple LINEAR probes on these same pooled features reach
    dev AUC 0.9788 mean / 0.9863 median, so the information is present and linearly
    readable.

Those facts leave one question, and everything else depends on its answer: is the pair
model's shared encoder to blame, or is the exact-SET metric itself unreachable at this
label density? Constraint exact-match over 931 classes requires zero false positives
across all ~929 negatives while catching every positive. That is a far harsher bar than
per-class AUC, and it may simply not be reachable.

This script fits a PLAIN LINEAR multi-label head (4096 -> 931) on the same frozen pooled
features with the same weighted BCE, and sweeps its threshold. A linear head is the
weakest useful reader, so what it reaches is a fair LOWER bound on what any head on these
features can reach.

Reading the result:
  * linear constraint-exact clearly above the pair model's ~4%  -> the encoder/optimiser
    is the bottleneck and another architecture can win;
  * linear constraint-exact in the same 3-6% band             -> the metric, not the
    model, is the wall; report the ceiling and stop chasing four-slot exact.

The decision threshold is chosen on a held-out slice of TRAIN, never on dev, so the dev
number is not quietly tuned. CPU or CUDA; prints JSON; writes nothing but the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_rwkv_slot_pairs as P  # noqa: E402


def constraint_exact_at(mask, references, vocab):
    exact = 0
    for row in range(mask.shape[0]):
        predicted = frozenset(
            vocab.triples.decode(position)
            for position in mask[row].nonzero(as_tuple=False).flatten().tolist())
        if predicted == references[row]["constraint"]:
            exact += 1
    return exact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--cache-identity", required=True,
                        help="data_hash the cache was extracted for; the cache refuses "
                             "to be read without it")
    parser.add_argument("--pool", default="meanmax")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-clamp", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    parser.add_argument("--thresholds", default="0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1")
    args = parser.parse_args(argv)

    torch = P.base._torch()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no CUDA available; pass --device cpu to run this on CPU")

    corpus = P.load_corpus(args.cases)
    train_cases, dev_cases = P.split_corpus(corpus)
    train_pairs = P.pair_lists(train_cases)
    dev_pairs = P.pair_lists(dev_cases)
    vocab = P.build_vocabularies(train_pairs)

    cache = P.ReadOnlyFeatureCache(Path(args.feature_cache), args.cache_identity, None,
                                   allow_any_backbone=True)
    cache.open()
    ordered = train_cases + dev_cases
    cache.require_all([case["id"] for case in ordered])
    pooled = P.build_pooled(cache, ordered, args.pool)
    train_count = len(train_cases)
    mean = pooled[:train_count].mean(dim=0, keepdim=True)
    std = pooled[:train_count].std(dim=0, keepdim=True).clamp_min(1e-5)
    features = (pooled - mean) / std

    # Threshold selection needs data the fit never saw, but it must not be dev or the
    # reported dev number would be tuned. Carve the calibration slice out of TRAIN.
    calibrate_count = max(1, train_count // 5)
    fit_count = train_count - calibrate_count
    x_fit = features[:fit_count].to(args.device)
    x_calib = features[fit_count:train_count].to(args.device)
    x_dev = features[train_count:].to(args.device)

    all_targets = P.encode_targets(train_pairs, vocab, torch).multi
    targets_fit = all_targets[:fit_count].to(args.device)
    targets_calib = all_targets[fit_count:train_count].to(args.device)
    calibrate_pairs = train_pairs[fit_count:train_count]

    # Reuse the run's own weighting so this differs from it in exactly one way: the head.
    pos_weight = None
    if args.pos_weight_power > 0:
        pos_weight = P.triple_positive_weight(all_targets, torch,
                                              power=args.pos_weight_power,
                                              clamp=args.pos_weight_clamp).to(args.device)

    torch.manual_seed(20261008)
    weight = torch.zeros(x_fit.shape[1], len(vocab.triples),
                         device=args.device).normal_(0, 0.01).requires_grad_(True)
    bias = torch.zeros(len(vocab.triples), device=args.device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(20261008)

    for step in range(args.steps):
        index = torch.randint(0, x_fit.shape[0], (args.batch_size,), generator=generator)
        optimizer.zero_grad()
        logits = x_fit[index].mm(weight) + bias
        loss = loss_fn(logits, targets_fit[index])
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            print("step %d/%d loss=%.6f" % (step + 1, args.steps, float(loss)), flush=True)

    with torch.no_grad():
        calib_logits = (x_calib.mm(weight) + bias).float().cpu()
        dev_logits = (x_dev.mm(weight) + bias).float().cpu()

    # Pick the threshold on TRAIN calibration only.
    best = None
    calibration = []
    for raw in args.thresholds.split(","):
        threshold = float(raw)
        mask = (torch.sigmoid(calib_logits) >= threshold)
        exact = constraint_exact_at(mask, calibrate_pairs, vocab)
        row = {"threshold": threshold, "calibration_exact": exact,
               "calibration_rate": exact / calibrate_count,
               "mean_predicted_size": float(mask.sum(dim=1).float().mean())}
        calibration.append(row)
        if best is None or exact > best["calibration_exact"]:
            best = row
    chosen = best["threshold"]

    dev_results = []
    for row in calibration:
        threshold = row["threshold"]
        mask = (torch.sigmoid(dev_logits) >= threshold)
        exact = constraint_exact_at(mask, dev_pairs, vocab)
        dev_results.append({"threshold": threshold, "dev_constraint_exact": exact,
                            "dev_constraint_rate": exact / len(dev_pairs),
                            "mean_predicted_size": float(mask.sum(dim=1).float().mean())})

    at_chosen = next(row for row in dev_results if row["threshold"] == chosen)
    report = {
        "split": {"fit": fit_count, "calibration": calibrate_count, "dev": len(dev_pairs)},
        "classes": len(vocab.triples),
        "positives_per_case": float(all_targets.sum() / all_targets.shape[0]),
        "positive_rate": float(all_targets.sum() / all_targets.numel()),
        "pos_weight_median": (float(pos_weight.median()) if pos_weight is not None else 1.0),
        "threshold_chosen_on_train_calibration": chosen,
        "dev_constraint_exact_at_chosen": at_chosen,
        "dev_sweep": dev_results,
        "calibration_sweep": calibration,
        "reference": {
            "pair_model_dev_constraint_exact_rate": 0.0344,
            "pair_model_oracle_topk_recall": 0.3159,
            "per_triple_linear_probe_dev_auc_mean": 0.9788,
            "note": ("four_slot_exact can never exceed constraint_exact, because it "
                     "requires the constraint slot to match as well"),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
