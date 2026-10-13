"""Does the ranking even CONTAIN the right constraint set?

The size-aware decoder lifted constraint exact from 11.9% to 21.2%, so decoding mattered.
But that is still far below the 80% four-slot target, and four-slot exact can never exceed
constraint exact. Before spending another training run, this asks the sharper question:

  for a given case, does ANY threshold on the model's own scores recover the gold set?

If yes, the case is decodable and a smarter decoder could in principle win it. If no, then
the score vector literally ranks some non-gold triple above a gold one, and no threshold,
size head, or calibration can fix that case -- the representation is the wall.

The quantity is separation: the lowest-scoring GOLD triple must outrank the highest-scoring
non-gold triple. Reported both as "separable" and as the size of the violation, so a near
miss is visible rather than being flattened into a yes/no.

Reads the cache read-only, fits the same linear multi-label head as the other measurements,
CPU or CUDA. Prints JSON; writes nothing.
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--cache-identity", required=True)
    parser.add_argument("--pool", default="meanmax")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-clamp", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    torch = P.base._torch()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no CUDA; pass --device cpu")

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
    x_train = features[:train_count].to(args.device)
    x_dev = features[train_count:].to(args.device)

    raw_targets = P.encode_targets(train_pairs, vocab, torch).multi
    targets = raw_targets.to(args.device)
    pos_weight = P.triple_positive_weight(raw_targets, torch, power=args.pos_weight_power,
                                          clamp=args.pos_weight_clamp).to(args.device)

    torch.manual_seed(20261008)
    weight = torch.zeros(x_train.shape[1], len(vocab.triples),
                         device=args.device).normal_(0, 0.01).requires_grad_(True)
    bias = torch.zeros(len(vocab.triples), device=args.device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=args.lr, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(20261008)
    for step in range(args.steps):
        index = torch.randint(0, x_train.shape[0], (args.batch_size,), generator=generator)
        optimizer.zero_grad()
        loss = loss_fn(x_train[index].mm(weight) + bias, targets[index])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        dev_logits = (x_dev.mm(weight) + bias).float().cpu()
    dev_targets = P.encode_targets(dev_pairs, vocab, torch).multi

    gold_positions = dev_targets > 0.5
    separable = 0
    empty_cases = 0
    margins = []
    blocking_pairs = []
    for row in range(dev_logits.shape[0]):
        scores = dev_logits[row]
        gold = gold_positions[row]
        gold_count = int(gold.sum())
        if gold_count == 0:
            empty_cases += 1
            continue
        worst_gold = float(scores[gold].min())
        non_gold = ~gold
        if not bool(non_gold.any()):
            separable += 1
            continue
        best_non_gold = float(scores[non_gold].max())
        margins.append(worst_gold - best_non_gold)
        if worst_gold > best_non_gold:
            separable += 1
        elif len(blocking_pairs) < 5:
            # Record what outranks a gold triple: is it a near-neighbour attribute/value,
            # or unrelated? That decides whether a constraint-structure decoder could help.
            offender = int(scores.masked_fill(gold, float("-inf")).argmax())
            victim = int(scores.masked_fill(~gold, float("inf")).argmin())
            blocking_pairs.append({
                "id": dev_pairs[row]["_id"],
                "gold_outranked": vocab.triples.decode(victim),
                "non_gold_above": vocab.triples.decode(offender),
                "gap": worst_gold - best_non_gold,
            })

    scored = len(dev_logits) - empty_cases
    margins.sort()
    report = {
        "dev_cases": len(dev_logits),
        "cases_with_gold": scored,
        "cases_with_empty_gold": empty_cases,
        "separable_cases": separable,
        "separable_rate": separable / max(1, scored),
        "median_margin_gap": (margins[len(margins) // 2] if margins else None),
        "worst_quartile_gap": (margins[len(margins) // 4] if margins else None),
        "examples_of_blocking_pairs": blocking_pairs,
        "reference": {
            "linear_threshold_dev_rate": 0.119,
            "linear_size_decoder_dev_rate": 0.2115,
            "linear_oracle_size_dev_rate": 0.2493,
            "pair_model_dev_rate": 0.0344,
            "four_slot_target": 0.80,
        },
        "reading": ("separable_rate is the fraction of cases a perfect threshold COULD win. "
                    "If it is near the observed 21-25%, no decoder change can help and the "
                    "frozen pooled features are the wall. If it is far higher, the gap is "
                    "decoding and a structure-aware decoder is worth building."),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
