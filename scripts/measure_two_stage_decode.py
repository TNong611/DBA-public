"""Two-stage decoding: predict WHICH (marker, attribute) groups, then one value each.

The measured ladder on this dev split, all from the same frozen pooled features:
  flat 931-class threshold            11.9%
  flat + predicted set size + top-k   21.2%
  flat separability (perfect thresh)  23.7%
  ORACLE groups + per-group argmax    51.8%   <-- the structural decoder's ceiling

That last number is the interesting one. Enforcing the constraint's real structure -- a
case activates a handful of (marker, attribute) groups and each group asserts essentially
one value (25307 groups assert 1 value, 33 assert 2) -- nearly doubles what a perfect flat
threshold can do. So the question is no longer whether the features carry the answer; it is
how well the ACTIVE GROUP SET can be predicted, since that is what the 51.8% oracle assumed.

This measures the realistic two-stage decoder end to end:
  stage 1: active (marker, attribute) groups -- 174 classes, a far smaller problem than 931
  stage 2: within each active group, take the argmax value from the flat head

and reports the decomposition, because the two stages need different fixes:
  * group set accurate, joint poor  -> the value decision is the bottleneck;
  * group set poor                  -> the group decision is the bottleneck.

Also measures a dedicated 174-class group head, since the group score can be derived as the
max over a group's members (free) or learned directly (one extra matrix).
"""
from __future__ import annotations

import argparse
import collections
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
    parser.add_argument("--group-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--pos-weight-clamp", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", default="0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2")
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
    references = [dict(entry) for entry in dev_pairs]

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

    # ---- group index -------------------------------------------------------
    triples = [vocab.triples.decode(position) for position in range(len(vocab.triples))]
    group_of = [(triple[0], triple[1]) for triple in triples]
    group_list = sorted(set(group_of))
    group_index = {group: index for index, group in enumerate(group_list)}
    member_positions = collections.defaultdict(list)
    for position, group in enumerate(group_of):
        member_positions[group].append(position)

    group_targets = torch.zeros(train_count, len(group_list), device=args.device)
    for row, entry in enumerate(train_pairs):
        for marker, attribute, _ in entry["constraint"]:
            group_targets[row, group_index[(marker, attribute)]] = 1.0
    group_weight = P.triple_positive_weight(group_targets.cpu(), torch,
                                            power=args.pos_weight_power,
                                            clamp=args.pos_weight_clamp).to(args.device)

    generator = torch.Generator().manual_seed(20261008)

    def fit(n_out, y, pw, steps, lr):
        torch.manual_seed(20261008)
        w = torch.zeros(x_train.shape[1], n_out, device=args.device).normal_(
            0, 0.01).requires_grad_(True)
        b = torch.zeros(n_out, device=args.device, requires_grad=True)
        opt = torch.optim.AdamW([w, b], lr=lr, weight_decay=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        for _ in range(steps):
            index = torch.randint(0, x_train.shape[0], (args.batch_size,),
                                  generator=generator)
            opt.zero_grad()
            loss = loss_fn(x_train[index].mm(w) + b, y[index])
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (x_dev.mm(w) + b).float().cpu()

    flat_logits = fit(len(vocab.triples), targets, pos_weight, args.steps, args.lr)
    flat_probs = torch.sigmoid(flat_logits)

    # Derived group score: the best member score. Free -- the same head, reinterpreted.
    derived_group_logits = torch.full((flat_logits.shape[0], len(group_list)), -30.0)
    group_argmax = torch.zeros((flat_logits.shape[0], len(group_list)), dtype=torch.long)
    for group, index in group_index.items():
        members = member_positions[group]
        best, argmax = flat_logits[:, members].max(dim=1)
        derived_group_logits[:, index] = best
        group_argmax[:, index] = torch.tensor(
            [members[position] for position in argmax.tolist()])

    learned_group_logits = fit(len(group_list), group_targets, group_weight,
                               args.group_steps, args.lr)

    gold_group_sets = [
        frozenset((marker, attribute) for marker, attribute, _ in entry["constraint"])
        for entry in dev_pairs]

    def evaluate_group_decoder(logits):
        rows = []
        for raw in args.thresholds.split(","):
            threshold = float(raw)
            active = torch.sigmoid(logits) >= threshold
            exact_sets = 0
            predicted = []
            for row in range(active.shape[0]):
                groups = frozenset(
                    group_list[index]
                    for index in active[row].nonzero(as_tuple=False).flatten().tolist())
                if groups == gold_group_sets[row]:
                    exact_sets += 1
                positions = [int(group_argmax[row, group_index[group]])
                             for group in groups if group in group_index]
                predicted.append(frozenset(vocab.triples.decode(p) for p in positions))
            hits = sum(1 for p, r in zip(predicted, references)
                       if p == r["constraint"])
            rows.append({"threshold": threshold,
                         "group_set_exact": exact_sets,
                         "group_set_exact_rate": exact_sets / len(references),
                         "constraint_exact": hits,
                         "constraint_exact_rate": hits / len(references),
                         "mean_groups": sum(len(s) for s in
                                            [frozenset(group_list[i] for i in
                                                       active[row].nonzero(
                                                           as_tuple=False).flatten().tolist())
                                             for row in range(active.shape[0])])
                                        / active.shape[0]})
        return rows

    report = {
        "groups": len(group_list),
        "triples": len(vocab.triples),
        "derived_group_decoder": evaluate_group_decoder(derived_group_logits),
        "learned_group_decoder": evaluate_group_decoder(learned_group_logits),
        "reference": {
            "flat_threshold_best": 0.119,
            "flat_size_decoder": 0.2115,
            "flat_oracle_size": 0.2493,
            "flat_separability": 0.2372,
            "oracle_groups_argmax": 0.5178,
            "pair_model": 0.0344,
            "four_slot_target": 0.80,
        },
        "reading": ("compare group_set_exact_rate with constraint_exact_rate. A high group "
                    "rate with a low constraint rate means the group decision is solved and "
                    "the VALUE decision is the bottleneck; both low means the group decision "
                    "is where a dedicated head should be spent."),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
