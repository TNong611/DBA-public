"""Does enforcing the constraint's GROUP structure raise the ceiling?

Measured on this dev split:
  * flat 931-class separability is 0.2372 -- a perfect threshold could win only 23.7% of
    cases, matching the oracle-size decoder's 0.2493. So plain decoding is at its limit;
  * of the cases that fail, the blocking examples are mostly WRONG ATTRIBUTE (the model
    answers 区域 / 制片国家地区 where the gold has 重点科室 / 年代), not wrong value;
  * the 931 triples factor into 174 (marker, attribute) groups, and a case asserts only a
    few of them.

The structural claim to test: exact match is so brutal partly because 929 negatives compete
through a maximum. A (marker, attribute) group can normally assert only ONE value, so
enforcing "at most one value per active group" removes a whole family of near-miss errors
without changing the model at all.

Decoders compared on the SAME logits from the SAME flat head:
  (a) global threshold              -- the baseline
  (b) per-group argmax + threshold  -- structure-enforcing, realistic
  (c) oracle active groups + argmax -- upper bound of the structural hypothesis

Reading the result:
  * (b) or (c) far above (a): the flat decoder was the wall and a structural decoder is
    the fix;
  * (c) near (a): the group structure does not rescue it and the frozen features are the
    wall regardless of decoding.

Also reports how many values a group actually asserts at once, since (b) is only sound if
that is 1.
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


def exact(predicted_sets, references):
    hits = sum(1 for predicted, reference in zip(predicted_sets, references)
               if predicted == reference["constraint"])
    return hits, hits / len(references)


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
        probs = torch.sigmoid(x_dev.mm(weight) + bias).float().cpu()

    # ---- group structure ---------------------------------------------------
    triples = [vocab.triples.decode(position) for position in range(len(vocab.triples))]
    group_of = [(triple[0], triple[1]) for triple in triples]
    group_members = collections.defaultdict(list)
    for position, group in enumerate(group_of):
        group_members[group].append(position)
    group_list = sorted(group_members)
    group_index = {group: index for index, group in enumerate(group_list)}

    # Soundness check for "at most one value per group": if this is >1 the structural
    # decoder could not express the reference and the comparison below would be unfair.
    values_per_group = 0
    group_multiplicity = collections.Counter()
    for entry in train_pairs + dev_pairs:
        per_group = collections.defaultdict(set)
        for marker, attribute, value in entry["constraint"]:
            per_group[(marker, attribute)].add(value)
        for values in per_group.values():
            group_multiplicity[len(values)] += 1
        values_per_group = max(values_per_group, max(
            (len(values) for values in per_group.values()), default=0))

    # Group score matrices: [dev, groups] best value score, and the argmax position.
    group_best = torch.full((probs.shape[0], len(group_list)), -1.0)
    group_argmax = torch.zeros((probs.shape[0], len(group_list)), dtype=torch.long)
    for group, index in group_index.items():
        members = group_members[group]
        block = probs[:, members]
        best, argmax = block.max(dim=1)
        group_best[:, index] = best
        group_argmax[:, index] = torch.tensor(
            [members[position] for position in argmax.tolist()])

    report = {
        "groups": len(group_list),
        "triples": len(vocab.triples),
        "values_per_group_distribution": dict(sorted(group_multiplicity.items())),
        "max_values_in_one_group": values_per_group,
        "structural_decoder_is_sound": values_per_group == 1,
        "global_threshold": [],
        "group_threshold": [],
        "oracle_groups": None,
    }

    for raw in args.thresholds.split(","):
        threshold = float(raw)
        # (a) flat threshold over all 931 classes
        mask = probs >= threshold
        predicted = [
            frozenset(vocab.triples.decode(position)
                      for position in mask[row].nonzero(as_tuple=False).flatten().tolist())
            for row in range(mask.shape[0])]
        hits, rate = exact(predicted, references)
        report["global_threshold"].append(
            {"threshold": threshold, "exact": hits, "rate": rate,
             "mean_size": sum(len(item) for item in predicted) / len(predicted)})

        # (b) per-group argmax, group active when its best score passes the threshold
        active = group_best >= threshold
        predicted = []
        for row in range(active.shape[0]):
            positions = [int(group_argmax[row, index])
                         for index in active[row].nonzero(as_tuple=False).flatten().tolist()]
            predicted.append(frozenset(vocab.triples.decode(p) for p in positions))
        hits, rate = exact(predicted, references)
        report["group_threshold"].append(
            {"threshold": threshold, "exact": hits, "rate": rate,
             "mean_size": sum(len(item) for item in predicted) / len(predicted)})

    # (c) oracle active groups: how far can the structure go if the GROUP SET were known?
    predicted = []
    for row, reference in enumerate(references):
        active_groups = {(marker, attribute)
                         for marker, attribute, _ in reference["constraint"]}
        positions = [int(group_argmax[row, group_index[group]])
                     for group in active_groups if group in group_index]
        predicted.append(frozenset(vocab.triples.decode(p) for p in positions))
    hits, rate = exact(predicted, references)
    report["oracle_groups"] = {"exact": hits, "rate": rate}

    report["reference"] = {
        "flat_separability_rate": 0.2372,
        "flat_best_threshold_rate": 0.119,
        "size_decoder_rate": 0.2115,
        "oracle_size_rate": 0.2493,
        "pair_model_rate": 0.0344,
        "four_slot_target": 0.80,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
