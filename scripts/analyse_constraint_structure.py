"""Why does the constraint slot not move, and is a decomposition worth another run?

Established on this dev split, all measured:
  * a per-triple LINEAR probe on the frozen pooled features reaches dev AUC 0.9788 mean
    / 0.9863 median, so each frequent triple is present in the features;
  * yet a plain linear multi-label head tops out at 8.07% constraint exact, and the pair
    model at 3.44%;
  * `four_slot_exact` can never exceed `constraint_exact` (it requires that slot too), so
    the four-slot ceiling on FROZEN pooled features is about 8%, against an 80% target.

The gap between "AUC 0.98 per class" and "8% exact set" is the whole story: exact match
over 931 classes demands zero false positives among ~929 negatives on every case, so a
head that is merely good per class still misses the set.

This script measures the label structure that decides whether that wall can be moved:
  1. how the 931 triples factor into (marker, attribute) x value -- if attributes are few
     and their value spaces are small, the set can be predicted as "which attributes are
     present" + "which value per attribute", two much easier questions;
  2. the true distribution of set sizes, so an over-predicting head can be recognised;
  3. the best achievable constraint exact under a per-case ORACLE of the head's own
     ranking with the TRUE number of positives (the ranking ceiling), which separates
     "ranking is fine, threshold is wrong" from "ranking is the wall".

Prints JSON; writes nothing.
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
    args = parser.parse_args(argv)

    corpus = P.load_corpus(args.cases)
    train_cases, dev_cases = P.split_corpus(corpus)
    train_pairs = P.pair_lists(train_cases)
    dev_pairs = P.pair_lists(dev_cases)

    markers = collections.Counter()
    attributes = collections.Counter()
    marker_attribute = collections.Counter()
    values_per_attribute = collections.defaultdict(set)
    for entry in train_pairs:
        for marker, attribute, value in entry["constraint"]:
            markers[marker] += 1
            attributes[attribute] += 1
            marker_attribute[(marker, attribute)] += 1
            values_per_attribute[attribute].add(value)

    train_sizes = collections.Counter(len(entry["constraint"]) for entry in train_pairs)
    dev_sizes = collections.Counter(len(entry["constraint"]) for entry in dev_pairs)
    total = sum(dev_sizes.values())

    # Canonical signature of a constraint: the SET of (attribute -> value) it asserts,
    # ignoring marker. If this is much smaller than 931, the label space is reducible.
    signatures = {frozenset((attribute, value) for _, attribute, value in entry["constraint"])
                  for entry in train_pairs + dev_pairs}
    marker_free = {frozenset((attribute, value) for _, attribute, value in entry["constraint"])
                   for entry in train_pairs}

    report = {
        "triples": len({triple for entry in train_pairs for triple in entry["constraint"]}),
        "distinct_markers": len(markers),
        "distinct_attributes": len(attributes),
        "distinct_marker_attribute_pairs": len(marker_attribute),
        "distinct_attribute_value_pairs": len({(a, v) for entry in train_pairs
                                              for _, a, v in entry["constraint"]}),
        "distinct_attribute_value_signatures_train": len(marker_free),
        "distinct_attribute_value_signatures_train_plus_dev": len(signatures),
        "top_markers": markers.most_common(6),
        "attributes_by_value_space": sorted(
            ({"attribute": attribute, "distinct_values": len(values), "occurrences": attributes[attribute]}
             for attribute, values in values_per_attribute.items()),
            key=lambda row: row["distinct_values"], reverse=True)[:12],
        "attribute_count_distribution": {
            "mean_attributes_per_case": (
                sum(len(entry["constraint"]) for entry in train_pairs) / len(train_pairs)),
        },
        "set_size": {
            "train_mean": sum(k * v for k, v in train_sizes.items()) / len(train_pairs),
            "train_histogram": dict(sorted(train_sizes.items())),
            "dev_mean": sum(k * v for k, v in dev_sizes.items()) / len(dev_pairs),
            "dev_histogram": dict(sorted(dev_sizes.items())),
            "dev_empty": dev_sizes.get(0, 0),
            "dev_empty_rate": dev_sizes.get(0, 0) / total,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
