#!/usr/bin/env python3
"""Frozen RWKV [72,2048] state -> all target tokens in parallel.

Non-autoregressive pilot: learned position queries cross-attend to all 72 raw state rows
and one forward pass returns [B,L,V] logits. No generated token is fed back.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0, str(SCRIPT_DIR))
import rwkv_slots_data as DATA
import train_rwkv_slot_classifier as base
from train_rwkv_slots import BackboneTokenizer, special_ids
TrainingError = base.TrainingError
DEFAULT_CASES = str(DATA.DEFAULT_CASES if hasattr(DATA, "DEFAULT_CASES") else r"D:\AI\DBA\training\web-sources\risawoz\memory_cases.jsonl")
DEFAULT_CACHE = r"D:\AI\DBA\training\risawoz-run-v3\features\1f0b28205d715209"
DEFAULT_MODEL = r"D:\AI\models\rwkv7\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"
DEFAULT_SHA = "23e13ff62ac6f80b28910ffc230324939e80ddb00b98f892a5951c0da071e700"

def make_model(torch, feature_dim, vocab_size, max_len, d_model=256, heads=4, layers=2):
    nn = torch.nn
    class ParallelDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.row_proj = nn.Linear(feature_dim, d_model)
            self.kind_emb = nn.Embedding(3, d_model)
            self.layer_emb = nn.Embedding(24, d_model)
            self.row_norm = nn.LayerNorm(d_model)
            enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers)
            self.query = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
            self.cross = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.cross_norm = nn.LayerNorm(d_model)
            self.out_norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)
        def forward(self, rows):
            b, device = rows.shape[0], rows.device
            kind = torch.arange(72, device=device) % 3
            layer = torch.arange(72, device=device) // 3
            memory = self.row_proj(rows) + self.kind_emb(kind) + self.layer_emb(layer)
            memory = self.encoder(self.row_norm(memory))
            queries = self.query.unsqueeze(0).expand(b, -1, -1)
            attended, _ = self.cross(self.cross_norm(queries), memory, memory, need_weights=False)
            return self.lm_head(self.out_norm(attended))
    return ParallelDecoder()

def encode_targets(tokenizer, ids, texts, max_len, torch, device):
    pad, result = ids["pad"], []
    for text in texts:
        tokens = [ids["bos"]] + tokenizer.encode(text) + [ids["eos"]]
        if len(tokens) > max_len: raise TrainingError("target length exceeds --max-len")
        result.append(tokens + [pad] * (max_len - len(tokens)))
    return torch.tensor(result, dtype=torch.long, device=device)

def load_rows(cache, cases, torch):
    rows = []
    for case in cases:
        row = cache.row(case["id"])
        if row is None: raise TrainingError("feature cache missing case %s" % case["id"])
        if tuple(row.shape) != (72, 2048): raise TrainingError("expected [72,2048], got %s" % (tuple(row.shape),))
        rows.append(row.float())
    return torch.stack(rows)

def self_test():
    import torch
    model = make_model(torch, 16, 31, 12, d_model=32, heads=4, layers=1)
    logits = model(torch.randn(3, 72, 16))
    assert tuple(logits.shape) == (3, 12, 31)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 31), torch.randint(0, 31, (3, 12)).reshape(-1))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(json.dumps({"self_test": "parallel logits [B,L,V]", "shape": list(logits.shape), "loss": float(loss.detach())}, ensure_ascii=False))
    return 0

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=DEFAULT_CASES); p.add_argument("--feature-cache", default=DEFAULT_CACHE); p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--backbone-sha256", default=DEFAULT_SHA); p.add_argument("--out-dir", default=r"D:\AI\DBA\training\rwkv-parallel-token"); p.add_argument("--max-len", type=int, default=192); p.add_argument("--d-model", type=int, default=256); p.add_argument("--heads", type=int, default=4); p.add_argument("--layers", type=int, default=2); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--steps", type=int, default=3000); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--seed", type=int, default=20261008); p.add_argument("--device", default="cuda"); p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test: return self_test()
    if not args.device.startswith("cuda"): raise TrainingError("GPU-only trainer refuses CPU")
    torch = base._torch()
    if not torch.cuda.is_available(): raise TrainingError("CUDA is required")
    random.seed(args.seed); torch.manual_seed(args.seed)
    items = DATA.all_cases(args.cases); train_items = [x for x in items if x["group"] == DATA.TRAIN]; dev_items = [x for x in items if x["group"] == DATA.HELDOUT]
    data_hash = DATA.data_version(items)
    cache = base.ReadOnlyFeatureCache(args.feature_cache, data_hash, args.backbone_sha256); cache_report = cache.open(); cache.require_all([x["id"] for x in train_items + dev_items])
    from nb_rwkv import TorchEngine
    engine = TorchEngine(args.model, chunk_tokens=256, generation_tokens=1); table = getattr(engine.pipeline.tokenizer, "idx2token", {}); vocab_size = max(table.keys()) + 1 if table else 0
    tokenizer = BackboneTokenizer(engine.pipeline, vocab_size); ids = special_ids(tokenizer)
    texts_train = [DATA.memory_text(x) for x in train_items]; texts_dev = [DATA.memory_text(x) for x in dev_items]
    max_seen = max(len([ids["bos"]] + tokenizer.encode(t) + [ids["eos"]]) for t in texts_train + texts_dev)
    if max_seen > args.max_len: raise TrainingError("--max-len below observed max %d" % max_seen)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(cache, train_items + dev_items, torch); mean = rows[:len(train_items)].mean(dim=(0,1), keepdim=True); std = rows[:len(train_items)].std(dim=(0,1), keepdim=True).clamp_min(1e-5); rows = (rows - mean) / std
    train_rows, dev_rows = rows[:len(train_items)], rows[len(train_items):]
    train_y = encode_targets(tokenizer, ids, texts_train, args.max_len, torch, args.device); dev_y = encode_targets(tokenizer, ids, texts_dev, args.max_len, torch, "cpu")
    model = make_model(torch, 2048, tokenizer.vocab_size, args.max_len, args.d_model, args.heads, args.layers).to(args.device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01); pad = ids["pad"]; order = list(range(len(train_items)))
    for step in range(args.steps):
        if step % len(order) == 0: random.shuffle(order)
        start = (step * args.batch_size) % len(order); batch = order[start:start+args.batch_size]; batch = (batch + order)[:args.batch_size] if len(batch) < args.batch_size else batch
        optimizer.zero_grad(set_to_none=True); logits = model(train_rows[batch].to(args.device)); loss = torch.nn.functional.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), train_y[batch].reshape(-1), ignore_index=pad); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if (step + 1) % 100 == 0: print("step %d/%d loss=%.5f peak=%dMiB" % (step+1, args.steps, float(loss.detach()), int(torch.cuda.max_memory_allocated()/1048576)), flush=True)
    model.eval()
    # Never materialise [1773, max_len, 65533] logits: that is an accidental 80+ GiB
    # allocation. The decoder still emits every position in parallel; evaluation only
    # chunks cases because the vocabulary projection is wide.
    predictions = []
    with torch.no_grad():
        for start in range(0, len(dev_rows), max(1, args.batch_size)):
            block = model(dev_rows[start:start + max(1, args.batch_size)].to(args.device))
            predictions.extend(block.argmax(dim=-1).cpu())
    exact = sum(1 for row, target in zip(predictions, dev_y) if row.tolist() == target.tolist())
    report = {"steps": args.steps, "dev_cases": len(dev_items), "parallel_token_exact": exact, "parallel_token_exact_rate": exact / len(dev_items), "vocab_size": tokenizer.vocab_size, "max_len": args.max_len, "cache": cache_report, "backbone_training": False, "note": "sealed memory_test.jsonl was not read"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(report, ensure_ascii=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
