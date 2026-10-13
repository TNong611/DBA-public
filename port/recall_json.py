"""把 `dba.recall` 的结果以 JSON 输出，供非 Python 调用方（DSH 壳子插件）使用。

用法：
    py -3.14 recall_json.py "关键词" [--top-k 4] [--scope ws] [--mode auto]

输出（单行 JSON，UTF-8）：
    {query, hits:[{id,file,scope,head,text,score,source}], text, channels,
     degraded, bytes}

失败时**不吞**：异常照样抛出（退出码非 0），stderr 带 traceback —— 壳子那侧
据此显示错误，而不是显示一份「看起来正常的空结果」。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

DBA_HOME = os.environ.get("DBA_HOME") or r"D:\AI\DBA"
if DBA_HOME not in sys.path:
    sys.path.insert(0, DBA_HOME)

dba = importlib.import_module("port.dba")


def main() -> int:
    parser = argparse.ArgumentParser(description="DBA recall -> JSON")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--scope", default=None)
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "kw", "vec", "both", "ladder"])
    args = parser.parse_args()

    result = dba.recall(args.query, top_k=args.top_k, scope=args.scope,
                        mode=args.mode)
    hits = []
    for hit in result.get("hits") or []:
        hits.append({
            "id": hit.get("id"),
            "file": hit.get("file"),
            "scope": hit.get("scope"),
            "head": hit.get("head"),
            "text": hit.get("text"),
            "score": hit.get("score"),
            "source": hit.get("source"),
        })
    payload = {
        "query": result.get("query"),
        "hits": hits,
        "text": result.get("text"),
        "channels": result.get("channels"),
        "degraded": result.get("degraded"),
        "bytes": result.get("bytes"),
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
