"""Baseline fixed-slot memory output from real RWKV-7 recurrent states."""
import json
import os
import time
from pathlib import Path

os.environ["RWKV_JIT_ON"] = "0"
from nb_rwkv import TorchEngine, DEFAULT_MODEL

SAMPLES = [
    {"user": "请记住：电机控制器的闭环调试还没做，不能写成已经验证。", "assistant": "已记录约束。下一步先检查编码器极性，再做低速空载测试。"},
    {"user": "这个项目只能用GPU，CPU推理不接受。", "assistant": "明白。本次 RWKV 推理使用 CUDA BF16；若 GPU 不可用则直接失败，不回退 CPU。"},
    {"user": "索引恢复失败了，先修复重建流程，不要动原始事件。", "assistant": "我会保留原始事件，检查损坏索引并尝试从正文重建；完成前不报告修复成功。"},
]
PROMPT = """
\nUser: 输出固定槽位中文记忆，严格按以下四行，每行一槽，缺少信息写“无”：
主题：<当前主题>
约束：<仍需遵守的条件>
未完成：<下一步或未解决目标>
结果：<已确认完成的事实；未验证必须明确写未验证>
不得添加输入中没有的事实，不得把计划写成已完成。只输出四行。
Assistant:
""".strip()

def main():
    engine = TorchEngine(model_path=DEFAULT_MODEL, chunk_tokens=64, generation_tokens=96)
    outputs = []
    torch = engine.torch
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    for i, sample in enumerate(SAMPLES):
        state, observed_tokens = engine.observe(None, "<observed_turn>\n" + json.dumps(sample, ensure_ascii=False) + "\n</observed_turn>")
        branch = engine.clone(state)
        prompt_tokens = engine.pipeline.encode(PROMPT)
        generated = []
        with torch.inference_mode():
            for offset in range(0, len(prompt_tokens), engine.chunk_tokens):
                logits, branch = engine.model.forward(prompt_tokens[offset:offset + engine.chunk_tokens], branch)
            for _ in range(engine.generation_tokens):
                token = int(torch.argmax(logits).item())
                if token == 0:
                    break
                generated.append(token)
                decoded = engine.pipeline.decode(generated)
                if decoded.count("\n") >= 4:
                    break
                if len(decoded) > 280:
                    break
                logits, branch = engine.model.forward([token], branch)
        text = engine.pipeline.decode(generated)
        outputs.append({"sample": i + 1, "observed_tokens": observed_tokens,
                        "prompt_tokens": len(prompt_tokens), "generated_tokens": len(generated), "slots": text})
    elapsed = time.perf_counter() - started
    print(json.dumps({"model": engine.identity, "device": torch.cuda.get_device_name(0),
        "dtype": "cuda:0 bf16", "samples": outputs, "elapsed_seconds": round(elapsed, 3),
        "seconds_per_sample": round(elapsed / len(SAMPLES), 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
