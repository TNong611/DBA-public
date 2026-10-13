"""Python 3.14 outbox -> isolated CUDA RWKV interpreter; no silent fallback."""
import json
import os
from pathlib import Path
import subprocess
import tempfile


def summarize(event, memory, ollama):
    import nb_turn
    home = Path(__file__).resolve().parents[1]
    python = Path(os.environ.get('DBA_RWKV_PYTHON', str(home / '.venv-rwkv' / 'Scripts' / 'python.exe')))
    if not python.is_file():
        raise nb_turn.TurnError('RWKV CUDA interpreter missing: ' + str(python))
    # Free Ollama models before allocating RWKV weights on the same 8GB GPU.
    for model in dict.fromkeys((ollama.model, ollama.event_model)):
        ollama.unload(model)
    env = dict(os.environ, DBA_HOME=str(Path(memory).parent), RWKV_V7_ON='1', RWKV_CUDA_ON='0', RWKV_JIT_ON='0')
    with tempfile.TemporaryDirectory(prefix='dba-rwkv-result-') as temp:
        output = Path(temp) / 'result.json'
        run = subprocess.run([str(python), str(home / 'scripts' / 'nb_rwkv.py'), 'process', '--output', str(output)],
            input=json.dumps(event, ensure_ascii=False), encoding='utf-8', stdout=subprocess.DEVNULL,
            stderr=None, env=env, timeout=int(os.environ.get('DBA_RWKV_TIMEOUT', '900')))
        if not output.is_file():
            raise nb_turn.TurnError('RWKV did not produce a result; exit=' + str(run.returncode))
        result = json.loads(output.read_text(encoding='utf-8'))
        if run.returncode or result.get('ok') is not True:
            raise nb_turn.TurnError('RWKV failed: ' + str(result.get('error') or result.get('summary_error')))
        if not result.get('source_committed') or not isinstance(result.get('summary'), str):
            raise nb_turn.TurnError('RWKV result contract invalid')
        return result
