"""tuning/get_iq3.py - fetch Qwen3.8-Flash-Next IQ3_XXS beside the IQ2_XS install for an A/B, the way setup.py does
steps 5-6 (shard 2 hard-linked from IQ2_XS: the same file for every size; shard 1 downloaded resumably; the pack
built by tools/iq_pack.py), without setup's engine build, settings or start script."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402

DATA = ROOT.parent / "Strata-data"        # where setup keeps the models (next to the Strata folder)
HF = "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/IQ3_XXS/"
dst = DATA / "models" / "IQ3_XXS"
dst.mkdir(parents=True, exist_ok=True)
s1 = dst / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
s2 = dst / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
src2 = DATA / "models" / "IQ2_XS" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"
if not s2.exists():
    os.link(src2, s2)
    S.mark(s2)
    print(f"shard 2 hard-linked from {src2.name}", flush=True)
S.download(HF + s1.name, s1)
S.check_shards([s1, s2])
print("shards OK", flush=True)
pack = DATA / "packs" / "iq3_xxs"
env = dict(os.environ, STRATA_GGUF_PY=str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
    S.run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(s1), "--out", str(pack)], env=env)
print(f"pack ready: {pack}", flush=True)
