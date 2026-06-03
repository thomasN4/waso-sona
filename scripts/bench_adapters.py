"""Benchmark fine-tuned QLoRA adapters for Toki Pona generation quality.

Compares N adapters on a fixed continuation-prompt set at both greedy and
sampled decoding, reporting:

- **Validity**: doc keep-rate (non-TP-letter ratio ≤ 0.05), sentence
  keep-rate (`bench_gemma.looks_like_toki_pona_sentence`), and the strict
  `augment_corpus._filter_sentence` pass-rate.
- **Diversity / repetition**: per-generation distinct-1/2/3 averaged
  across outputs — the direct measure of the looping/degeneration that
  dominates quality at this corpus scale (low distinct-n = heavy
  repetition).

Generation is HF/PEFT (loads base in 4-bit nf4 + applies the adapter),
mirroring scripts/infer.py, so it works on local adapter dirs that aren't
in Ollama. Raw generations are saved per (model, setting) for eyeballing.

Usage::

    python scripts/bench_adapters.py
    python scripts/bench_adapters.py --max-new-tokens 96 --n-sampled 2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

# Reduce allocator fragmentation across the three sequential model loads —
# the un-pruned baseline alone is ~7.3 GB on the 8 GB card, so the previous
# model must be fully released before the next loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus  # noqa: E402
from bench_gemma import (  # noqa: E402
    _SENT_SPLIT_RE,
    looks_like_toki_pona_sentence,
    non_tp_letter_ratio,
)

BENCH_DIR = REPO_ROOT / "data" / "bench"

# (label, adapter dir, base model). Bases differ: baseline rides the stock
# Gemma 4, the pruned adapters ride the pruned base.
DEFAULT_MODELS = [
    ("baseline", "models/runs/qlora-20260527T223141Z/final",
     "google/gemma-4-E2B-it"),
    ("parity", "models/runs/qlora-20260528T022232Z/final",
     "models/base-pruned"),
    ("bumped", "models/runs/qlora-20260528T085717Z/final",
     "models/base-pruned"),
]

# Diverse grammatical TP seeds, fed through augment_corpus._continuation_prompt
# (the real inference shape the models were trained on).
SEEDS = [
    "mi lon tomo mi. tenpo pimeja li kama.",
    "jan lili li musi lon ma kasi.",
    "tenpo suno ni la mi wile pali e ijo.",
    "soweli mi li moku e telo.",
    "jan lili li lukin e waso lon sewi.",
    "mi tawa ma kasi. mi lukin e kasi mute.",
    "telo sewi li kama lon tenpo pimeja.",
    "mi pilin pona tan ni: jan pona mi li kama.",
    "tomo mi li lon poka pi telo suli.",
    "kulupu mi li moku lon tenpo suno.",
    "waso laso li tawa sewi.",
    "mi sona ala e nasin tawa tomo sina.",
    "tenpo pini la mi lukin e jan sin.",
    "kasi suli li lon insa pi ma mi.",
    "jan mute li toki lon tomo pi sona.",
]

_WORD_RE = re.compile(r"[a-z]+")


# ---------------------------------------------------------------------------
# Model loading (mirrors scripts/infer.py)
# ---------------------------------------------------------------------------

def load_model(adapter_dir: Path, base_model: str):
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map={"": 0},
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if hasattr(model, "model"):
        for sub in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            mod = getattr(model.model, sub, None)
            if mod is not None:
                mod.to("cpu")
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tok


def generate(model, tok, seed: str, greedy: bool, max_new_tokens: int,
             gen_seed: int) -> str:
    user_msg = augment_corpus._continuation_prompt(seed)
    chat = [{"role": "user", "content": user_msg}]
    encoded = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    dev = model.get_input_embeddings().weight.device
    input_ids = (encoded["input_ids"] if hasattr(encoded, "input_ids") else encoded).to(dev)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if greedy:
        kwargs["do_sample"] = False
    else:
        torch.manual_seed(gen_seed)
        kwargs.update(do_sample=True, temperature=0.8, top_p=0.95)
    with torch.no_grad():
        out = model.generate(input_ids, **kwargs)
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _distinct_n(words: list[str], n: int) -> float:
    if len(words) < n:
        return 1.0
    grams = list(zip(*[words[i:] for i in range(n)]))
    return len(set(grams)) / len(grams) if grams else 1.0


def score_texts(texts: list[str]) -> dict:
    n = len(texts)
    docs_kept = 0
    sents_total = sents_valid = sents_strict = 0
    d1 = d2 = d3 = 0.0
    wlen = 0.0
    nonempty = 0
    for text in texts:
        if not text.strip():
            continue
        nonempty += 1
        if non_tp_letter_ratio(text) <= 0.05:
            docs_kept += 1
        words = _WORD_RE.findall(text.lower())
        wlen += len(words)
        d1 += _distinct_n(words, 1)
        d2 += _distinct_n(words, 2)
        d3 += _distinct_n(words, 3)
        for chunk in _SENT_SPLIT_RE.split(text):
            s = chunk.strip()
            if not s:
                continue
            sents_total += 1
            if looks_like_toki_pona_sentence(s):
                sents_valid += 1
            ok, _ = augment_corpus._filter_sentence(s)
            if ok:
                sents_strict += 1
    k = max(nonempty, 1)
    return {
        "n": n,
        "nonempty": nonempty,
        "doc_keep_rate": docs_kept / k,
        "sent_keep_rate": sents_valid / max(sents_total, 1),
        "strict_pass_rate": sents_strict / max(sents_total, 1),
        "distinct_1": d1 / k,
        "distinct_2": d2 / k,
        "distinct_3": d3 / k,
        "avg_words": wlen / k,
        "n_sents": sents_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--n-sampled", type=int, default=2,
                    help="sampled generations per seed at temp=0.8 (default 2)")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="how many seeds to use (default: all)")
    args = ap.parse_args(argv)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[: args.seeds]
    summary: dict[str, dict] = {}

    for label, adapter_rel, base in DEFAULT_MODELS:
        adapter_dir = REPO_ROOT / adapter_rel
        if not (adapter_dir / "adapter_model.safetensors").exists():
            print(f"[{label}] adapter missing at {adapter_dir}; skipping", file=sys.stderr)
            continue
        print(f"\n=== {label} ===  base={base}", flush=True)
        base_path = base if "/" not in base or base.startswith("google/") else str(REPO_ROOT / base)
        model, tok = load_model(adapter_dir, base_path)

        records = []
        greedy_texts, sampled_texts = [], []
        for si, seed in enumerate(seeds):
            g = generate(model, tok, seed, greedy=True,
                         max_new_tokens=args.max_new_tokens, gen_seed=0)
            greedy_texts.append(g)
            records.append({"seed": seed, "setting": "greedy", "sample": 0, "text": g})
            for k in range(args.n_sampled):
                s = generate(model, tok, seed, greedy=False,
                             max_new_tokens=args.max_new_tokens, gen_seed=42 + k)
                sampled_texts.append(s)
                records.append({"seed": seed, "setting": "sampled", "sample": k, "text": s})
            print(f"  [{label}] seed {si+1}/{len(seeds)} done", flush=True)

        # Save raw generations.
        outpath = BENCH_DIR / f"adapters_{label}.jsonl"
        with outpath.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary[label] = {
            "greedy": score_texts(greedy_texts),
            "sampled": score_texts(sampled_texts),
        }
        # Free VRAM before next model. del alone doesn't release CUDA memory
        # until the referent is collected, so force a gc pass first.
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  → {outpath.relative_to(REPO_ROOT)}  "
              f"(VRAM free after cleanup: {free/1e9:.2f} GB)", flush=True)

    (BENCH_DIR / "adapters_summary.json").write_text(json.dumps(summary, indent=2))

    # Comparison tables.
    def _table(setting: str):
        print(f"\n## {setting}")
        hdr = ("model", "doc_keep", "sent_keep", "strict", "dist-1", "dist-2",
               "dist-3", "avg_w")
        print("  " + " ".join(f"{h:>9}" for h in hdr))
        for label in summary:
            m = summary[label][setting]
            print("  " + " ".join(f"{v:>9}" for v in (
                f"{label:>9}",
                f"{m['doc_keep_rate']*100:.0f}%",
                f"{m['sent_keep_rate']*100:.0f}%",
                f"{m['strict_pass_rate']*100:.0f}%",
                f"{m['distinct_1']:.2f}",
                f"{m['distinct_2']:.2f}",
                f"{m['distinct_3']:.2f}",
                f"{m['avg_words']:.0f}",
            )))

    print("\n" + "=" * 60)
    print("distinct-n: higher = less repetition (1.0 = no repeats)")
    _table("greedy")
    _table("sampled")
    print(f"\n→ {(BENCH_DIR / 'adapters_summary.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
