"""Benchmark the from-scratch Toki Pona student LM for *fine-tune readiness*.

Pre-training (Stage 3) is done; this characterizes the base model's latent
capabilities and weaknesses so we can reason about what to fine-tune it for —
rather than reporting a single held-out loss. Five sections:

- **A. Perplexity** — exact (strided-window) NLL/perplexity. One *honest*
  held-out number on the 200 tatoeba docs the model never trained on (comparable
  to the training `real_loss≈2.08`), plus per-source IN-SAMPLE perplexity as a
  register-coverage indicator (loudly labelled — those sources were trained on,
  real upsampled ×6, so they are NOT generalization).
- **B. Free generation** — BOS-only and seeded continuations across a temperature
  sweep (greedy → 1.0). Validity (`score_texts`), diversity (distinct-n), vocab
  coverage, OOV rate, and a degeneration/looping rate. Greedy is the most
  diagnostic setting: heavy looping there means deterministic use needs sampling.
- **C. Prompted-continuation capability map** — how coherently the base continues
  prompts of different registers (narrative / mi-sina / seme-question /
  encyclopedic / instruction-ish). The ranking by validity is the most direct
  "what is it good at" signal for picking a fine-tune target.
- **D. Throughput** — batch-1 decode tok/s (the local-hardware / IME story).
- **E. Interpretation** — rule-based mapping of the above to candidate fine-tune
  directions, each tagged ready / promising / needs-work / not-supported.

Runs on the deliverable plus (by default) two earlier student checkpoints for
comparison. The model is ~30 MB, so — unlike the teacher bench — there's no need
to free the desktop GPU session.

Usage::

    python scripts/bench_student.py --smoke         # fast sanity, 1 checkpoint
    python scripts/bench_student.py                 # full run, 3 checkpoints
    python scripts/bench_student.py --checkpoints rebal --also-cpu
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_student import ModelConfig, TinyTPLM, _load_real_texts  # noqa: E402
import augment_corpus  # noqa: E402  (TP_VOCAB, _filter_sentence; import-safe)
from bench_gemma import (  # noqa: E402  (import-safe: stdlib + sitelen only)
    _SENT_SPLIT_RE,
    looks_like_toki_pona_sentence,
    non_tp_letter_ratio,
)
from sitelen.glyphs import WORD_TO_CODEPOINT  # noqa: E402  (137-word core vocab)

DEFAULT_TOKENIZER = REPO_ROOT / "models" / "student_tokenizer" / "spm.model"
DEFAULT_CORPUS = REPO_ROOT / "data" / "processed" / "corpus.filtered.jsonl"
RUNS_DIR = REPO_ROOT / "models" / "student_runs"
BENCH_DIR = REPO_ROOT / "data" / "bench"

# Student checkpoints to compare. Same tokenizer throughout → all numbers are
# directly comparable. The first is the deliverable (rebalanced 8L/384d); the
# rest trace the PIPELINE.md "Student runs" progression (real_loss 4.0→2.16→
# 2.22→2.08) so the new capability axes line up with the known loss history.
DEFAULT_CHECKPOINTS: dict[str, str] = {
    "rebal": "student-8L384d-rebal-20260607T051752Z/best.pt",   # deliverable, 8L/384d, real_loss 2.08
    "scaled": "student-8L384d-20260607T023646Z/best.pt",        # 8L/384d unbalanced, 2.22
    "v2": "student-20260605T073925Z/best.pt",                   # 6L/256d real+trans+synth mix, 2.16
    "v1": "student-20260602T192239Z/best.pt",                   # 6L/256d synthetic-only floor, 4.0
}

# corpus.filtered.jsonl is FULLY ORDERED BY SOURCE. Half-open [start, end)
# index spans (verified against the file). Used to slice per-source eval sets.
SOURCE_SPANS: dict[str, tuple[int, int]] = {
    "tatoeba": (0, 76909),
    "poki": (76909, 78484),
    "nltk-tp": (78484, 78951),
    "lipu": (78951, 78964),
    "tp1k": (78964, 79364),
    "tokwiki": (79364, 83355),
    "toki-ramble": (83355, 83363),
}
# train_student holds out real_all[:HELDOUT_N] for the honest real_loss — which
# (given the ordering) is entirely tatoeba single-sentences.
HELDOUT_N = 200
# Per-source IN-SAMPLE perplexity: which registers to report and in what order.
# lipu (13) / toki-ramble (8) omitted — too few tokens for a stable number.
INSAMPLE_SOURCES = ["tatoeba", "poki", "nltk-tp", "tp1k", "tokwiki"]

CAVEATS = [
    "Only the 200 tatoeba held-out docs (tatoeba_heldout_200) are true "
    "generalization data; the model never trained on them.",
    "All other per-source perplexity is IN-SAMPLE: the model trained on those "
    "docs (real corpus upsampled ×6). Treat in-sample ppl as a register-coverage "
    "indicator (which registers it represents tightly), NOT as generalization.",
    "Training `real_loss≈2.08` was a sampled loss on the same held-out tatoeba "
    "slice; section A recomputes it exactly (exp(2.08)≈8.0 ppl reference).",
    "The base only ever saw Toki Pona — any English/instruction-format target "
    "needs new data the base has never seen; the bench quantifies only TP-side "
    "starting quality.",
]


# ===========================================================================
# Scoring — vendored verbatim from scripts/bench_adapters.py (the source of
# truth). NOT imported: bench_adapters loads transformers+peft+CUDA at module
# top, which we don't want to drag into this in-process tiny-model bench.
# ===========================================================================

# Diverse grammatical TP seeds (bench_adapters.SEEDS).
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

_WORD_RE = re.compile(r"[a-z]+")  # bench_adapters' lowercase word regex


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


# ===========================================================================
# New metric helpers
# ===========================================================================

_CASE_WORD_RE = re.compile(r"[A-Za-z]+")
_CORE_VOCAB = set(WORD_TO_CODEPOINT)  # 137 words incl. particles


def vocab_metrics(texts: list[str]) -> dict:
    """Lexical richness vs. the 137-word core TP vocab. Capitalized tokens are
    treated as proper names (excluded from OOV, as in `_filter_sentence`)."""
    pooled_core: set[str] = set()
    all_forms: set[str] = set()
    per_gen = []
    oov = total = 0
    for t in texts:
        if not t.strip():
            continue
        gen_core: set[str] = set()
        for w in _CASE_WORD_RE.findall(t):
            lw = w.lower()
            all_forms.add(lw)
            if w[0].isupper():
                continue  # proper name — not a vocab/OOV signal
            total += 1
            if lw in _CORE_VOCAB:
                gen_core.add(lw)
                pooled_core.add(lw)
            else:
                oov += 1
        per_gen.append(len(gen_core) / len(_CORE_VOCAB))
    return {
        "vocab_coverage_core_pooled": len(pooled_core) / len(_CORE_VOCAB),
        "vocab_coverage_core_per_gen": sum(per_gen) / max(len(per_gen), 1),
        "oov_word_rate": oov / max(total, 1),
        "distinct_word_forms": len(all_forms),
    }


def loop_metrics(texts: list[str]) -> dict:
    """Degeneration/looping rate: a generation loops if its most frequent word
    exceeds 40% of its tokens (`_filter_sentence` rule 3, applied doc-level)."""
    loops = 0
    n = 0
    fracs = []
    for t in texts:
        words = [w.lower() for w in _CASE_WORD_RE.findall(t)]
        if len(words) < 3:
            continue
        n += 1
        frac = max(Counter(words).values()) / len(words)
        fracs.append(frac)
        if frac > 0.4:
            loops += 1
    return {
        "loop_rate": loops / max(n, 1),
        "mean_max_unigram_frac": sum(fracs) / max(len(fracs), 1),
        "n_scored": n,
    }


def gen_health(texts: list[str]) -> dict:
    """score_texts + vocab + loop, merged — the per-setting metric block."""
    return {**score_texts(texts), **vocab_metrics(texts), **loop_metrics(texts)}


# ===========================================================================
# Data / perplexity
# ===========================================================================

def build_capped_stream(sp, texts: list[str], bos_id: int, eos_id: int,
                        max_doc_tokens: int | None) -> torch.Tensor:
    """Tokenize + BOS/EOS-frame each doc (matching train_student._build_stream),
    truncating any single doc to `max_doc_tokens` so one giant doc can't
    dominate a per-source perplexity."""
    ids: list[int] = []
    for t in texts:
        toks = sp.encode(t, out_type=int)
        if max_doc_tokens is not None:
            toks = toks[:max_doc_tokens]
        ids.append(bos_id)
        ids.extend(toks)
        ids.append(eos_id)
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def exact_ppl(model, stream: torch.Tensor, block_size: int,
              device: torch.device, stride: int | None = None) -> dict:
    """Exact, full-coverage perplexity via the standard fixed-stride sliding
    window: every target token is scored exactly once with maximal left context.

    The model does NOT shift internally (train_student passes pre-shifted x,y),
    so we feed inputs = stream[b:e], targets = stream[b+1:e+1] and only count
    the `e - prev_end` newly-covered targets per window."""
    stride = stride or block_size // 2
    seq_len = stream.numel()
    if seq_len < 2:
        return {"nll_per_token": float("nan"), "bits_per_token": float("nan"),
                "ppl": float("nan"), "n_tokens": 0}
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + block_size, seq_len - 1)  # need a target after `end`
        trg_len = end - prev_end                     # newly-covered targets
        if trg_len <= 0:
            if end == seq_len - 1:
                break
            continue
        inp = stream[begin:end].unsqueeze(0).to(device)
        tgt = stream[begin + 1:end + 1].clone()
        tgt[:-trg_len] = -100                          # count only the new tail
        tgt = tgt.unsqueeze(0).to(device)
        logits, _ = model(inp)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), tgt.view(-1),
            ignore_index=-100, reduction="sum",
        )
        nll_sum += loss.item()
        n_tokens += int((tgt != -100).sum().item())
        prev_end = end
        if end == seq_len - 1:
            break
    nll = nll_sum / max(n_tokens, 1)
    return {
        "nll_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "ppl": math.exp(nll) if n_tokens else float("nan"),
        "n_tokens": n_tokens,
    }


# ===========================================================================
# Generation
# ===========================================================================

PROMPT_CATEGORIES: dict[str, list[str]] = {
    # long-form prose openers → story / prose-generation target
    "narrative_opener": [
        "tenpo pini la", "tenpo suno ni la mi", "jan lili li lon ma kasi.",
        "telo sewi li kama lon tenpo pimeja.",
    ],
    # personal mi/sina register → chat / diary target
    "mi_sina_personal": [
        "mi pilin pona tan ni:", "sina sona ala sona e ni:", "mi wile e ni:",
        "tenpo ni la mi",
    ],
    # question completion → constrained / IME target
    "seme_question": [
        "ni li seme?", "jan seme li", "sina wile e seme?", "ona li seme?",
    ],
    # encyclopedic register (tokwiki) → factual / definition target
    "topical_encyclopedic": [
        "toki pona li", "ma li", "kulupu pi jan mute li", "ilo ni li",
    ],
    # imperative frame — near negative-control (a base LM never saw instructions)
    "instruction_ish": [
        "o pali e", "o toki e", "o lukin e", "o pana e moku tawa",
    ],
}


@torch.no_grad()
def generate_one(model, sp, prompt: str, *, bos_id: int, eos_id: int,
                 max_new_tokens: int, temperature: float, top_k: int | None,
                 device: torch.device) -> tuple[str, str]:
    """Generate from BOS(+prompt). Returns (full_text, continuation_only).
    BOS is stripped; for continuations the prompt tokens are stripped too so
    metrics measure what the model *added*."""
    prompt_ids = sp.encode(prompt, out_type=int) if prompt else []
    idx = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k,
                         eos_token_id=eos_id)
    ids = out[0].tolist()
    if ids and ids[0] == bos_id:
        ids = ids[1:]
    full = sp.decode(ids).strip()
    cont = sp.decode(ids[len(prompt_ids):]).strip()
    return full, cont


def _temp_key(t: float) -> str:
    return "greedy" if t == 0 else f"{t:g}"


@torch.no_grad()
def throughput(model, sp, device: torch.device, *, n_tokens: int, iters: int,
               prompt: str = "mi") -> dict:
    bos_id, eos_id = sp.bos_id(), sp.eos_id()
    ids = torch.tensor([[bos_id] + sp.encode(prompt, out_type=int)],
                       dtype=torch.long, device=device)
    model.generate(ids, max_new_tokens=16, temperature=0.8, top_k=50,
                   eos_token_id=None)  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    rates = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.monotonic()
        out = model.generate(ids, max_new_tokens=n_tokens, temperature=0.8,
                             top_k=50, eos_token_id=None)  # None → full length
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.monotonic() - t0
        rates.append((out.size(1) - ids.size(1)) / dt)
    rates.sort()
    return {"tok_per_s_median": rates[len(rates) // 2], "n_tokens": n_tokens,
            "iters": iters}


# ===========================================================================
# Interpretation — rule-based fine-tune-readiness verdicts
# ===========================================================================

def interpret(A: dict, B: dict, C: dict, D: dict) -> dict:
    """Map measured metrics to candidate fine-tune directions."""
    def lvl(ready, promising):
        return "ready" if ready else ("promising" if promising else "needs-work")

    s09 = B["seed"].get("0.9") or next(iter(B["seed"].values()))  # sampled ref
    # Greedy reference = BOS-only (free) greedy: seeded greedy returns empty
    # continuations because the SEEDS are already complete sentences (the model
    # correctly emits EOS), which would make the metrics degenerate.
    greedy = B["bos"].get("greedy", s09)
    held_nll = A["tatoeba_heldout_200"]["nll_per_token"]
    cuda_tps = (D.get("cuda") or {}).get("tok_per_s_median")

    def cat(name, temp="0.9", field="strict_pass_rate", default=0.0):
        c = C.get(name, {})
        d = c.get(temp) or (next(iter(c.values())) if c else {})
        return d.get(field, default)

    mi_strict = cat("mi_sina_personal")
    narr_keep = cat("narrative_opener", field="sent_keep_rate")
    instr_strict = cat("instruction_ish")

    out = {}
    out["conversational_chat"] = {
        "readiness": lvl(mi_strict >= 0.6 and s09["sent_keep_rate"] >= 0.8,
                         mi_strict >= 0.4),
        "evidence": f"mi/sina strict={mi_strict:.2f}, sampled sent_keep={s09['sent_keep_rate']:.2f}",
    }
    out["story_prose_generation"] = {
        "readiness": lvl(narr_keep >= 0.7 and s09["loop_rate"] <= 0.15,
                         narr_keep >= 0.5),
        "evidence": f"narrative sent_keep={narr_keep:.2f}, loop_rate@0.9={s09['loop_rate']:.2f}",
    }
    out["autocomplete_ime"] = {
        "readiness": lvl(held_nll <= 2.3 and s09["oov_word_rate"] <= 0.02,
                         held_nll <= 3.0),
        "evidence": f"heldout nll={held_nll:.2f} (ppl={math.exp(held_nll):.1f}), "
                    f"oov={s09['oov_word_rate']:.3f}"
                    + (f", {cuda_tps:.0f} tok/s" if cuda_tps else ""),
    }
    out["grammar_style_correction"] = {
        "readiness": lvl(s09["strict_pass_rate"] >= 0.7, s09["strict_pass_rate"] >= 0.5),
        "evidence": f"sampled strict={s09['strict_pass_rate']:.2f}, greedy strict={greedy['strict_pass_rate']:.2f}",
    }
    out["constrained_templated_gen"] = {
        "readiness": lvl(greedy["strict_pass_rate"] >= 0.5 and greedy["loop_rate"] <= 0.3,
                         greedy["strict_pass_rate"] >= 0.3),
        "evidence": f"greedy strict={greedy['strict_pass_rate']:.2f}, greedy loop={greedy['loop_rate']:.2f}",
    }
    out["instruction_following"] = {
        "readiness": "promising" if instr_strict >= 0.5 else "needs-work",
        "evidence": f"instruction-ish strict={instr_strict:.2f} — base never saw "
                    "instructions; SFT required",
    }
    out["en_to_tp_translation"] = {
        "readiness": "promising-as-decoder",
        "evidence": f"TP-side fluency (sampled sent_keep={s09['sent_keep_rate']:.2f}); "
                    "needs an English encoder/conditioning the base lacks",
    }
    out["tp_to_en_translation"] = {
        "readiness": "not-supported-by-base",
        "evidence": "base only ever produced Toki Pona; no English in vocab/training",
    }
    return out


# ===========================================================================
# Per-checkpoint driver
# ===========================================================================

def load_model(ckpt_path: Path, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**state["cfg"])
    model = TinyTPLM(cfg).to(device=device, dtype=torch.float32)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, state


def run_checkpoint(label: str, ckpt_path: Path, sp, real_all, device, args,
                   gens_out) -> dict:
    print(f"\n{'='*70}\n=== {label}  ({ckpt_path.relative_to(REPO_ROOT)})\n{'='*70}",
          flush=True)
    model, cfg, state = load_model(ckpt_path, device)
    assert sp.vocab_size() == cfg.vocab_size, \
        f"tokenizer vocab {sp.vocab_size()} != model vocab {cfg.vocab_size}"
    bos_id, eos_id = sp.bos_id(), sp.eos_id()
    bs = cfg.block_size
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_step, ckpt_val = state.get("step"), state.get("val_loss")
    print(f"  {cfg.n_layer}L×{cfg.n_embd}d, {n_params/1e6:.1f}M params, "
          f"ctx={bs}, stored val_loss={ckpt_val} @ step {ckpt_step}",
          flush=True)

    # --- A. Perplexity -----------------------------------------------------
    print("  [A] perplexity…", flush=True)
    A: dict = {}
    held = real_all[:HELDOUT_N]
    stream = build_capped_stream(sp, held, bos_id, eos_id, args.ppl_doc_max_tokens)
    A["tatoeba_heldout_200"] = {**exact_ppl(model, stream, bs, device),
                                "in_sample": False, "n_docs": len(held)}
    for src in INSAMPLE_SOURCES:
        a, b = SOURCE_SPANS[src]
        if src == "tatoeba":
            a = HELDOUT_N  # disjoint from the held-out slice
        docs = real_all[a:b][:args.ppl_docs_per_source]
        if not docs:
            continue
        stream = build_capped_stream(sp, docs, bos_id, eos_id, args.ppl_doc_max_tokens)
        key = f"{src}_insample" if src == "tatoeba" else src
        A[key] = {**exact_ppl(model, stream, bs, device), "in_sample": True,
                  "n_docs": len(docs), "doc_truncated_to": args.ppl_doc_max_tokens}

    # --- B. Free generation ------------------------------------------------
    print("  [B] free generation…", flush=True)
    B: dict = {"bos": {}, "seed": {}}
    temps = [0.0] + args.temps
    for temp in temps:
        tk = None if temp == 0 else args.top_k
        # BOS-only
        bos_texts = []
        for j in range(args.n_bos):
            torch.manual_seed(args.seed + j)
            full, _ = generate_one(model, sp, "", bos_id=bos_id, eos_id=eos_id,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=temp, top_k=tk, device=device)
            bos_texts.append(full)
            gens_out.append({"checkpoint": label, "section": "B_free",
                             "setting": "bos", "temp": temp, "prompt": "",
                             "sample": j, "text": full, "continuation": full})
        B["bos"][_temp_key(temp)] = gen_health(bos_texts)
        # seeded (continuation scored on full text, like the adapter bench)
        seed_texts = []
        reps = 1 if temp == 0 else args.n_sampled
        for si, seed in enumerate(SEEDS[:args.n_seeds]):
            for k in range(reps):
                torch.manual_seed(args.seed + 100 * si + k)
                full, cont = generate_one(model, sp, seed, bos_id=bos_id,
                                          eos_id=eos_id,
                                          max_new_tokens=args.max_new_tokens,
                                          temperature=temp, top_k=tk, device=device)
                seed_texts.append(cont)
                gens_out.append({"checkpoint": label, "section": "B_free",
                                 "setting": "seed", "temp": temp, "prompt": seed,
                                 "sample": k, "text": full, "continuation": cont})
        B["seed"][_temp_key(temp)] = gen_health(seed_texts)

    # --- C. Prompted-continuation capability map ---------------------------
    print("  [C] prompted continuation…", flush=True)
    C: dict = {}
    for cat_name, prompts in PROMPT_CATEGORIES.items():
        C[cat_name] = {}
        for temp in args.cont_temps:
            conts = []
            for pi, prompt in enumerate(prompts):
                for k in range(args.n_cont):
                    torch.manual_seed(args.seed + 31 * pi + k)
                    full, cont = generate_one(
                        model, sp, prompt, bos_id=bos_id, eos_id=eos_id,
                        max_new_tokens=args.max_new_tokens, temperature=temp,
                        top_k=args.top_k, device=device)
                    conts.append(cont)
                    gens_out.append({"checkpoint": label, "section": "C_prompted",
                                     "setting": cat_name, "temp": temp,
                                     "prompt": prompt, "sample": k, "text": full,
                                     "continuation": cont})
            C[cat_name][_temp_key(temp)] = gen_health(conts)

    # --- D. Throughput -----------------------------------------------------
    print("  [D] throughput…", flush=True)
    D: dict = {}
    dev_name = "cuda" if device.type == "cuda" else "cpu"
    D[dev_name] = {**throughput(model, sp, device, n_tokens=args.throughput_tokens,
                                iters=args.throughput_iters), "dtype": "fp32"}
    if device.type == "cuda":  # the real inference dtype
        model_bf16 = model.to(torch.bfloat16)
        D["cuda_bf16"] = {**throughput(model_bf16, sp, device,
                                       n_tokens=args.throughput_tokens,
                                       iters=args.throughput_iters), "dtype": "bf16"}
    if args.also_cpu and device.type == "cuda":
        cpu_model, _, _ = load_model(ckpt_path, torch.device("cpu"))
        D["cpu"] = {**throughput(cpu_model, sp, torch.device("cpu"),
                                 n_tokens=min(args.throughput_tokens, 128),
                                 iters=max(1, args.throughput_iters // 2)),
                    "dtype": "fp32"}
        del cpu_model

    interp = interpret(A, B, C, D)

    del model, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
            "step": ckpt_step, "stored_val_loss": ckpt_val,
            "cfg": {"n_layer": cfg.n_layer, "n_embd": cfg.n_embd,
                    "n_head": cfg.n_head, "block_size": cfg.block_size,
                    "n_params": n_params},
            "A_perplexity": A, "B_free": B, "C_prompted": C,
            "D_throughput": D, "interpretation": interp}


# ===========================================================================
# Printing
# ===========================================================================

def _pct(x):
    return f"{x*100:.0f}%"


def print_report(summary: dict):
    print("\n" + "#" * 72)
    print("# STUDENT BENCH — fine-tune readiness")
    print("#" * 72)
    print("\nCAVEATS:")
    for c in summary["caveats"]:
        print(f"  • {c}")

    for label, r in summary["per_checkpoint"].items():
        cfg = r["cfg"]
        print(f"\n{'━'*72}\n▶ {label}  —  {cfg['n_layer']}L×{cfg['n_embd']}d, "
              f"{cfg['n_params']/1e6:.1f}M params   ({r['checkpoint']})")

        # A
        print("\n  [A] Perplexity (exact strided-window NLL)")
        print("      reference: training real_loss≈2.08 → ppl≈8.0 (tatoeba held-out)")
        hdr = ("source", "n_tok", "bits/tok", "ppl", "nll", "in-sample", "Δnll")
        print("      " + " ".join(f"{h:>11}" for h in hdr))
        held_nll = r["A_perplexity"]["tatoeba_heldout_200"]["nll_per_token"]
        for src, a in r["A_perplexity"].items():
            dn = a["nll_per_token"] - held_nll
            print("      " + " ".join(f"{v:>11}" for v in (
                f"{src:>11}", f"{a['n_tokens']:>11}", f"{a['bits_per_token']:.3f}",
                f"{a['ppl']:.2f}", f"{a['nll_per_token']:.3f}",
                "yes" if a["in_sample"] else "NO·held",
                f"{dn:+.2f}")))

        # B
        for setting in ("bos", "seed"):
            print(f"\n  [B] Free generation — {setting}")
            hdr = ("temp", "doc", "sent", "strict", "d-1", "d-2", "d-3",
                   "avg_w", "core/gen", "oov", "loop")
            print("      " + " ".join(f"{h:>8}" for h in hdr))
            for tk, m in r["B_free"][setting].items():
                print("      " + " ".join(f"{v:>8}" for v in (
                    f"{tk:>8}", _pct(m["doc_keep_rate"]), _pct(m["sent_keep_rate"]),
                    _pct(m["strict_pass_rate"]), f"{m['distinct_1']:.2f}",
                    f"{m['distinct_2']:.2f}", f"{m['distinct_3']:.2f}",
                    f"{m['avg_words']:.0f}", _pct(m["vocab_coverage_core_per_gen"]),
                    f"{m['oov_word_rate']:.3f}", _pct(m["loop_rate"]))))

        # C — sorted by strict (best temp per category)
        print("\n  [C] Prompted-continuation capability map (sorted by strict, temp=0.9)")
        hdr = ("category", "sent", "strict", "d-2", "avg_w", "loop")
        print("      " + " ".join(f"{h:>16}" if i == 0 else f"{h:>8}"
                                  for i, h in enumerate(hdr)))
        ct = "0.9" if "0.9" in next(iter(r["C_prompted"].values())) \
            else list(next(iter(r["C_prompted"].values())))[0]
        rows = sorted(r["C_prompted"].items(),
                      key=lambda kv: kv[1][ct]["strict_pass_rate"], reverse=True)
        for cat_name, temps in rows:
            m = temps[ct]
            print("      " + f"{cat_name:>16} " + " ".join(f"{v:>8}" for v in (
                _pct(m["sent_keep_rate"]), _pct(m["strict_pass_rate"]),
                f"{m['distinct_2']:.2f}", f"{m['avg_words']:.0f}",
                _pct(m["loop_rate"]))))

        # D
        d = r["D_throughput"]
        parts = [f"{k}={v['tok_per_s_median']:.0f} tok/s ({v['dtype']})"
                 for k, v in d.items()]
        print(f"\n  [D] Throughput (batch-1 decode): " + "  ".join(parts))

        # E
        print("\n  [E] Fine-tune readiness")
        for direction, v in r["interpretation"].items():
            print(f"      [{v['readiness']:>22}] {direction:<26} — {v['evidence']}")


# ===========================================================================
# Main
# ===========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS),
                    help="labels from DEFAULT_CHECKPOINTS, or paths to .pt")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--ppl-docs-per-source", type=int, default=300)
    ap.add_argument("--ppl-doc-max-tokens", type=int, default=1024)
    ap.add_argument("--n-bos", type=int, default=20)
    ap.add_argument("--n-sampled", type=int, default=3, help="sampled gens per seed per temp")
    ap.add_argument("--n-seeds", type=int, default=len(SEEDS))
    ap.add_argument("--n-cont", type=int, default=2, help="continuations per prompt per temp")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.7, 0.9, 1.0],
                    help="sampled temps for section B (greedy always included)")
    ap.add_argument("--cont-temps", type=float, nargs="+", default=[0.7, 0.9],
                    help="temps for section C")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--throughput-tokens", type=int, default=256)
    ap.add_argument("--throughput-iters", type=int, default=5)
    ap.add_argument("--also-cpu", action="store_true",
                    help="also measure CPU decode throughput (the laptop story)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-summary", type=Path,
                    default=BENCH_DIR / "student_bench_summary.json")
    ap.add_argument("--out-gens", type=Path, default=BENCH_DIR / "student_gens.jsonl")
    ap.add_argument("--smoke", action="store_true",
                    help="fast sanity run on 1 checkpoint")
    args = ap.parse_args(argv)

    if args.smoke:
        args.checkpoints = args.checkpoints[:1]
        args.ppl_docs_per_source = 20
        args.ppl_doc_max_tokens = 256
        args.n_bos = 4
        args.n_sampled = 1
        args.n_seeds = 4
        args.n_cont = 1
        args.temps = [0.9]
        args.cont_temps = [0.9]
        args.throughput_tokens = 64
        args.throughput_iters = 1
        args.also_cpu = False

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    real_all = _load_real_texts(args.corpus)
    print(f"Loaded {len(real_all):,} real docs from {args.corpus.name}; "
          f"device={device}", flush=True)

    summary = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device), "seed": args.seed, "caveats": CAVEATS,
        "per_checkpoint": {},
    }
    gens_out: list[dict] = []

    for spec in args.checkpoints:
        if spec in DEFAULT_CHECKPOINTS:
            label, ckpt = spec, RUNS_DIR / DEFAULT_CHECKPOINTS[spec]
        else:
            ckpt = Path(spec)
            label = ckpt.parent.name
        if not ckpt.exists():
            print(f"  ! {label}: checkpoint not found at {ckpt}; skipping",
                  file=sys.stderr)
            continue
        summary["per_checkpoint"][label] = run_checkpoint(
            label, ckpt, sp, real_all, device, args, gens_out)

    with args.out_gens.open("w", encoding="utf-8") as f:
        for g in gens_out:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print_report(summary)
    print(f"\n→ summary: {args.out_summary.relative_to(REPO_ROOT)}")
    print(f"→ raw gens: {args.out_gens.relative_to(REPO_ROOT)} ({len(gens_out)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
