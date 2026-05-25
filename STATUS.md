# waso-sona — handoff status

**Last updated**: 2026-05-25
**Branch with this doc + WIP code**: `claude/handoff-status`

This is a snapshot for a fresh Claude conversation to pick up. The local session's automemory under `~/.claude/projects/-home-thomasn-Documents-toki-pona-waso-sona/memory/` is *not* portable to claude.ai/code, so the relevant facts from there are inlined below.

---

## Project goal

A small monolingual Toki Pona language model trained locally on the user's RTX 5060 (Blackwell sm_120, 8 GB VRAM), output in **sitelen pona UCSUR** glyphs (not Latin script).

**Two-stage pipeline as currently scoped:**

1. **Generate / augment a Toki Pona text corpus** (Latin script) by fine-tuning Gemma 4 e2b on real TP, then using it as a paraphrase/continuation engine over the real corpus.
2. **Convert Latin → UCSUR mechanically**, tokenize with the repo's existing UCSUR tokenizer, and train a small from-scratch nanoGPT-style transformer that fits in 8 GB.

Stage 2's tokenizer + training are out of scope for the work below. Everything here is about getting stage 1 to a usable point.

---

## What's in `main` already

| PR | Title | What it does |
|---|---|---|
| #5 | Corpus filter + Gemma 4 feasibility benchmark | `scripts/filter_corpus.py` (doc-level non-TP-letter filter, drops contamination 13.6% → 1.0% UNK). `scripts/bench_gemma.py` (throughput + quality benchmark over the three local Ollama Gemma 4 variants). |
| #6 | `scripts/augment_corpus.py` | Paraphrase + continuation pipeline via Ollama. **Source of truth for the inference-time prompt shape**: `_paraphrase_prompt`, `_continuation_prompt`, `FEWSHOT_EXAMPLES`. Strict per-sentence filter (`_filter_sentence`). |
| #7 | QLoRA deps | Adds transformers/peft/bitsandbytes/accelerate/datasets/huggingface_hub to `pyproject.toml`. |
| #8 | `scripts/build_sft_dataset.py` | Produces `data/processed/sft_train.jsonl` (9,305) + `sft_val.jsonl` (501) in HF messages format. User-side prompts are verbatim copies of `augment_corpus._continuation_prompt` so train/inference shapes match. Token p50/90/99/max = 213 / 271 / 323 / 402. |

Plus a small uncommitted PR (`claude/qlora-deps` was #7; this handoff branch will be PR #9).

---

## Key findings carried forward (the non-obvious bits)

### Gemma 4 de-novo Toki Pona is unusable

Bench on 2026-05-24 (`scripts/bench_gemma.py`, artifacts in gitignored `data/bench/`):

- **Throughput**: best 25.7 tok/s on `gemma4:e2b` (the 2B-active MatFormer variant) at concurrency 2. Larger variants (4B, 8B) are slower because they spill to CPU. At 25.7 tok/s, 100 M tokens ≈ 45 days continuous; 1 B is unreachable on this hardware.
- **Quality**: existing letter/vocab filters pass 95–100 % but the real acceptance rate is ~10–25 %. Failure modes: `wawa`-loop mode collapse ("wawa lili li lon wawa wawa"), invented TP-shaped words (`wulo`, `lepa`, `papu`, `katek`), foreign drop-ins (`grass`, `wino`, `pikinini`).
- **Implication**: de-novo synthetic generation is off the table. Stage 1 must be **augmentation only** — paraphrase/continuation anchored on real TP text. The student LM is bounded by Gemma's TP fluency; bigger Gemma doesn't help (the 8B introduces *more* foreign drop-ins, not fewer).
- **Fine-tune hypothesis (still untested)**: QLoRA-fine-tuning `google/gemma-4-E2B-it` on the real corpus might push Gemma's TP quality up enough that the augment pipeline produces usable output. Whether this actually works has not been verified.

### Preprocessing state

- Real corpus: `data/processed/corpus.filtered.jsonl`, 79,364 docs, 8.23 M chars. **97 % is Tatoeba single sentences** (avg 37 chars); 3 % is poki/nltk-tp paragraphs (avg 2,100–2,619 chars). One outlier doc is 322 K chars (unsplit blob).
- The doc-level filter (`scripts/filter_corpus.py`) brought UNK rate from 13.6 % → 1.0 % and reduced 5.87 M → 2.54 M tokens.
- **Tokiponizer still TBD** (proper-noun → TP-phonotactics mapper, e.g. "Toronto" → "Tolonto"). Only matters for stage 2 (UCSUR encoding); not on the critical path for the Gemma fine-tune.
- `nltk-tp` source was flagged as suspect (more invented/non-pu vocab than others).

### Gemma 4 chat template — silent loss-mask trap

Gemma 4's stock chat template does **not** include `{% generation %}` markers, so TRL's `SFTConfig(assistant_only_loss=True)` silently produces an all-zero mask and would train on the full sequence (including the huge few-shot user prompt). Fix that's wired into `scripts/train_qlora.py`: override the tokenizer's chat template with a minimal one that renders byte-identically to the stock template but adds the markers around the model turn. Verified — see `--check-collator` in the script.

---

## Where the QLoRA fine-tune is stuck

Plan file (machine-readable): `~/.claude/plans/valiant-squishing-dove.md` on the user's machine.

The script `scripts/train_qlora.py` (in this commit, not in `main` yet) implements that plan: QLoRA with nf4 + LoRA r=16/alpha=32 on language-model linears only, TRL `SFTTrainer`, val-loss early-stop, TB scalars, sample-gen callback gated behind `--sample-gen`, `--smoke` mode for 50-step pipeline validation.

**`--check-collator` passes** (5/5 examples, mask exactly covers the assistant response).

**`--smoke` fails OOM** on the 8 GB card:

| Component | VRAM |
|---|---|
| Quantized model resident (text decoder nf4 + embed_tokens bf16 + lm_head bf16) | ~6.47 GB |
| bnb scratch / CUDA context | ~0.17 GB |
| Desktop compositor + Firefox + Steam helper (other process) | ~0.83 GB |
| **Total before activations** | **~7.47 GB on a 7.52 GB usable card** |

Effectively zero headroom for the 200 MB logits tensor (vocab 262,144 × seq 256 × bf16 = ~100 MB, plus a softcap-divide copy doubles it). Every attempt OOMs by 50–100 MiB.

**Mitigations already applied in the script:**
- `device_map={"": 0}` (not `"auto"`, which silently CPU-offloaded layers)
- Replaced `prepare_model_for_kbit_training` with a minimal kbit prep — peft's default upcasts every bf16 param to fp32, which on vocab=262144 alone OOMs (1.6 GB embed_tokens copy)
- Manually moved `vision_tower` / `audio_tower` / `embed_vision` / `embed_audio` to CPU (saves ~260 MB)
- `gradient_checkpointing` with `use_reentrant=False`, `paged_adamw_8bit`, `bf16=True`
- `tokenizer.truncation_side = "left"` so over-length sequences truncate the user prompt, never the assistant response
- `max_seq_length` default lowered from 1024 → 256
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `exclude_modules=".*(vision_tower|audio_tower).*"` in the LoraConfig (their custom `Gemma4ClippableLinear` wrapper isn't recognized by PEFT)

**What the user agreed to try but didn't get to yet:** close Firefox + reboot first to free ~500–800 MB of compositor/browser VRAM, then re-run `--smoke`. Whether that's actually enough headroom for forward+backward at seq=256 is untested — it'd be tight (resident 6.5 GB + activations ~0.8 GB = 7.3 GB on a 7.7 GB usable card after closing).

---

## Decisions to rethink with the next conversation

These are the questions the user wants to revisit before more code is written:

1. **Is local QLoRA fine-tuning on this 8 GB card the right approach?**
    - Option A: fight for every MB on-device (close apps, smaller seq, chunked CE loss via Liger Kernel, etc.). Risky — might still OOM during eval (which pads to longest in batch) or during the sample-gen callback (KV cache spikes 200–500 MB).
    - Option B: rent a 24 GB A10G or similar for the fine-tune (~$0.50–1/hr, full run in 4–6 h, ~$3–6 total). No memory pressure, faster, user keeps using their machine. Adapter is small (~150 MB), trivial to bring back home.
    - Option C: skip the Gemma fine-tune entirely. Just use few-shot prompting at inference time in `augment_corpus.py` (already wired), accept the ~20 % quality and 25 tok/s throughput, and generate however much TP that produces over a few days.

2. **Is the augment-anchored corpus expansion actually worth doing?** The real corpus is 2.54 M tokens post-filter. The augment pipeline + a fine-tuned Gemma would maybe 10–20× this to ~30–50 M tokens. Stage 2 (the from-scratch nanoGPT for UCSUR) on an 8 GB card needs maybe 30–100 M tokens, so the math arguably checks out — but the user should re-evaluate whether the quality ceiling makes this worthwhile vs. just training stage 2 on the 2.54 M real tokens directly.

3. **Should the fine-tune use instruction-tuning at all, or a continuation/PT-style setup?** `gemma-4-E2B-it` is already instruction-tuned; the script does SFT in the chat format to make it follow the augment prompts more faithfully. An alternative is continuation-style PT on a flat TP text stream — simpler, no prompt template at all, but loses the instruction-following the augment pipeline depends on.

4. **Sequence length & lm_head.** Gemma 4's vocab=262,144 makes the unquantized lm_head (805 MB bf16) the single biggest activation cost. Worth investigating chunked cross-entropy or vocab pruning to TP-relevant tokens — but only if option A above is the chosen path.

---

## Repo layout cheat-sheet

```
scripts/
  fetch_data.py            # corpora → data/raw/, data/processed/corpus.jsonl
  filter_corpus.py         # PR #5; doc-level non-TP-letter filter
  bench_gemma.py           # PR #5; Gemma 4 throughput + quality probe
  augment_corpus.py        # PR #6; paraphrase + continuation via Ollama
  build_sft_dataset.py     # PR #8; messages-format SFT dataset
  train_qlora.py           # this handoff branch; OOM-blocked on 8 GB
tokenizer/
  glyphs.py                # PU + KU SULI vocab → UCSUR codepoints
  syllabify.py / tokenizer.py
data/                      # gitignored; raw, processed, bench, training/runs
pyproject.toml             # cu128 torch index, qlora + trl deps
```

Memory files (machine-local, not in repo, *not* visible to claude.ai/code):

```
~/.claude/projects/-home-thomasn-Documents-toki-pona-waso-sona/memory/
  project_goal.md
  preprocessing_pipeline.md
  synthetic_corpus_strategy.md
```

Plan file (machine-local):

```
~/.claude/plans/valiant-squishing-dove.md   # latest = the QLoRA plan
```

---

## How to resume on a fresh machine (if needed)

```sh
git clone git@github.com:thomasN4/waso-sona.git
cd waso-sona
uv venv --python 3.13
uv pip install -e .
python scripts/fetch_data.py              # ~5 min, populates data/raw + data/processed
python scripts/filter_corpus.py           # produces data/processed/corpus.filtered.jsonl
python scripts/build_sft_dataset.py       # produces sft_{train,val}.jsonl, ~10 s
# Model download (10 GB, ~30 min):
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.md','tokenizer*'])"
python scripts/train_qlora.py --check-collator   # offline mask audit
python scripts/train_qlora.py --smoke            # 50-step pipeline validation (OOMs on 8 GB)
```
