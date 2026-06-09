# Pipeline

The deliverable is a **tiny Toki Pona language model trained from scratch**,
runnable locally on consumer hardware (developed on an RTX 5060, 8 GB VRAM).
The obstacle is data: the real, human-written TP corpus is far too small to
train a from-scratch LM well. So Gemma 4 is brought in as a **teacher** —
fine-tuned on the real corpus, then used to *augment* it — and the tiny model
is trained from scratch on the enlarged corpus.

Three stages:

1. **Teacher (built).** QLoRA fine-tune Gemma 4 on the real TP corpus so it
   produces fluent, grounded Toki Pona. Most of this document details this
   stage — fetch, filter, SFT-dataset build, optional vocab-pruning, training.
2. **Augment.** Use the fine-tuned Gemma to expand the corpus. *v1 (done)*
   paraphrases + continues real sentences — but that only recombines existing
   TP and saturated *below* the real corpus's size (see "Stage 2 v2"). *v2
   (done)* translates diverse English into TP via a `waso-translator`, injecting
   265k new-content TP sentences (2.63 M words).
3. **Student (built — the actual product).** Train the tiny TP LM **from
   scratch** (nanoGPT-style decoder). Training on the real+translated+synthetic
   mix dropped the honest `real_loss` from ~4.0 (synthetic-only) → 2.16 → **2.08**
   with a balance-corrected 15 M-param model (see "Student runs").

```
  Stage 1 — TEACHER (built)
    public TP corpora
        │  fetch_data.py
        ▼
    data/raw/  (7 sources)
        │  filter_corpus.py
        ▼
    corpus.filtered.jsonl
        │  build_sft_dataset.py
        ▼
    sft_train.jsonl / sft_val.jsonl
        │  train_qlora.py        (optional: prune_vocab.py first)
        ▼
    fine-tuned Gemma 4 adapter  (models/runs/qlora-<UTC>/final/, ~96 MB)

  Stage 2 — AUGMENT
        │  v1 (done):    augment_corpus.py    — paraphrase + continuation of real TP
        │  v2 (done):    translate_corpus.py  — waso-translator turns English → TP
        ▼
    augmented corpus  (real + synthetic TP; v2 adds new-content TP from English)

  Stage 3 — STUDENT (built; the deliverable)
        │  train_tokenizer.py (SP BPE, vocab 2048)  →  spm.model
        │  train_student.py   (nanoGPT-style decoder, ~5M params)
        ▼
    tiny Toki Pona LM   ◀── the actual product

  Application layer (not in training; wraps the student model)
    UCSUR in  ──▶ sitelen.ucsur_to_latin ──▶ tiny LM ──▶ Latin out
    Latin out ──▶ sitelen.latin_to_ucsur ──▶ UCSUR display
```

Every model in the pipeline — **both** the Gemma teacher and the from-scratch
student — **only ever sees Latin-script Toki Pona**. UCSUR (sitelen pona) is a
display concern handled deterministically by the `sitelen/` package on the way
in and out — see [`sitelen/translate.py`](sitelen/translate.py).

**Current state:** All stages built. Stage 1 (the QLoRA teacher) is documented
across the "Data sources / preprocessing" and "Training: train_qlora.py"
sections. Stage 2 v1 (`augment_corpus.py` → `waso-teacher` over Ollama with
global dedup) produced **201,567 records / 1.82 M words ≈ 1.9 M SentencePiece
tokens** (the "~25.9 M tokens" in earlier notes was the teacher's gross
`eval_count`, ~93 % discarded by truncation + dedup — *not* trainable size) —
*smaller* than the 3.16 M-token real corpus and only ~1,800 word-forms, which
motivated **Stage 2 v2**. Stage 2 v2 (built 2026-06-05) trains a standalone
`waso-translator` and translates 300k simple English sentences into **265,477
new TP sentences / 2.63 M words** (see "Stage 2 v2 — Translation augmentation").
Stage 3 (`train_tokenizer.py`, `train_student.py`, `talk_to_student.py` — the
deliverable) first trained synthetic-only (2026-06-02, real_val ~4.0); the
real+translated+synthetic mix dropped the honest **real_loss to 2.16**, and a
balance-corrected 15 M-param model on the scaled corpus reached **2.08** — the
student now models real TP (see "Student runs" for the volume-vs-balance lesson).

---

## Data sources and preprocessing

### 1. Fetch — `scripts/fetch_data.py`

Pulls 7 Toki Pona corpora into `data/raw/<source>/` and normalizes them
into `data/processed/corpus.jsonl` (one JSON record per document) plus
`data/processed/sentences.txt` (deduped, sentence-split, TP-shaped lines).

| Source | What it is | Docs | Chars | Cleanliness |
|---|---|---:|---:|---|
| `poki` | Long-form prose (`kulupu-lapo/poki`) | 1,627 | 5.66 M | Mixed; some contamination |
| `nltk-tp` | Older crawled corpus (`davidar/nltk-tp`) | 1,825 | 4.44 M | Suspect — high non-TP letter ratio |
| `tatoeba` | Crowd-sourced single sentences | 76,941 | 3.11 M | Clean |
| `tokwiki` | `tok.wikipedia.org` page dump (XML, main-ns only, wikitext stripped) | 4,314 | 2.35 M | Clean; some residual `&nbsp;`/`[` artifacts caught by sentence filter |
| `toki-ramble` | `hecko-yes/toki-ramble` free-form prose (CC0) | 8 | 19 k | Clean original writing |
| `tp1k` | 1,000 hand-curated sentences | 400 | 18 k | Clean |
| `lipu` | Parallel translation set (TP side only) | 13 | 154 | Tiny but clean |

Stdlib only; pure download + light text normalization. Pure idempotent.
`tokwiki` is fetched as a bz2 XML dump and parsed with `xml.etree.iterparse`;
wikitext is stripped by a minimal regex pass (templates, refs, links,
tables, magic words) — anything that survives but isn't TP-shaped is
dropped downstream by `_looks_like_toki_pona`.

### 2. Filter — `scripts/filter_corpus.py`

Drops documents whose non-Toki-Pona-letter ratio (`b c d f g h q r v x y z`)
exceeds a threshold (default `0.05`). Writes `corpus.filtered.jsonl`. On the
current corpora this cuts the post-tokenization UNK rate from ~13.6 % →
~1.0 % and trims tokens from 5.87 M → 2.54 M.

Optionally also applies a sentence-level quality check (a targeted subset of
`augment_corpus._filter_sentence` — unknown lowercase word, double `li`,
`li e`, leading `li`, high uni/bigram repetition — but skipping length and
missing-predicate checks that are too strict for real prose). Enable with
`--min-sentence-pass-rate 0.5`; on the current corpora this drops an
additional ~550 docs (mostly YAML/CSS contamination in `poki` and
English-glossed headers in `nltk-tp`). Vocab is extended with `ali`,
`powe`, `majuna`, `po` to avoid false positives on historical Toki Pona.

### 3. SFT dataset builder — `scripts/build_sft_dataset.py`

Produces `data/processed/sft_train.jsonl` (~21,200 rows) and `sft_val.jsonl`
(~1,140 rows) in HF messages format. Three example modes:

- **Continuation pairs**: split a doc chunk into a prefix (1–3 sentences,
  ≥ 5 words) and a 1–2-sentence suffix; the user message is **the exact
  `_continuation_prompt(prefix)` shape that `scripts/augment_corpus.py`
  will send at inference**, including the same few-shot block. This keeps
  the model's train/inference distribution aligned.
- **Topic prompts**: one of 10 generic instruction templates ("Write a
  short Toki Pona story…") paired with a real chunk as the response.
- **Paraphrase pairs**: `_paraphrase_prompt(a)` → `b`, where `(a, b)` are
  TP sentences sharing an English Tatoeba translation
  (`scripts/build_paraphrase_pairs.py` → `paraphrase_pairs.jsonl`, ~11.7k
  ordered pairs). See the paraphrase caveat under "Stage 2 serving" — this
  mode trains, but faithful paraphrasing did **not** materialize.

Realized mode counts on the current corpora: ~7,980 continuation / ~4,440
topic / ~9,950 paraphrase. Each candidate response runs through
`augment_corpus._filter_sentence` (strict: zero-tolerance non-vocab,
double-`li` / `li e` rejection, repetition caps, missing-predicate check).
Dedup is mode-aware: paraphrase keys on `(prompt, response)` (clusters reuse
a target across sources), others on response only. Then drop any example
over `--max-tokens` (default 1024). Split is deterministic by id-hash
(`val_frac=0.05`).

Token-length percentiles on a typical build: p50 = 215, p90 = 263,
p99 = 319, max = 396.

---

## Vocabulary pruning (optional) — `scripts/prune_vocab.py`

Gemma 4's `vocab_size = 262,144` is the dominant VRAM cost on the 8 GB
card (the embedding/`lm_head` and the logits tensor all scale with it).
The Latin-only TP corpus touches only ~6 % of those tokens, so we can
slice the unused rows out of `embed_tokens` (tied to `lm_head`) and
filter the BPE tokenizer to match.

`prune_vocab.py` builds the keep-set by tokenizing the filtered corpus +
the SFT dataset (through the training chat template) + the
`augment_corpus` inference prompts, then unions in Gemma's special
tokens and all 256 byte-fallback tokens so any UTF-8 input still
tokenizes. On the current corpora:

| | value |
|---|---|
| vocab | 262,144 → **15,291** (94.2 % reduction) |
| BPE merges | 514,906 → 23,499 (invalid ones dropped) |
| corpus round-trip | 100 / 100 decoded identically |
| VRAM at model load | 6.74 GB → **5.98 GB** (−760 MB) |

Output lands in `models/base-pruned/` (a plain HF safetensors
checkpoint + pruned tokenizer). To train against it, pass
`--model-id models/base-pruned` to `train_qlora.py` — no other
flags change, and `chunked_causal_lm_loss` reads the new `vocab_size`
from the config automatically. The pruned tokenizer is saved into each
run's `final/` adapter dir, so `infer.py` picks it up; only the
`--base-model models/base-pruned` flag is needed at inference.

**Parity cost:** retraining at identical hyperparameters
(`qlora-20260528T022232Z` vs the un-pruned `qlora-20260527T223141Z`)
showed pruning alone costs ~0.01 `eval/loss` (0.7476 → 0.7578 best,
~1.3 % relative), consistent across the whole curve — the kept
embeddings were trained in the geometry of the full vocab and
fine-tuning recovers most but not all of it. The existing un-pruned
adapter is *not* compatible for production use on the pruned base (it
was trained against full-vocab logits); always retrain.

**Spending the freed headroom did *not* help here.** A bumped run on the
pruned base (`--max-seq-length 256 --lora-r 32 --lora-alpha 64`,
`qlora-20260528T085717Z`) fits comfortably (~1.2 GB VRAM to spare) but
showed no quality gain: a fixed-prompt generation comparison put it on
par with the parity and un-pruned adapters, all three limited by the
same repetition/degeneration ceiling. A follow-up with more steps and a
lower LR (`max_steps=6000`, `lr=1e-4`) descended *slower* toward the
same plateau and was stopped. Takeaway: at this corpus size the
bottleneck is data + repetition, not adapter capacity or context length,
so the canonical pruned model is the **parity** config (seq=160, r=16).

**Caveat on comparing loss across configs:** `eval/loss` is *not*
directly comparable across runs that differ in vocab size (15,291 vs
262,144 softmax) *or* in `max_seq_length` (different left-truncation →
different scored tokens; the dataset's median example exceeds both 160
and 256 tokens). The bumped run's much higher raw `eval/loss` (~1.42) is
largely this truncation artifact, **not** a real regression — which is
why the generation comparison, not the loss curve, is the arbiter here.

---

## The sitelen translator

`sitelen/` is application-layer only — never imported by `train_qlora.py`.
Two pure text-to-text functions:

- `latin_to_ucsur(text)` — words → glyphs; `.!?` → middle dot; `:` →
  middle colon; commas dropped; capitalized unknowns → cartouche spelled
  one glyph per letter (first-letter acrostic, per pu), using a fixed
  representative word for each of the 14 Toki Pona letters.
- `ucsur_to_latin(text)` — inverse. Cartouche reads each glyph by its
  word's first letter, concatenated and capitalized. Stacking and
  scaling joiners → spaces. Long-glyph and reverse-long-glyph markers
  silently stripped. Extended cartouches treated as normal cartouches.

Round-trips cleanly for any TP-phonotactic input. See
[`tests/test_sitelen.py`](tests/test_sitelen.py) for the contract.

---

## Training: `scripts/train_qlora.py`

QLoRA fine-tune of `google/gemma-4-E2B-it` (multimodal but used in
text-only mode) with TRL's `SFTTrainer` and `assistant_only_loss=True`.

### Defaults

| Knob | Default | Notes |
|---|---|---|
| `--lora-r` | 16 | Applied to text decoder `q/k/v/o/gate/up/down`; vision/audio towers excluded |
| `--lora-alpha` | 32 | |
| `--max-seq-length` | 160 | Capped *down* from 256 — see "OOM mitigations" |
| `--learning-rate` | 2e-4 | Cosine schedule, warmup_ratio=0.03 |
| `--batch-size` | 1 | per-device |
| `--grad-accum` | 8 | Effective batch size 8 |
| `--max-steps` | 3500 | ≈ 3 epochs over `sft_train.jsonl` at effective batch 8 |
| `--eval-steps` | 200 | |
| `--save-steps` | 200 | `save_total_limit=3` |
| `--early-stop-patience` | 3 | On `eval_loss` |
| Optimizer | `paged_adamw_8bit` | Required to fit |
| Quantization | nf4 + double-quant + bf16 compute | |
| Gradient checkpointing | on, `use_reentrant=False` | |

### Non-obvious choices

These all exist because the default codepaths don't work on a single
8 GB card with Gemma 4's vocab=262,144 lm_head. Every one of them is
load-bearing.

| Mitigation | Where | Why |
|---|---|---|
| **Custom minimal chat template** | `MINIMAL_CHAT_TEMPLATE` constant + `load_tokenizer` | Gemma 4's stock chat template lacks `{% generation %}` markers, so `assistant_only_loss=True` silently masks nothing and would train on the entire few-shot user prompt. Our template renders byte-identically to Gemma's but exposes the markers. `--check-collator` audits this. |
| **Skip PEFT's `prepare_model_for_kbit_training`** | `load_quantized_model` | The default upcasts every bf16 param to fp32 "for stability". For vocab=262144, the `embed_tokens` copy alone is 1.6 GB → OOM at load time. We instead do minimal manual `requires_grad=False` and rely on bnb's on-the-fly nf4 → bf16 dequant. |
| **Vision + audio towers moved to CPU** | `load_quantized_model` | Saves ~260 MB of VRAM we'd rather give to logits and grads. Text-only SFT doesn't use them. |
| **Patch `accelerate.utils.operations.convert_to_fp32` to a no-op** | module top | Accelerate's mixed-precision wrapper post-processes the model output by `.float()`-casting every bf16/fp16 tensor *after* the forward returned the loss scalar. For Gemma 4 that's a 204 MB upcast of the logits tensor that nothing downstream consumes — OOM. The loss is already fp32; we don't need accelerate to also upcast logits. |
| **`chunked_causal_lm_loss` replaces `model.loss_function`** | top-of-file fn | Default `ForCausalLMLoss` does `logits.float()` on the whole tensor (~268 MB fp32 allocation) before calling F.cross_entropy. Our version slices logits into 4-token bf16 chunks and calls F.cross_entropy per chunk; internal log-softmax then peaks at `chunk_size * vocab * 4 bytes` ≈ 4 MB. |
| **Graph-anchored `total_loss`** | inside `chunked_causal_lm_loss` | If every chunk's labels are `ignore_index` (heavily truncated batches), the skip-empty-chunks optimization would leave `total_loss` as a fresh non-grad zero, breaking `loss.backward()`. We initialize `total_loss = logits_flat[0, 0] * 0.0` so the autograd graph is anchored. |
| **`use_liger_kernel=True` in `SFTConfig`** | `main()` | We are *not* actually using Liger kernels — its registry has `gemma4_text` but not the top-level `gemma4` multimodal wrapper we load, so the flag is a no-op patcher. It's set purely because TRL's `compute_loss` gates its vocab-sized entropy metric (`entropy_from_logits`) behind `elif not self.args.use_liger_kernel:` — flipping it true makes TRL skip that 64 MB allocation. liger-kernel is a real dependency for the same reason (TRL ImportErrors otherwise). |
| **`EmptyCacheCallback` on `on_step_end`** | callback class | Releases PyTorch's reserved-but-unallocated GPU blocks after each step. paged_adamw_8bit's lazy state allocation + variable per-batch sequence lengths leave the allocator fragmented; this gives back enough contiguous space for the backward of the next step. |
| **`max_seq_length=160` (down from 256)** | `--max-seq-length` default | Worst-case `grad_logits = actual_seq * vocab * 2 bytes`. At seq=256 a long batch needs ~134 MB contiguous, which OOMs after fragmentation. seq=160 caps it at ~80 MB. Tradeoff: more of the user-side prompt gets truncated; the assistant response is preserved via `tokenizer.truncation_side="left"`. |
| **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** | env at top | Lets the allocator grow segments dynamically. Helps but is not sufficient on its own. |

### CLI modes

```sh
# Offline mask audit (no GPU needed) — verifies the chat-template fix
.venv/bin/python scripts/train_qlora.py --check-collator

# 50-step end-to-end smoke (~2.5 min on RTX 5060)
.venv/bin/python scripts/train_qlora.py --smoke

# Full run (~3 hours)
.venv/bin/python scripts/train_qlora.py
```

Useful flags:
- `--sample-gen` — on each eval, generate from a fixed prompt set and
  write outputs to the `Text` tab in TensorBoard. Off by default. Cost
  is a 200–500 MB KV-cache spike per eval; wall time is unmeasured (the
  callback was buggy in earlier runs and never executed end-to-end —
  see commit `b5c69ab`).
- `--run-dir PATH` — pin the output directory (default is auto-stamped).
- `--max-steps N`, `--lora-r N`, `--learning-rate LR`, etc. — every
  hyperparameter is overridable.

---

## Operational checklist for a real run

The RTX 5060 has 8 GB total; KDE + Wayland normally hold ~600 MB of that
through the framebuffer/compositor even when the screen is locked. The
training step needs essentially all of the remaining VRAM, so:

1. **Free the GPU.** From an SSH session, kill the local graphical
   session — `nvidia-smi` should drop to ~230 MB used after:
   ```sh
   sudo systemctl isolate multi-user.target
   # If KDE processes survive (they often do, even after isolate):
   loginctl list-sessions
   kill -9 <pids of plasmashell kwin_wayland kscreenlocker_g Xwayland and friends>
   ```
   Verify with `nvidia-smi --query-gpu=memory.free --format=csv`. Should
   be ≥ 7,400 MiB.

2. **(Optional) Validate the mask.** `scripts/train_qlora.py --check-collator`
   takes ~5 sec and confirms the chat-template override still masks
   correctly. Worth running after any transformers/TRL upgrade.

3. **Launch training in tmux** so it survives SSH disconnects:
   ```sh
   .venv/bin/python scripts/train_qlora.py
   ```

4. **(Optional) TensorBoard** for live monitoring. In a separate tmux
   window:
   ```sh
   .venv/bin/tensorboard --logdir models/runs/ --port 6006
   ```
   Then from a laptop terminal:
   ```sh
   ssh -N -L 6006:localhost:6006 thomasn@<rtx-host>
   ```
   and open `http://localhost:6006`. The `--logdir` points at the parent
   so all runs (current + historical) appear side-by-side.

5. **When done**, bring KDE back:
   ```sh
   sudo systemctl isolate graphical.target
   ```

---

## Output layout

Each run writes to `models/runs/qlora-<UTC>/`:

```
qlora-20260526T203942Z/
├── run_config.json              # snapshot of all CLI args (written before training)
├── events.out.tfevents.*        # TensorBoard scalars + (if --sample-gen) text
├── checkpoints/
│   └── checkpoint-<N>/          # save_total_limit=3 keeps the 3 most recent
└── final/                       # best by eval_loss (load_best_model_at_end)
    ├── adapter_model.safetensors    # ~96 MB LoRA weights, portable
    ├── adapter_config.json
    ├── chat_template.jinja          # our minimal {% generation %}-aware template
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── training_args.bin
    └── README.md                    # auto-generated by transformers
```

The `final/` adapter is what you load on top of base Gemma 4 E2B for
inference; nothing else from the run dir is needed.

---

## Stage 2 serving — running the teacher for augmentation

This is how the fine-tuned Gemma teacher is invoked to generate augmented
data (Stage 2). It is *not* the deliverable's inference path — the deliverable
is the from-scratch student model (Stage 3), not yet built.

### Current: direct HF + PEFT — `scripts/infer.py`

Loads the base Gemma 4 E2B in 4-bit nf4 (same config as training), applies
the latest run's `final/` adapter, and generates from a seed sentence
wrapped in the same `_continuation_prompt` / `_paraphrase_prompt` shape used
during SFT. `--compare-base` generates with the adapter disabled too on the
same seed for side-by-side comparison.

```sh
.venv/bin/python scripts/infer.py                       # latest adapter, default seed
.venv/bin/python scripts/infer.py --compare-base        # adapter vs base on same seed
.venv/bin/python scripts/infer.py --mode paraphrase --prompt "…"
```

Two implementation notes (see commits `eb4be3e`, `b5c69ab`):

- Current `transformers` returns a `BatchEncoding` from `apply_chat_template(...,
  return_tensors="pt")` rather than a bare tensor. `infer.py` unwraps it; the
  matching pattern in `train_qlora.py`'s `SampleGenerationCallback` was
  patched alongside.
- After the multimodal-tower CPU offload, `model.device` resolves to CPU,
  so input tensors are routed off the GPU and the embedding lookup errors
  with a device mismatch. Both inference and the training callback now pin
  to `model.get_input_embeddings().weight.device` instead.

### Ollama via merged GGUF (built — the chosen serving path)

`scripts/merge_adapter.py` (peft `merge_and_unload`) → llama.cpp
`convert_hf_to_gguf.py --outtype q8_0` → `ollama create`. The merged
tokenizer + `MINIMAL_CHAT_TEMPLATE` carry into the GGUF metadata, so Ollama
applies the correct chat template automatically; query via `/api/chat`. This
is fast enough for bulk Stage-2 generation (HF + 4-bit single-stream was too
slow for the ~10⁸ tokens the student needs).

Notes from building it:
- **Gemma 4 is supported** by current llama.cpp (`conversion/gemma.py`
  registers `Gemma4ForConditionalGeneration`); the text model converts and the
  vision/audio towers are dropped (text-only GGUF).
- **The pruned models convert fine too** — the converter reads `tokenizer.json`
  via `LlamaHfVocab` (Gemma 4 ships no `tokenizer.model`), so the filtered
  pruned vocab is no obstacle.

**Teacher selection (2026-05-28).** All three fine-tunes were merged → GGUF →
served, then compared on ~10k tokens of continuation generation
(`scripts/bench_gguf.py`, temp=0.8). Validity was uniformly high (95–96 %
strict-valid); diversity (distinct-2) ranked **baseline 0.61 ≥ parity 0.58 >
bumped 0.50**. The pruned-`parity` diversity edge seen in the earlier HF bench
(0.72) did **not** reproduce here — it was noise. So the **un-pruned baseline
fine-tune (`qlora-20260527T223141Z`) is the teacher** (`ollama` model
`waso-baseline`): marginally most diverse, standard full vocab, no pruned-base
dependency. Pruning gave no serving or robust diversity benefit.

**Superseded by the multi-task teacher (`waso-teacher`, 2026-05-29).** To add
paraphrasing, the SFT was extended with a `paraphrase` mode (Tatoeba sibling
pairs) and retrained on stock Gemma 4 → merged → `ollama` model `waso-teacher`.
Continuation quality is **unchanged** vs `waso-baseline` (bench_gguf: d-2 0.60
vs 0.61, strict 96 % both — multi-task didn't hurt it). Faithful paraphrasing,
however, did **not** materialize (see caveat below). `waso-teacher` is the
current `augment_corpus.py` default; `waso-baseline` remains as a fallback.

### Application-layer I/O

Whichever endpoint, the I/O conversion is the same: feed `latin_to_ucsur`
output to the user-facing display, accept `ucsur_to_latin` of any UCSUR
input as the model's prompt.

---

## Bench reference

`scripts/bench_gemma.py` measures quality + throughput for Ollama-served
Gemma 4 variants on Toki Pona generation. Results live in
`data/bench/` (gitignored). Not on the critical path for fine-tuning, but
useful for sanity-checking the un-fine-tuned baseline.

`scripts/augment_corpus.py` is the **Stage 2** corpus expander: it paraphrases
+ continues real sentences to grow the corpus that will train the from-scratch
student. It generates from the teacher (`--model waso-teacher`, default) via
Ollama `/api/chat` — the chat endpoint is required so the teacher's training
chat template wraps the prompt. Both `_continuation_prompt` and
`_paraphrase_prompt` are now in the teacher's SFT, so train == inference for
both modes. A global cross-seed / cross-run dedup (lowercased text) keeps the
student corpus duplicate-free; per-seed dedup + `_filter_sentence` run first.

### Decoding / repetition findings (2026-05-28)
- **Sampling, not penalties, is the repetition fix.** Greedy decoding loops
  catastrophically (distinct-1 ≈ 0.10); temp=0.8 sampling already lifts it to
  distinct-2 ≈ 0.60. Sweeping Ollama `repeat_penalty` (1.0→1.5) and
  `repeat_last_n` (64/256/full) on `waso-baseline` moved distinct-n by ≤0.01
  and left validity at ~96 % — i.e. **negligible**. The residual repetition is
  TP's natural function-word density (`li e pi la mi`…), which penalties can't
  remove without breaking grammar. The knobs are exposed (`--repeat-penalty`,
  `--repeat-last-n`) but default to Ollama's 1.1/64; raising them isn't worth it.
- **Faithful paraphrasing did not materialize (2026-05-29).** Two SFT
  iterations were tried — paraphrase pairs with a generic few-shot, then with a
  task-demonstrating (A→B) few-shot (`PARAPHRASE_FEWSHOT`). Neither produced
  reliable paraphrases: at temp=0.5 on content-rich seeds, output mostly drifts
  to unrelated TP, sometimes copies the input or flips a negation. Root cause:
  the Tatoeba signal is loose — "two TP sentences sharing an English
  translation" is weak equivalence, so ~10k noisy A→B mappings teach "emit some
  plausible TP," not a faithful rewrite (a strong capability a 2B QLoRA-r16
  model doesn't acquire from this). **It's tolerable anyway:** the output is
  still valid grammatical TP (passes `_filter_sentence`), so as student-corpus
  material it's "extra related valid TP," just not meaning-grounded
  multiplication. Continuation is unaffected. Revisit only if the student turns
  out to need tight paraphrase grounding (would need cleaner pair data).

Throughput: ~120 tok/s on the RTX 5060 via Ollama (q8_0), vs ~15–30 tok/s for
the HF + 4-bit path — the reason bulk augmentation goes through GGUF/Ollama.

---

## Stage 2 v2 — Translation augmentation (built)

**Built (2026-06-05).** Pipeline: `build_translation_pairs.py` →
`build_sft_dataset.py --translation-only` → `train_qlora.py` → merge → GGUF →
**standalone `waso-translator`** (Ollama) → `build_english_source.py` →
`translate_corpus.py` → student retrain. Measured results:
- **Translator gate** (`eval_translator.py`, 500 held-out *unseen-English*
  pairs): chrF **68.6**, **99.0 %** valid TP, 0 % copied/empty — genuine
  translation of novel English, not recall.
- **Translated corpus**: 300k simple English (≈82 % Tatoeba-without-TP-link /
  ≈18 % Simple-English Wikipedia) → **265,477** unique TP sentences /
  **2.63 M words** (≈97 % pilot acceptance) — *larger* than the 2.3 M-word real
  corpus, and genuinely new content (Algeria, the Bible, horses…).
- **Student payoff**: retraining on **real + translated + v1-synthetic** (8.44 M
  train tokens vs v1's 2.17 M; same tokenizer + architecture for a clean A/B)
  dropped the honest **`real_loss` ~4.0 → 2.16**, and the val↔real gap collapsed
  — the student now models real TP, not just the teacher's distribution.
  Early-stopped at step 18000 (vs v1's 7000).
- Deviations from the plan below: a **standalone** `waso-translator` (not a 4th
  teacher task); the shared prompt is `_translate_prompt`; a dedicated
  `scripts/build_english_source.py` builds the source pool; the translate-loop
  rejects are `empty` / `copied_source` (leftover English surfaces as
  `_filter_sentence`'s `unknown_word`).

The design, as planned (still accurate):

**Why v1 hit a wall.** Paraphrase + continuation can only *recombine* Toki Pona
that already exists — the teacher conditions on a TP seed and emits more TP from
the same ~137-word vocabulary. It cannot add semantic *content*. The 30k-seed
bulk run made this concrete: output saturated (accept rate 70 %→59 %,
records/seed 7.74→6.51) and yielded 201,567 records / 1.82 M words ≈ **1.9 M
SentencePiece tokens** — *smaller* than the 3.16 M-token real corpus it was
meant to multiply 10×, built from only ~1,800 distinct word-forms. The
"paraphrase" half doesn't even track the seed's meaning (the en-sibling signal
is too loose — see the v1 caveat above), so both modes effectively emit generic
valid TP. Surface-unique, semantically thin.

**The fix: translation.** Translate diverse **English** source text into TP. Now
the novelty comes from the *source content*, constrained into TP's vocabulary —
the corpus gains real topical range instead of more function-word permutations.
Translation is the only lever that actually multiplies a closed-vocabulary
language's data, because the new information enters from outside the language.

**1. Translator (`waso-translator`, or a translate task on `waso-teacher`).**
Reuse the QLoRA pipeline unchanged (`train_qlora.py`, base `gemma-4-E2B-it`,
r16) with a new en→tp SFT task: `user = "Translate to Toki Pona: <english>"`,
`assistant = "<tp>"`, same `MINIMAL_CHAT_TEMPLATE`. Recommended: fold it into the
existing **multi-task teacher** as a 4th task (alongside continuation /
paraphrase / topic) so one Ollama model serves all of Stage 2; fall back to a
standalone `waso-translator` if the task mix dilutes quality.

**2. Parallel SFT data (~40k pairs, mostly already latent).** Tatoeba's
`data/raw/tatoeba/tok-eng_links.tsv` (~45k tok↔eng id links, already fetched) +
the local `tok_sentences.tsv`, joined against Tatoeba's `eng_sentences.tsv` (one
extra download), yields ~40k aligned **(English, TP)** pairs. Restore the 13
`lipu` direct pairs (English is currently discarded in `iter_lipu`). Filter every
TP target through `_filter_sentence` (the same strict validator). New builder:
`scripts/build_translation_pairs.py` feeding a `translation` mode in
`build_sft_dataset.py`. Hold a slice out for translator eval.

**3. English source for inference — the actual augmentation.** Translate *novel*
English that is **not** in the tp-linked set, so the TP output is genuinely new:
- Primary: Tatoeba English sentences that lack a TP link (simple
  single-sentence register — the same distribution the translator trains on —
  and very large volume).
- Stretch: Simple-English Wikipedia sentences (concrete encyclopedic content).
- Constraint: keep the source **simple and concrete**. TP cannot express
  abstract / technical English; those inputs translate into unknown words and
  get dropped by `_filter_sentence` — wasted teacher time. Pre-filter the source
  for short, concrete sentences before translating.

**4. Generation loop — reuse the v1 machinery.** A `translate` mode in
`augment_corpus.py` (or a sibling `scripts/translate_corpus.py`) reuses
`_ollama_generate` (`/api/chat`), `_split_into_sentences`, `_filter_sentence`,
the per-seed + global dedup, and the `.done` resumability sidecar **unchanged**.
New parts are small: an `_english_to_tp_prompt(english)` few-shot builder, an
English-source loader (replacing the TP `sentences.txt` input), and two
translation-specific rejects — **leftover English** (non-TP letters / English
tokens survived in the output) and **copy-of-source / refusal**. Output:
`data/processed/translated.jsonl` — `{source: "waso-translator/translate", eng,
text}`.

**5. Eval.** Translator quality on the held-out Tatoeba slice (chrF / BLEU vs the
reference TP) plus the `_filter_sentence` validity pass-rate at inference; the
end-to-end signal is the **student's `eval/real_loss`** (v1 leaves it ~4.0 vs
syn_val ~1.7 — translated content is the lever expected to close that gap).
Optional QC: tp→en back-translation and similarity to the source English, to
catch fluent mistranslations the grammar filter can't see.

**6. Student integration.** The student then trains on **real (primary) +
translated (new content) + v1 synthetic (grammar regularization)**, with a
held-out real slice kept *out* of training for the honest eval. Note
`train_student.py` currently trains on the v1 synthetic only and uses just 200
real docs purely for eval — folding the real + translated corpora into training
is the companion change on the Stage 3 side.

---

## Stage 3 — Student (from-scratch tiny TP LM, the deliverable)

The actual product. A nanoGPT-style decoder-only transformer trained
**from scratch** on the augmented corpus produced by Stage 2 — small enough
to run locally on consumer hardware, focused entirely on Toki Pona.

Three scripts:

### `scripts/train_tokenizer.py`
Trains a small **SentencePiece BPE** tokenizer (vocab=2,048, byte-fallback
on) over `synthetic.jsonl` + a chars-capped slice of `corpus.filtered.jsonl`.
Output → `models/student_tokenizer/spm.{model,vocab}`. On the current
data: ~1.15 tokens/word for known TP, byte-fallback for arbitrary names.
Special-token IDs: `<pad>=0 <unk>=1 <bos>=2 <eos>=3`.

### `scripts/train_student.py`
Minimal PyTorch training loop (no HF Trainer — from-scratch training has
different needs than the QLoRA teacher path: no quantization, no chat
template, no adapter merge). Defaults: **~5M params** (6 layers × 256 hidden
× 4 heads × 512 context), bf16 on CUDA, AdamW with warmup + cosine LR,
gradient clipping. Records are framed with `<bos>…<eos>` and concatenated
into a token stream; batches sample random `block_size` windows.

Eval has **two signals**:
- `eval/syn_loss` — held-out 5% of synthetic (matches train distribution).
- `eval/real_loss` — held-out slice of the real filtered corpus (the
  *honest* signal — real_loss > syn_loss confirms the model is fitting
  teacher distribution faithfully but tells us how it transfers to actual TP).

Periodic sample generation (logged to TensorBoard `sample` text tag) +
checkpoint saving (`best.pt` by syn_val, `last.pt`, `final.pt`). `--smoke`
runs 100 steps with a tiny config for sanity checking.

### `scripts/talk_to_student.py`
Inference. Loads the latest `best.pt` by default + the SP tokenizer,
generates from a Latin-TP prompt (or just `<bos>`), and with `--ucsur`
renders the output through `sitelen.translate.latin_to_ucsur` for sitelen
pona display. The student itself only ever sees Latin script; UCSUR is the
application layer wrapping it, exactly as the diagram in the intro shows.

### Status (2026-06-05)
All three scripts exist and have run end-to-end. The 30k-seed v1 bulk
augmentation produced `synthetic.jsonl` — **201,567 records / 1.82 M word-tokens
≈ 1.9 M SP-tokens** (the "~25.9 M Ollama-tokens" sometimes quoted is the
teacher's gross `eval_count`, ~93 % discarded — *not* trainable size) with
**1,121 unique TP words** (per-record `distinct-2 = 0.987`, no internal
looping; balanced ~99k continuation / ~102k paraphrase).

A "record" here = one accepted JSON line — typically a single short TP
sentence (~9 words avg) plus `{source, seed, mode}` metadata. The student's
trainer concatenates all `text` fields with `<bos>…<eos>` framing into one
token stream and samples random windows.

**Saturation hit on schedule.** Accept rate dropped 70.4 % → 59.2 % between
the first 5k seeds and the next 25k; records/seed slid 7.74 → 6.51. The
dup-rate-vs-seeds knee predicted at ~30 k landed at ~30 k. Going further
would yield diminishing unique-record returns — for the targeted ~25 M
tokens this is the right stopping point.

**Student runs — and the balance lesson.** Honest `real_loss` (held-out real
docs, same 200-doc slice + tokenizer throughout, so the numbers are comparable):

| run | model | train tokens | real share | real_loss |
|---|---|---|---|---|
| v1 (2026-06-02) | 6L/256d, 5.4 M | 2.2 M | 0 % (eval-only) | 4.0 |
| v2 (2026-06-05) | 6L/256d, 5.4 M | 8.9 M | 37 % | 2.16 |
| scaled, unbalanced | 8L/384d, 15.2 M | 24.3 M | 13 % | 2.22 |
| **rebalanced (best)** | **8L/384d, 15.2 M** | **50.9 M** | **37 %** | **2.08** |

- **v1 → v2**: adding the translated corpus to a real-anchored mix collapsed the
  val↔real gap and took `real_loss` 4.0 → 2.16 — translation injects real-world
  content paraphrase/continuation can't.
- **Scaling translated to dominate *regressed* it** (2.16 → 2.22). After the full
  English pool, `translated.jsonl` reached **1,576,213 sentences / 16 M words**,
  but at all of it the real corpus fell to **13 %** of the mix and the model
  drifted toward machine-*translationese* (grammatical but not idiomatic).
- **Balance, not volume, is the lever.** Upsampling real ×6 (`--real-weight`) to
  restore the 37 % share — same 15 M model, same translated corpus, *only* the
  proportion changed — swung `real_loss` **2.22 → 2.08**, beating v2. The
  ~3.3 M-token human corpus is the quality ceiling; `real_loss` is near its floor.

The **rebalanced 8L/384d checkpoint is the current deliverable**
(`models/student_runs/student-8L384d-rebal-20260607T051752Z/best.pt`).
Generations are fluent and worldly Latin TP (e.g. `mi toki lon toki sina`,
`jan Ton li tawa kama lon poka pi jan Mewi`), which the `sitelen/` layer renders
to sitelen pona glyphs on output — the model itself only ever emits Latin TP.

### Student bench — fine-tune readiness (`bench_student.py`, 2026-06-07)

`scripts/bench_student.py` characterizes the *base* student model so we can
reason about what to fine-tune it for, rather than reading a single loss. It runs
five sections over the deliverable plus the three prior student runs (same
tokenizer → directly comparable), writing `data/bench/student_bench_summary.json`
+ `student_gens.jsonl` (1,240 raw generations). Sections: **(A)** exact
strided-window perplexity, **(B)** free generation across a temperature sweep,
**(C)** a prompted-continuation capability map by register, **(D)** decode
throughput, **(E)** rule-based fine-tune-readiness verdicts. Validity/diversity
reuse `bench_adapters.score_texts` + `augment_corpus._filter_sentence`; the
perplexity uses an exact full-coverage window, **not** the random-window
`eval_loss` (which never covers the tail).

**The bench reproduces the training `real_loss` history exactly** (held-out
tatoeba, exact NLL → the numbers validate both the bench and the run ranking):

| run | arch | held-out tatoeba nll (ppl) | training `real_loss` |
|---|---|---|---|
| **rebal (deliverable)** | 8L/384d | **2.105 (8.2)** | 2.08 |
| v2 (real+trans+synth) | 6L/256d | 2.171 (8.8) | 2.16 |
| scaled (unbalanced) | 8L/384d | 2.254 (9.5) | 2.22 |
| v1 (synthetic-only) | 6L/256d | 4.228 (68.5) | 4.0 |

**Honest-eval caveat (load-bearing).** `corpus.filtered.jsonl` is ordered by
source, and training held out only `real_all[:200]` — which is *entirely tatoeba
single-sentences*. So only `tatoeba_heldout_200` is a generalization number;
every other per-source perplexity is **in-sample** (those docs were trained on,
real upsampled ×6) and is a *register-coverage* indicator, not generalization.
The clean same-register isolation of that effect: rebal scores tatoeba **2.105
held-out vs 1.936 in-sample** — a 0.17-nll memorization gap. The deliverable
represents *all* real registers tightly (poki/nltk/tokwiki in-sample nll
1.3–1.5); the synthetic-only v1, which never saw real long-form prose, blows up
on exactly those registers (poki/tokwiki nll 6.3–6.6) — concrete evidence that
the real+translated corpus, not the synthetic, bought the long-form coverage.

**Capability map (deliverable, continuations scored at temp 0.9, by validity):**

| register | sent-valid | strict-valid | reads as |
|---|---|---|---|
| narrative opener | 100 % | 80 % | story / prose |
| mi/sina personal | 100 % | 75 % | chat / diary |
| seme question | 100 % | 50 % | Q-completion |
| topical / encyclopedic | 100 % | 40 % | factual (longer, looser) |
| instruction-ish (`o …`) | 100 % | 25 % | *never seen — needs SFT* |

Free generation is **clean and non-degenerate**: doc/sentence/strict validity
~100 %, distinct-2 ≈ 1.0, OOV ≤ 0.4 %, and — unlike the Gemma teacher, which
looped catastrophically at greedy — the from-scratch student **does not loop even
at greedy** (loop-rate 0 % at every temperature). Decode throughput: **1,245
tok/s bf16 GPU / 228 tok/s CPU** for the 15 M deliverable (the 5.4 M v2: 1,566 /
516) — real-time for an in-app IME even on CPU.

**Fine-tune readiness (deliverable):** *ready* for conversational/chat SFT,
story/prose generation, autocomplete/IME, grammar-style correction, and
constrained/templated generation; *needs-work* for instruction following (the
base never saw an instruction format — that's SFT distance, not a defect);
en→tp is *promising-as-decoder* (the base is a strong TP-side LM prior but an
English encoder/conditioning must be added); tp→en is *not supported by the base*
(it only ever produced TP — needs a bilingual corpus + likely vocab extension).
Every "ready" verdict is about **TP-side fluency**; targets needing English or
instruction formats require new data the base has never seen.

**Strategic note (base selection for fine-tuning).** The deliverable wins on
perplexity (2.10 vs 2.17) and encyclopedic coverage, but v2 (5.4 M) has a nearly
identical capability map — it actually edges the deliverable on mi/sina (88 %)
and encyclopedic (55 %) continuations — at **~1.3× the throughput**. For a
latency-sensitive target (e.g. an on-device IME) v2 is a strong, cheaper base;
for quality/perplexity-sensitive targets, use the rebalanced deliverable.
