# Pipeline

End-to-end view of how waso-sona goes from raw Toki Pona text on the
internet to a fine-tuned Gemma 4 adapter that speaks Toki Pona, with a
deterministic Latin ↔ sitelen pona UCSUR translator wrapping the model
at the application layer.

```
                 fetch_data.py             filter_corpus.py
   public TP    ─────────────▶  data/raw/   ────────────────▶
   corpora                       (7 sources)

   data/processed/corpus.jsonl        build_sft_dataset.py
   data/processed/corpus.filtered.jsonl ───────────────────▶

       data/processed/sft_train.jsonl       train_qlora.py
       data/processed/sft_val.jsonl    ───────────────────▶

                                    data/training/runs/qlora-<UTC>/final/
                                       └── adapter_model.safetensors   (~96 MB)

   ┌────────────────────────────────────────────────────────────┐
   │  Application layer (not in training)                        │
   │                                                             │
   │   UCSUR input ──▶ sitelen.ucsur_to_latin ──▶  model         │
   │                                                ▼            │
   │   UCSUR output ◀── sitelen.latin_to_ucsur ──  Latin output  │
   └────────────────────────────────────────────────────────────┘
```

The model **only ever sees Latin-script Toki Pona**. UCSUR is a display
concern handled deterministically by the `sitelen/` package on the way in
and out — see [`sitelen/translate.py`](sitelen/translate.py).

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

Produces `data/processed/sft_train.jsonl` (~11,800 rows) and `sft_val.jsonl`
(~630 rows) in HF messages format. Two example types:

- **Continuation pairs**: split a doc chunk into a prefix (1–3 sentences,
  ≥ 5 words) and a 1–2-sentence suffix; the user message is **the exact
  `_continuation_prompt(prefix)` shape that `scripts/augment_corpus.py`
  will send at inference**, including the same few-shot block. This keeps
  the model's train/inference distribution aligned.
- **Topic prompts**: one of 10 generic instruction templates ("Write a
  short Toki Pona story…") paired with a real chunk as the response.

The intended split is ~80 % continuation / 20 % topic (`--topic-prompt-frac
0.20`), but the realized split on the current corpora is ~56 % / 44 %.
The skew comes from short `tatoeba` docs (avg 40 chars) failing the
`MIN_PREFIX_WORDS=5` check for continuation pairs and falling through to
topic prompts.

Each candidate assistant response runs through `augment_corpus._filter_sentence`
(strict: zero-tolerance non-vocab, double-`li` / `li e` rejection, repetition
caps, missing-predicate check). Then dedup by response hash, then drop any
example whose rendered chat template exceeds `--max-tokens` (default 1024).
Split is deterministic by id-hash (`val_frac=0.05`).

Token-length percentiles on a typical build: p50 = 224, p90 = 277,
p99 = 332, max = 396.

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

Output lands in `data/training/base-pruned/` (a plain HF safetensors
checkpoint + pruned tokenizer). To train against it, pass
`--model-id data/training/base-pruned` to `train_qlora.py` — no other
flags change, and `chunked_causal_lm_loss` reads the new `vocab_size`
from the config automatically. The pruned tokenizer is saved into each
run's `final/` adapter dir, so `infer.py` picks it up; only the
`--base-model data/training/base-pruned` flag is needed at inference.

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
  middle colon; commas dropped; capitalized unknowns → cartouche of
  per-syllable glyphs (PU-word preferred, then shortest, then alphabetic)
  with a CV+n fallback for CVN syllables that have no representative word.
- `ucsur_to_latin(text)` — inverse. Cartouche reads each glyph by its
  word's first syllable, concatenated and capitalized. Stacking and
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
   .venv/bin/tensorboard --logdir data/training/runs/ --port 6006
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

Each run writes to `data/training/runs/qlora-<UTC>/`:

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

## Inference

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

### Planned: Ollama via merged GGUF

`peft` merge → `convert_hf_to_gguf` → `ollama create`. Lets
`scripts/augment_corpus.py` use the fine-tuned model with no code changes
(it already talks to Ollama).

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

`scripts/augment_corpus.py` is the inference-side corpus expander that
calls Gemma (via Ollama) to paraphrase + continue real sentences. Both
SFT dataset prompts and the augmentation prompts come from this file's
`_continuation_prompt` / `_paraphrase_prompt`, so the SFT data trains
the model on exactly the shape it will be queried with later.
