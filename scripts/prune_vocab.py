"""Prune Gemma 4's 262,144-token vocab down to what waso-sona actually uses.

Workflow:

1. Tokenize the filtered corpus, the SFT dataset (via the chat template
   that training uses), and the inference-time prompt skeletons from
   augment_corpus.
2. Union the resulting token-ID set with Gemma's special tokens and all
   256 byte-fallback tokens so any UTF-8 input still tokenizes.
3. Slice ``embed_tokens`` (and ``lm_head`` if untied) row-wise to the
   kept set.
4. Filter ``tokenizer.json``'s ``vocab`` and ``merges`` to drop entries
   referencing pruned IDs, renumbering by the new contiguous index.
5. Save the pruned base under ``data/training/base-pruned/`` and verify
   by round-tripping a corpus sample.

Pure CPU; loads the base in bf16 (~5 GB RAM) for the one-shot slice.
No bnb/CUDA dependency — the resulting checkpoint is plain HF safetensors.

Usage::

    python scripts/prune_vocab.py             # defaults
    python scripts/prune_vocab.py --dry-run   # build keep-set only, no save
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import augment_corpus  # noqa: E402
from train_qlora import MINIMAL_CHAT_TEMPLATE, MODEL_ID  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "data/processed/corpus.filtered.jsonl"
DEFAULT_SFT_TRAIN = REPO_ROOT / "data/processed/sft_train.jsonl"
DEFAULT_SFT_VAL = REPO_ROOT / "data/processed/sft_val.jsonl"
DEFAULT_OUT = REPO_ROOT / "data/training/base-pruned"


# ---------------------------------------------------------------------------
# Keep-set construction
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_used_ids(
    tok,
    corpus_path: Path,
    sft_paths: list[Path],
) -> set[int]:
    """Run the tokenizer over every artifact that will be tokenized at
    training or inference time. Return the union of seen token IDs."""
    used: set[int] = set()

    # 1. The filtered corpus (every doc, as plain text).
    n_docs = 0
    for rec in _iter_jsonl(corpus_path):
        ids = tok(rec["text"], add_special_tokens=False)["input_ids"]
        used.update(ids)
        n_docs += 1
    print(f"  corpus: {n_docs} docs, |used| so far = {len(used):,}", flush=True)

    # 2. Every SFT example, via apply_chat_template — captures the
    # chat-template literal tokens (`<|turn>`, `<turn|>`, role strings,
    # newlines) in exactly the form training renders.
    #   We render to string then tokenize (rather than tokenize=True) so we
    #   always get a flat list[int]. The fast-tokenizer path of
    #   apply_chat_template(tokenize=True) returns a BatchEncoding whose
    #   iteration yields its dict keys — silently feeding strings into the
    #   keep set.
    n_sft = 0
    for path in sft_paths:
        if not path.exists():
            continue
        for rec in _iter_jsonl(path):
            text = tok.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False
            )
            ids = tok(text, add_special_tokens=False)["input_ids"]
            used.update(ids)
            n_sft += 1
    print(f"  sft:    {n_sft} examples, |used| so far = {len(used):,}", flush=True)

    # 3. Inference-time prompt skeletons (continuation + paraphrase),
    # rendered with a representative TP seed so the seed-shape tokens
    # also enter the keep set.
    seeds = [
        "mi tawa ma kasi.",
        "tenpo suno ni la mi pilin pona.",
        "soweli lili li lon supa.",
    ]
    for seed in seeds:
        for builder in (
            augment_corpus._continuation_prompt,
            augment_corpus._paraphrase_prompt,
        ):
            prompt = builder(seed)
            # Also wrap in chat template — that's how the SFT dataset
            # actually presents it, so this catches the same templated
            # neighborhood.
            for use_template in (False, True):
                if use_template:
                    text = tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    ids = tok(text, add_special_tokens=False)["input_ids"]
                else:
                    ids = tok(prompt, add_special_tokens=False)["input_ids"]
                used.update(ids)
    print(f"  prompts: |used| so far = {len(used):,}", flush=True)

    return used


def add_safety_tokens(tok, used: set[int]) -> set[int]:
    """Augment with: all declared special tokens, plus the 256 byte-fallback
    tokens so any UTF-8 input remains tokenizable post-prune."""
    safe = set(used)
    safe.update(tok.all_special_ids)
    n_specials = len(safe) - len(used)

    # Byte-fallback tokens. SentencePiece-derived BPE conventionally names
    # these "<0x00>".."<0xFF>"; resolve via convert_tokens_to_ids.
    byte_added = 0
    for b in range(256):
        s = f"<0x{b:02X}>"
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid != tok.unk_token_id:
            if tid not in safe:
                byte_added += 1
            safe.add(tid)
    print(
        f"  +{n_specials} specials, +{byte_added} byte-fallback "
        f"→ |keep| = {len(safe):,}",
        flush=True,
    )
    return safe


# ---------------------------------------------------------------------------
# Model slicing
# ---------------------------------------------------------------------------

def prune_model(model, new_to_old: list[int]):
    """Slice embed_tokens (and lm_head if untied) to keep only the rows
    indicated by new_to_old. Updates config.vocab_size in place."""
    old_embed = model.get_input_embeddings()
    old_head = model.get_output_embeddings()
    tied = old_embed.weight.data_ptr() == old_head.weight.data_ptr()
    print(
        f"  embedding tying: {'tied' if tied else 'untied'} "
        f"(config.tie_word_embeddings={getattr(model.config, 'tie_word_embeddings', None)})",
        flush=True,
    )

    idx = torch.tensor(new_to_old, dtype=torch.long)
    new_vocab = idx.numel()
    hidden = old_embed.embedding_dim

    new_embed_w = old_embed.weight.detach().index_select(0, idx).contiguous()
    new_embed = nn.Embedding(
        new_vocab, hidden, dtype=new_embed_w.dtype, device=new_embed_w.device
    )
    new_embed.weight.data.copy_(new_embed_w)
    model.set_input_embeddings(new_embed)

    if tied:
        model.tie_weights()
    else:
        new_head_w = old_head.weight.detach().index_select(0, idx).contiguous()
        new_head = nn.Linear(
            hidden, new_vocab, bias=False,
            dtype=new_head_w.dtype, device=new_head_w.device,
        )
        new_head.weight.data.copy_(new_head_w)
        model.set_output_embeddings(new_head)

    model.config.vocab_size = new_vocab
    if hasattr(model.config, "text_config"):
        # Multimodal Gemma 4 carries vocab on the inner text_config too.
        if getattr(model.config.text_config, "vocab_size", None) is not None:
            model.config.text_config.vocab_size = new_vocab


# ---------------------------------------------------------------------------
# Tokenizer slicing
# ---------------------------------------------------------------------------

def prune_tokenizer_json(tok, kept_ids: set[int], old_to_new: dict[int, int]) -> str:
    """Return a modified tokenizer.json string with vocab and merges filtered."""
    raw = tok.backend_tokenizer.to_str()
    data = json.loads(raw)

    bpe = data.get("model", {})
    if bpe.get("type") != "BPE":
        raise RuntimeError(
            f"Expected model.type=='BPE' for Gemma tokenizer; got {bpe.get('type')!r}"
        )

    # Identify kept token strings (the union we built is over IDs).
    old_vocab: dict[str, int] = bpe.get("vocab", {})
    kept_strings = {tok_str for tok_str, tid in old_vocab.items() if tid in kept_ids}

    # Renumber vocab by new contiguous IDs.
    new_vocab = {
        tok_str: old_to_new[tid]
        for tok_str, tid in old_vocab.items()
        if tid in kept_ids
    }
    bpe["vocab"] = new_vocab

    # Filter merges. The format can be:
    #   - list of "a b" strings (older `tokenizers` versions), or
    #   - list of [a, b] two-element lists (newer versions).
    # In both cases a merge is valid iff both operands AND the concat result
    # are still in the kept vocab.
    def _split(m):
        if isinstance(m, str):
            a, b = m.split(" ", 1)
            return a, b, m
        if isinstance(m, list) and len(m) == 2:
            return m[0], m[1], [m[0], m[1]]
        raise RuntimeError(f"Unrecognized merge entry: {m!r}")

    kept_merges = []
    dropped = 0
    for m in bpe.get("merges", []):
        a, b, original = _split(m)
        if a in kept_strings and b in kept_strings and (a + b) in kept_strings:
            kept_merges.append(original)
        else:
            dropped += 1
    bpe["merges"] = kept_merges

    # Renumber added_tokens IDs.
    for at in data.get("added_tokens", []):
        old_id = at.get("id")
        if old_id in old_to_new:
            at["id"] = old_to_new[old_id]
        elif old_id is not None and old_id not in kept_ids:
            # Should not happen — added tokens were in tok.all_special_ids
            # which we union'd into kept_ids. Loud failure if it does.
            raise RuntimeError(
                f"added_token id={old_id} ({at.get('content')!r}) was pruned"
            )

    print(
        f"  tokenizer: vocab {len(old_vocab):,} → {len(new_vocab):,}, "
        f"merges kept {len(kept_merges):,} (dropped {dropped:,})",
        flush=True,
    )
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _normalize_for_roundtrip(s: str) -> str:
    # SentencePiece detokenization can normalize whitespace; we only
    # care about character identity for tokens we kept.
    return " ".join(s.split())


def verify_round_trip(
    orig_tok, pruned_tok, corpus_path: Path, sample_size: int = 100
) -> tuple[int, int]:
    docs = list(_iter_jsonl(corpus_path))
    random.seed(0)
    sample = random.sample(docs, min(sample_size, len(docs)))
    ok = 0
    for rec in sample:
        text = rec["text"]
        ids = pruned_tok(text, add_special_tokens=False)["input_ids"]
        decoded = pruned_tok.decode(ids, skip_special_tokens=True)
        if _normalize_for_roundtrip(decoded) == _normalize_for_roundtrip(text):
            ok += 1
    return ok, len(sample)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--sft-train", type=Path, default=DEFAULT_SFT_TRAIN)
    ap.add_argument("--sft-val", type=Path, default=DEFAULT_SFT_VAL)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute keep-set and exit without slicing/saving.")
    args = ap.parse_args(argv)

    print(f"Loading tokenizer {args.model_id}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.chat_template = MINIMAL_CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    orig_vocab = tok.vocab_size
    print(f"  original vocab_size = {orig_vocab:,}", flush=True)

    print("Building keep-set from corpus + SFT + prompts…", flush=True)
    used = collect_used_ids(tok, args.corpus, [args.sft_train, args.sft_val])
    kept = add_safety_tokens(tok, used)

    new_to_old = sorted(kept)
    old_to_new = {old: new for new, old in enumerate(new_to_old)}
    new_vocab = len(new_to_old)
    reduction = 1 - new_vocab / orig_vocab
    print(
        f"Keep-set: {new_vocab:,} / {orig_vocab:,} "
        f"({reduction*100:.1f}% reduction)",
        flush=True,
    )

    if args.dry_run:
        print("--dry-run: stopping before slice/save", flush=True)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "id_map.json").write_text(
        json.dumps({"new_to_old": new_to_old}, ensure_ascii=False)
    )

    print(f"Loading model {args.model_id} in bf16 (CPU)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print("Slicing embeddings…", flush=True)
    prune_model(model, new_to_old)

    print(f"Saving pruned model → {args.out_dir}", flush=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    # Free CPU RAM before we run the round-trip check.
    del model

    print("Pruning tokenizer.json…", flush=True)
    new_tok_json = prune_tokenizer_json(tok, kept, old_to_new)
    # Save through the fast-tokenizer path so all auxiliary files
    # (tokenizer_config.json, special_tokens_map.json) get regenerated
    # with consistent IDs.
    (args.out_dir / "tokenizer.json").write_text(new_tok_json, encoding="utf-8")

    # Patch the tokenizer's special-token IDs in-place before save_pretrained
    # so tokenizer_config.json gets correct numbers. We do this by reloading
    # the modified tokenizer.json through a fresh fast-tokenizer wrapper.
    from transformers import PreTrainedTokenizerFast
    pruned_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(args.out_dir / "tokenizer.json"),
        bos_token=tok.bos_token,
        eos_token=tok.eos_token,
        pad_token=tok.pad_token,
        unk_token=tok.unk_token,
    )
    pruned_tok.chat_template = MINIMAL_CHAT_TEMPLATE
    pruned_tok.truncation_side = "left"
    pruned_tok.save_pretrained(args.out_dir)

    print("Verifying round-trip on corpus sample…", flush=True)
    ok, total = verify_round_trip(tok, pruned_tok, args.corpus, sample_size=100)
    rate = ok / total if total else 0.0
    print(f"  round-trip: {ok}/{total} = {rate*100:.1f}%", flush=True)
    if rate < 0.99:
        print(
            "WARNING: round-trip rate below 99 %. Inspect dropped merges "
            "or expand the keep-set.",
            file=sys.stderr,
        )
        return 2

    # Quick stats on memory savings (using the bf16 model files we saved).
    saved_bytes = (orig_vocab - new_vocab) * 2048 * 2  # bf16, hidden ~= 2048
    print(
        f"\nDone. Pruned base saved to {args.out_dir.relative_to(REPO_ROOT)}/",
        flush=True,
    )
    print(
        f"Approx embedding savings: "
        f"{saved_bytes/1e9:.2f} GB (tied; doubled if untied)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
