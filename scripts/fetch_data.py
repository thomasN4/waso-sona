"""Fetch and normalize Toki Pona text corpora into ./data.

Each source is fetched into ``data/raw/<source>/`` and then normalized into
two files under ``data/processed/``:

- ``corpus.jsonl``  -- one JSON record per document: {source, id, text}
- ``sentences.txt`` -- one sentence per line, deduped, for LM training

Usage::

    python scripts/fetch_data.py                       # all sources
    python scripts/fetch_data.py --source poki tatoeba # subset
    python scripts/fetch_data.py --force               # re-download

Pure stdlib; no third-party deps required.
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from sitelen.glyphs import WORD_TO_CODEPOINT  # noqa: E402

TP_VOCAB = set(WORD_TO_CODEPOINT)
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

@dataclass
class Source:
    name: str
    fetch: Callable[[Path], None]      # populates raw_dir
    iter_docs: Callable[[Path], Iterator[tuple[str, str]]]  # (doc_id, text)


def _run(cmd: list[str], **kw) -> None:
    subprocess.run(cmd, check=True, **kw)


def _git_clone(url: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    _run(["git", "clone", "--depth=1", "--quiet", url, str(dest)])


def _http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "waso-sona/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


# ---- Tatoeba --------------------------------------------------------------

TATOEBA_URL = (
    "https://downloads.tatoeba.org/exports/per_language/tok/tok_sentences.tsv.bz2"
)


def fetch_tatoeba(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "tok_sentences.tsv.bz2"
    _http_download(TATOEBA_URL, archive)
    out = raw_dir / "tok_sentences.tsv"
    with bz2.open(archive, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)


def iter_tatoeba(raw_dir: Path) -> Iterator[tuple[str, str]]:
    path = raw_dir / "tok_sentences.tsv"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] == "tok":
                yield parts[0], parts[2]


# ---- kulupu-lapo/poki -----------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MD_HEADER_RE = re.compile(r"^#+\s*", re.MULTILINE)
_MD_EMPH_RE = re.compile(r"[*_]{1,3}([^*_\n]+)[*_]{1,3}")
_TRAILING_BACKSLASH_RE = re.compile(r"\\\s*$", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    text = _FRONTMATTER_RE.sub("", text, count=1)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_EMPH_RE.sub(r"\1", text)
    text = _TRAILING_BACKSLASH_RE.sub("", text)
    return text.strip()


def fetch_poki(raw_dir: Path) -> None:
    _git_clone("https://github.com/kulupu-lapo/poki.git", raw_dir)


def iter_poki(raw_dir: Path) -> Iterator[tuple[str, str]]:
    plaintext = raw_dir / "plaintext"
    if not plaintext.exists():
        return
    for md in sorted(plaintext.rglob("*.md")):
        raw = md.read_text(encoding="utf-8", errors="replace")
        body = _strip_markdown(raw)
        if body:
            yield str(md.relative_to(raw_dir)), body


# ---- davidar/nltk-tp ------------------------------------------------------

def fetch_nltk_tp(raw_dir: Path) -> None:
    _git_clone("https://github.com/davidar/nltk-tp.git", raw_dir)


def iter_nltk_tp(raw_dir: Path) -> Iterator[tuple[str, str]]:
    corpus = raw_dir / "Corpus"
    if not corpus.exists():
        return
    for txt in sorted(corpus.rglob("*.txt")):
        try:
            body = txt.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if body:
            yield str(txt.relative_to(raw_dir)), body


# ---- nymwa/lipu (small parallel set; TP side only) ------------------------

def fetch_lipu(raw_dir: Path) -> None:
    _git_clone("https://github.com/nymwa/lipu.git", raw_dir)


def iter_lipu(raw_dir: Path) -> Iterator[tuple[str, str]]:
    tsv = raw_dir / "translation.tsv"
    if not tsv.exists():
        return
    for i, line in enumerate(tsv.read_text(encoding="utf-8").splitlines()):
        parts = line.split("\t")
        if parts and parts[0].strip():
            yield f"translation.tsv:{i}", parts[0].strip()


# ---- nymwa/Toki-Pona-1000-Sentences-Corpus --------------------------------

def fetch_tp1k(raw_dir: Path) -> None:
    _git_clone(
        "https://github.com/nymwa/Toki-Pona-1000-Sentences-Corpus.git", raw_dir
    )


def iter_tp1k(raw_dir: Path) -> Iterator[tuple[str, str]]:
    txt = raw_dir / "pona_corpus1000.txt"
    if not txt.exists():
        return
    for i, line in enumerate(txt.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line:
            yield f"pona_corpus1000.txt:{i}", line


# ---- tok.wikipedia.org dump -----------------------------------------------

TOKWIKI_URL = (
    "https://dumps.wikimedia.org/tokwiki/latest/"
    "tokwiki-latest-pages-articles.xml.bz2"
)

# Minimal MediaWiki-markup stripper. Not a full parser — drops the constructs
# that contribute the most non-Toki-Pona letters (templates, refs, link
# targets, table syntax) and leaves the natural-language body alone.
_WIKI_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_WIKI_REF_BLOCK_RE = re.compile(r"<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
_WIKI_REF_SELF_RE = re.compile(r"<ref[^>]*?/>", re.IGNORECASE)
_WIKI_DROP_BLOCK_RE = re.compile(
    r"<(nowiki|gallery|math|syntaxhighlight|source|pre|code|score|timeline)"
    r"[^>]*?>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_WIKI_FILE_LINK_RE = re.compile(
    r"\[\[(?:File|Image|sitelen):[^\[\]]*?(?:\[\[[^\[\]]*?\]\][^\[\]]*?)*\]\]",
    re.IGNORECASE,
)
_WIKI_INTERWIKI_LINK_RE = re.compile(r"\[\[[a-z\-]{2,12}:[^\]]*?\]\]")
_WIKI_PIPED_LINK_RE = re.compile(r"\[\[[^\[\]|]+\|([^\[\]]+)\]\]")
_WIKI_SIMPLE_LINK_RE = re.compile(r"\[\[([^\[\]|]+)\]\]")
_WIKI_EXT_LINK_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
_WIKI_BARE_LINK_RE = re.compile(r"\[https?://\S+\]")
_WIKI_HEADER_RE = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
_WIKI_BOLD_ITAL_RE = re.compile(r"'{2,5}")
_WIKI_TABLE_LINE_RE = re.compile(r"^[!|].*$", re.MULTILINE)
_WIKI_TABLE_DELIM_RE = re.compile(r"^\{\||^\|\}|^\|-.*$", re.MULTILINE)
_WIKI_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_WIKI_BLANKLINES_RE = re.compile(r"\n{3,}")


def _strip_templates(text: str) -> str:
    """Iteratively remove ``{{...}}`` blocks (templates can nest)."""
    while True:
        new = re.sub(r"\{\{[^{}]*?\}\}", "", text, flags=re.DOTALL)
        if new == text:
            return new
        text = new


def _strip_wikitext(text: str) -> str:
    text = _WIKI_COMMENT_RE.sub("", text)
    text = _WIKI_REF_BLOCK_RE.sub("", text)
    text = _WIKI_REF_SELF_RE.sub("", text)
    text = _WIKI_DROP_BLOCK_RE.sub("", text)
    text = _strip_templates(text)
    text = _WIKI_FILE_LINK_RE.sub("", text)
    text = _WIKI_INTERWIKI_LINK_RE.sub("", text)
    text = _WIKI_PIPED_LINK_RE.sub(r"\1", text)
    text = _WIKI_SIMPLE_LINK_RE.sub(r"\1", text)
    text = _WIKI_EXT_LINK_RE.sub(r"\1", text)
    text = _WIKI_BARE_LINK_RE.sub("", text)
    text = _WIKI_TABLE_LINE_RE.sub("", text)
    text = _WIKI_TABLE_DELIM_RE.sub("", text)
    text = _WIKI_HEADER_RE.sub(r"\1", text)
    text = _WIKI_BOLD_ITAL_RE.sub("", text)
    text = _WIKI_HTML_TAG_RE.sub("", text)
    text = _WIKI_BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def fetch_tokwiki(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "tokwiki-latest-pages-articles.xml.bz2"
    _http_download(TOKWIKI_URL, archive)
    out = raw_dir / "tokwiki-pages-articles.xml"
    with bz2.open(archive, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)


def iter_tokwiki(raw_dir: Path) -> Iterator[tuple[str, str]]:
    path = raw_dir / "tokwiki-pages-articles.xml"
    if not path.exists():
        return
    # MediaWiki dumps use a versioned XML namespace; match any.
    ns_re = re.compile(r"^\{[^}]+\}")
    title: str | None = None
    ns: str | None = None
    for event, elem in ET.iterparse(path, events=("end",)):
        tag = ns_re.sub("", elem.tag)
        if tag == "title":
            title = (elem.text or "").strip()
        elif tag == "ns":
            ns = (elem.text or "").strip()
        elif tag == "text":
            # Only main namespace (0); skip Talk:, User:, Template:, etc.
            if ns == "0" and title and elem.text:
                if not elem.text.lstrip().upper().startswith("#REDIRECT"):
                    body = _strip_wikitext(elem.text)
                    if body:
                        yield title, body
        elif tag == "page":
            title = None
            ns = None
            elem.clear()


# ---- hecko-yes/toki-ramble ------------------------------------------------

def fetch_toki_ramble(raw_dir: Path) -> None:
    _git_clone("https://github.com/hecko-yes/toki-ramble.git", raw_dir)


def iter_toki_ramble(raw_dir: Path) -> Iterator[tuple[str, str]]:
    toki = raw_dir / "toki"
    if not toki.exists():
        return
    for txt in sorted(toki.rglob("*.txt")):
        body = txt.read_text(encoding="utf-8", errors="replace").strip()
        if body:
            yield str(txt.relative_to(raw_dir)), body


SOURCES: dict[str, Source] = {
    "tatoeba": Source("tatoeba", fetch_tatoeba, iter_tatoeba),
    "poki": Source("poki", fetch_poki, iter_poki),
    "nltk-tp": Source("nltk-tp", fetch_nltk_tp, iter_nltk_tp),
    "lipu": Source("lipu", fetch_lipu, iter_lipu),
    "tp1k": Source("tp1k", fetch_tp1k, iter_tp1k),
    "tokwiki": Source("tokwiki", fetch_tokwiki, iter_tokwiki),
    "toki-ramble": Source("toki-ramble", fetch_toki_ramble, iter_toki_ramble),
}


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

# Split on sentence-final punctuation while keeping sentences reasonably sized.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# A Toki Pona line should mostly consist of ASCII letters + basic punctuation
# and have a high fraction of words drawn from the TP vocab. Capitalized
# words (names) are accepted as-is since they can be any letter sequence.
_TP_LINE_RE = re.compile(r"^[A-Za-z\s.,;:!?'\"-]+$")
_WORD_RE = re.compile(r"[A-Za-z]+")

# Tunable: a sentence passes if at least this fraction of its lowercase
# (non-name) words are Toki Pona vocab. 0.6 keeps real TP (which is mostly
# pu/ku words) while rejecting English (where almost no words match).
TP_WORD_RATIO_THRESHOLD = 0.6


def _looks_like_toki_pona(sentence: str) -> bool:
    if not _TP_LINE_RE.match(sentence):
        return False
    words = _WORD_RE.findall(sentence)
    # Names (capitalized) are skipped from the ratio — they're allowed to be
    # anything. We measure how TP-like the remaining lowercase words are.
    lowercase_words = [w for w in words if w[0].islower()]
    if not lowercase_words:
        return False
    hits = sum(1 for w in lowercase_words if w.lower() in TP_VOCAB)
    return hits / len(lowercase_words) >= TP_WORD_RATIO_THRESHOLD


def _split_sentences(text: str) -> Iterator[str]:
    for chunk in _SENTENCE_SPLIT_RE.split(text):
        s = chunk.strip()
        if s:
            yield s


def write_outputs(selected: list[str]) -> tuple[int, int]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = PROCESSED_DIR / "corpus.jsonl"
    sentences_path = PROCESSED_DIR / "sentences.txt"

    docs = 0
    seen_sentences: set[str] = set()
    with open(corpus_path, "w", encoding="utf-8") as cj, \
         open(sentences_path, "w", encoding="utf-8") as st:
        for name in selected:
            source = SOURCES[name]
            raw_dir = RAW_DIR / name
            if not raw_dir.exists():
                continue
            for doc_id, text in source.iter_docs(raw_dir):
                docs += 1
                cj.write(json.dumps(
                    {"source": name, "id": doc_id, "text": text},
                    ensure_ascii=False,
                ) + "\n")
                for sent in _split_sentences(text):
                    if not _looks_like_toki_pona(sent):
                        continue
                    key = sent.lower()
                    if key in seen_sentences:
                        continue
                    seen_sentences.add(key)
                    st.write(sent + "\n")
    return docs, len(seen_sentences)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", nargs="+", choices=list(SOURCES), default=list(SOURCES),
        help="Which sources to fetch (default: all).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download sources even if their raw_dir already exists.",
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Don't download anything; only re-process existing raw_dirs.",
    )
    args = parser.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for name in args.source:
        source = SOURCES[name]
        raw_dir = RAW_DIR / name
        if args.skip_fetch:
            print(f"[{name}] skip-fetch")
            continue
        if raw_dir.exists() and not args.force:
            print(f"[{name}] already present, skipping (use --force to refetch)")
            continue
        print(f"[{name}] fetching...")
        try:
            source.fetch(raw_dir)
            print(f"[{name}] done -> {raw_dir}")
        except (urllib.error.URLError, subprocess.CalledProcessError, OSError) as e:
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
            if raw_dir.exists():
                shutil.rmtree(raw_dir, ignore_errors=True)
            failures.append(name)

    available = [n for n in args.source if (RAW_DIR / n).exists()]
    if not available:
        print("No sources available to process.", file=sys.stderr)
        return 1

    print(f"\nProcessing: {', '.join(available)}")
    docs, sentences = write_outputs(available)
    print(f"Wrote {docs} documents and {sentences} unique sentences to "
          f"{PROCESSED_DIR.relative_to(REPO_ROOT)}/")

    if failures:
        print(f"\nSources that failed: {', '.join(failures)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
