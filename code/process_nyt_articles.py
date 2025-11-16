import os
from collections import defaultdict # This import is no longer used
from transformers import pipeline
import re as re
import numpy as np
from typing import Dict, List, Pattern, Tuple
import pandas as pd # Needed for pd.Timestamp


def _simple_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", (name or "")).lower()

# Use explicit tokenizer, return all scores via top_k=None
_clf = None

def get_finbert():
    global _clf
    if _clf is None:
        _clf = pipeline(
            "text-classification",
            model="yiyanghkust/finbert-tone",
            tokenizer="yiyanghkust/finbert-tone",
            top_k=None
        )
    return _clf

# ---------------- Parsing helpers for ProQuest-style blocks ---------------- #
META_HEADERS = {
    "Subject:", "Location:", "People:", "Company / organization:", "URL:", "Title:",
    "Publication title:", "Pages:", "Publication year:", "Publication date:", "Section:",
    "Publisher:", "Place of publication:", "Country of publication:", "Publication subject:",
    "ISSN:", "Source type:", "Language of publication:", "Document type:",
    "ProQuest document ID:", "Document URL:", "Copyright:", "Last updated:", "Database:"
}

SEP_LINE = "_" * 60 # Define SEP_LINE here as it is used by split_articles

def split_articles(raw_text: str):
    parts = [p.strip() for p in raw_text.split("____________________________________________________________") if p.strip()]
    return parts


def extract_title_and_fulltext(block: str):
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    title_line = next((l for l in lines if l.startswith("Title:")), None)
    title = title_line.replace("Title:", "").strip() if title_line else (lines[0] if lines else "")
    # find Full text:
    ft_idx = next((i for i, l in enumerate(lines) if l.startswith("Full text:")), None)
    if ft_idx is None:
        return title, ""
    text_lines = []
    for l in lines[ft_idx:]:
        if any(l.startswith(h) for h in META_HEADERS) and not l.startswith("Full text:"):
            break
        if l.startswith("Full text:"):
            l = l[len("Full text:"):].strip()
        text_lines.append(l)
    return title, " ".join(text_lines).strip()


# ---------------- Synonyms & FinBERT helpers ---------------- #

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _token_to_pattern(token: str) -> Pattern:
    tok = token.strip().strip('"')
    if not tok:
        # match nothing
        return re.compile(r"a^")
    escaped = re.escape(tok)
    escaped = re.sub(r"\\ ", r"\\s+", escaped)
    pattern = r"(?i)(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])"
    return re.compile(pattern)


def _parse_synonyms_line(line: str) -> List[str]:
    s = line.strip()
    if not s or s.startswith('#'):
        return []
    # Prefer content within the first parenthesis if present
    if s.startswith('(') and ')' in s:
        s = s[1:s.find(')')]
    # Drop any advanced boolean bits after AND / NOT
    s = re.split(r"\s+AND\s+", s, flags=re.IGNORECASE)[0]
    s = re.split(r"\s+NOT\s+", s, flags=re.IGNORECASE)[0]
    parts = [p.strip() for p in re.split(r"\s+OR\s+", s, flags=re.IGNORECASE) if p.strip()]
    # Strip surrounding quotes
    parts = [p.strip().strip('"') for p in parts if p.strip()]
    return parts


def load_company_synonyms(syn_path: str) -> Dict[str, List[Pattern]]:
    patterns: Dict[str, List[Pattern]] = {}
    if not os.path.exists(syn_path):
        return patterns
    with open(syn_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = _parse_synonyms_line(line)
            if not tokens:
                continue
            canon = _simple_key(tokens[0])
            pats = [_token_to_pattern(t) for t in tokens]
            patterns[canon] = pats
    return patterns


_ALIAS_PATTERNS: Dict[str, List[Pattern]] = load_company_synonyms(
    os.path.join(_repo_root(), 'code', 'company_synonyms.txt')
)

def preprocess_article_text(article_content):
    """
    Preprocess an article text for BERT input: collapse whitespace, remove underscores.
    Returns cleaned string (do NOT lowercase or strip punctuation).
    """
    text = article_content.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Fair, relevance-filtered aggregation (keeps Neutral)
def _finbert_doc_average(texts, classifier):
    print(f"    FinBERT: {len(texts)} sentences for classification")
    outs = classifier(texts, truncation=True, max_length=512, top_k=None)
    agg = {'Positive': 0.0, 'Neutral': 0.0, 'Negative': 0.0}
    for out in outs:
        for d in out:
            agg[d['label']] += d['score']
    n = len(outs) if len(outs) > 0 else 1
    for k in agg:
        agg[k] /= n
    return agg

def sentences_with_company(text: str, company_key: str, max_sentences: int = 200):
    sents = re.split(r'(?<=[.!?])\s+', text)
    print(f"    Total sentences before filtering: {len(sents)}")
    pats = _ALIAS_PATTERNS.get(company_key, None)
    if pats:
        hits = [s for s in sents if any(p.search(s) for p in pats)]
    else:
        hits = [s for s in sents if company_key in _simple_key(s)]
    print(f"    Sentences matching company '{company_key}': {len(hits)}")
    return (hits[:max_sentences] or sents[:max_sentences])


def score_article_company_fair(text: str, company_key: str):
    # Use full text for sentence extraction, as alias matching already filters for relevance.
    sents = sentences_with_company(text, company_key)
    return _finbert_doc_average(sents, get_finbert())