import os
from collections import defaultdict # This import is no longer used
from clean_text import clean_text
from transformers import pipeline
import re as re
import numpy as np
from typing import Dict, List, Pattern, Tuple
import pandas as pd # Needed for pd.Timestamp


def _simple_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", (name or "")).lower()


# Existing process_nyt_articles function (now renamed to _old_process_nyt_articles if it's no longer the main entry point)
# If you still need this for some reason, ensure it's properly integrated or removed.
# For now, keeping it as is, but consider if `iter_nyt_articles` replaces its core functionality.

def process_nyt_articles(data_dir):
    print(f"Processing NYT articles in: {data_dir}")
    company_articles = defaultdict(list)
    for filename in os.listdir(data_dir):
        if not filename.endswith('.txt'):
            continue
        filepath = os.path.join(data_dir, filename)
        base = filename.rsplit('.', 1)[0]
        m = re.match(r"^(.*?)-(\d+)$", base)
        if m:
            base = m.group(1)
        company_key = _simple_key(base)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            company_articles[company_key].append(content)
    return company_articles


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

def iter_nyt_articles(nyt_dir: str):
    print(f"  Iterating articles in {nyt_dir}")
    for fname in os.listdir(nyt_dir):
        if not fname.endswith(".txt"):
            continue
        base = fname.rsplit(".", 1)[0]
        # If ends with -<digits>, strip the trailing part only
        m = re.match(r"^(.*?)-(\d+)$", base)
        if m:
            base = m.group(1)
        company_key = _simple_key(base)
        fpath = os.path.join(nyt_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
        
        blocks = split_articles(raw)
        print(f"    File {fname}: {len(blocks)} blocks found")
        for block in blocks:
            # The extract_title_body_date function is not available in process_nyt_articles, 
            # so I'll create one based on the logic in the original build_weekly_panel.py.
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            title_line = next((l for l in lines if l.startswith("Title:")), None)
            title = title_line.replace("Title:", "").strip() if title_line else (lines[0] if lines else "")

            ft_idx = next((i for i, l in enumerate(lines) if l.startswith("Full text:")), None)
            body = ""
            if ft_idx is not None:
                text_lines = []
                for l in lines[ft_idx:]:
                    if any(l.startswith(h) for h in META_HEADERS) and not l.startswith("Full text:"):
                        break
                    if l.startswith("Full text:"):
                        l = l[len("Full text:"):].strip()
                    text_lines.append(l)
                body = " ".join(text_lines).strip()
            
            date_line = next((l for l in lines if l.startswith("Publication date:")), None)
            pub_date = pd.NaT
            if date_line:
                pub_date = parse_pub_date(date_line.replace("Publication date:", "").strip())
            text = f"{title} {body}".strip()
            if not text:
                continue
            yield company_key, pub_date, text


def yield_article_texts(file_text: str):
    for block in split_articles(file_text):
        title, body = extract_title_and_fulltext(block)
        text = f"{title} {body}".strip()
        if text:
            yield text


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



def summarize_company_fair(article_probs):
    n = len(article_probs)
    if n == 0:
        return {
            'n_docs': 0,
            'mean_pos': 0.0,
            'mean_neu': 0.0,
            'mean_neg': 0.0,
            'sentiment_score': 0.0,
            'share_pos_docs': 0.0,
            'share_neg_docs': 0.0,
            'non_neutral_share': 0.0,
        }
    return {
        'n_docs': n, # number of articles scored
        'mean_pos': float(np.mean([p['Positive'] for p in article_probs])), # mean positive sentiment score
        'mean_neu':  float(np.mean([p['Neutral'] for p in article_probs])), # mean neutral sentiment score
        'mean_neg': float(np.mean([p['Negative'] for p in article_probs])), # mean negative sentiment score
        'sentiment_score': float(np.mean([p['Positive'] for p in article_probs])) - float(np.mean([p['Negative'] for p in article_probs])), # sentiment score (pos-neg)
        'share_pos_docs': float(np.mean([max(p, key=p.get) == 'Positive' for p in article_probs])), # share of articles with positive sentiment
        'share_neg_docs': float(np.mean([max(p, key=p.get) == 'Negative' for p in article_probs])), # share of articles with negative sentiment
        'non_neutral_share': 1.0 - float(np.mean([p['Neutral'] for p in article_probs])), # share of articles with non-neutral sentiment
    }


# ---------------- Publication date parsing ------------------- #

_ACCEPT_YEARS = {2023, 2024}  # Valid calendar years for this project


def _clean_pub_date_str(s: str) -> str:
    """Fix common OCR/ProQuest glitches like duplicated day digit (e.g. "Mar 1 1, 2023"→"Mar 11, 2023")."""
    s = s.strip()
    # Pattern: Month D D, YYYY  (duplicated single-digit day parts)
    m = re.match(r"(?i)^([A-Za-z]+)\s+(\d)\s+(\d),\s*(\d{4})$", s)
    if m:
        month, d1, d2, year = m.groups()
        day = int(f"{d1}{d2}")
        # Sanity check – compose back only if day is valid (≤31)
        if 1 <= day <= 31:
            return f"{month} {day}, {year}"
    return s


def parse_pub_date(date_str: str):
    """Return pd.Timestamp for allowed years or pd.NaT when unparsable/out-of-range."""
    if not date_str:
        return pd.NaT
    date_str = _clean_pub_date_str(date_str)

    # Try ISO first
    try:
        ts = pd.to_datetime(date_str, errors="raise")
    except Exception:
        # Fallback patterns (e.g., "Jan 31, 2024")
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y"):
            try:
                ts = pd.to_datetime(pd.to_datetime(date_str, format=fmt).date())
                break
            except Exception:
                ts = pd.NaT
    # Year filter
    if ts is pd.NaT or ts.year not in _ACCEPT_YEARS:
        return pd.NaT
    return ts.normalize()


if __name__ == '__main__':
    nyt_data_dir = '/Users/beszabo/bene/topicality-online/data/nyt'
    processed_data = process_nyt_articles(nyt_data_dir)

    # Score all articles for each company and print a summary
    for company_key, files in list(processed_data.items()):
        print(f"Company: {company_key}")
        company_article_probs = []
        total_articles = 0
        for file_text in files:
            articles = list(yield_article_texts(file_text))
            for art_text in articles:
                pre = preprocess_article_text(art_text)
                probs = score_article_company_fair(pre, company_key)
                company_article_probs.append(probs)
            total_articles += len(articles)
        summary = summarize_company_fair(company_article_probs)
        print(f"  Total articles scored: {summary['n_docs']}")
        print(f"  mean_pos: {summary['mean_pos']:.4f}, mean_neu: {summary['mean_neu']:.4f}, mean_neg: {summary['mean_neg']:.4f}")
        print(f"  sentiment_score (pos-neg): {summary['sentiment_score']:.4f}")
        print(f"  share_pos_docs: {summary['share_pos_docs']:.4f}, share_neg_docs: {summary['share_neg_docs']:.4f}")
        print(f"  non_neutral_share: {summary['non_neutral_share']:.4f}")
