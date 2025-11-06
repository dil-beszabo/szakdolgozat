import os
from collections import defaultdict
from clean_text import clean_text
from transformers import pipeline


def process_nyt_articles(data_dir):
    company_articles = defaultdict(list)
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            company_name = filename.split('-')[0].split('.')[0] # Handles 'company.txt' and 'company-1.txt'
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                company_articles[company_name].append(content)
    return company_articles


# Use explicit tokenizer, return all scores via top_k=None
clf = pipeline(
    "text-classification",
    model="yiyanghkust/finbert-tone",
    tokenizer="yiyanghkust/finbert-tone",
    top_k=None
)


# ---------------- Parsing helpers for ProQuest-style blocks ---------------- #
META_HEADERS = {
    "Subject:", "Location:", "People:", "Company / organization:", "URL:", "Title:",
    "Publication title:", "Pages:", "Publication year:", "Publication date:", "Section:",
    "Publisher:", "Place of publication:", "Country of publication:", "Publication subject:",
    "ISSN:", "Source type:", "Language of publication:", "Document type:",
    "ProQuest document ID:", "Document URL:", "Copyright:", "Last updated:", "Database:"
}


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


def yield_article_texts(file_text: str):
    for block in split_articles(file_text):
        title, body = extract_title_and_fulltext(block)
        text = f"{title} {body}".strip()
        if text:
            yield text


# ---------------- FinBERT helpers ---------------- #

def preprocess_article_text(article_content):
    """
    Preprocess an article text for BERT input: collapse whitespace, remove underscores.
    Returns cleaned string (do NOT lowercase or strip punctuation).
    """
    import re
    text = article_content.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def finbert_paragraph_avg(text, clf, max_paragraphs=6):
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    # If no paragraph breaks present, fall back to sentence-ish chunks
    if not paras:
        paras = [text]
    # Limit to first N paragraphs for speed; BERT truncation still applies per paragraph
    paras = paras[:max_paragraphs]
    outs = clf(paras, truncation=True, max_length=512, top_k=None)
    agg = {}
    for out in outs:
        for d in out:
            agg[d['label']] = agg.get(d['label'], 0.0) + d['score']
    for k in agg:
        agg[k] /= len(outs)
    return agg


def aggregate_article_scores(article_scores):
    agg = defaultdict(float)
    n = 0
    for sc in article_scores:
        if not sc:
            continue
        n += 1
        for k, v in sc.items():
            agg[k] += v
    if n > 0:
        for k in list(agg.keys()):
            agg[k] /= n
    return n, dict(agg)


if __name__ == '__main__':
    nyt_data_dir = '/Users/beszabo/bene/topicality-online/data/nyt'
    processed_data = process_nyt_articles(nyt_data_dir)
 
    # Score all articles for each company and print a summary
    for company, files in list(processed_data.items()):
        print(f"Company: {company}")
        company_article_scores = []
        total_articles = 0
        for i, file_text in enumerate(files):
            articles = list(yield_article_texts(file_text))
            for art_text in articles:
                pre = preprocess_article_text(art_text)
                sc = finbert_paragraph_avg(pre, clf)
                company_article_scores.append(sc)
            total_articles += len(articles)
        n, avg = aggregate_article_scores(company_article_scores)
        print(f"  Total articles scored: {n}")
        print(f"  Avg sentiment: {avg}")
