"""Fetch NYT monthly archives (free API) and convert to ProQuest-style txt blocks.

This fills the 2024-02…2024-12 gap without needing paid Historical Article
Search. The Article Archive API returns *all* articles for a month; we post-
filter by company tokens from ``data/text_lists/company_synonyms.txt`` using the
same logic as the rest of the pipeline.

Example usage:

    NYT_API_KEY=your_key uv run python -m code.fetch_nyt_archive \
        --start 2024-02 --end 2024-12 --companies amazon apple

Outputs: ``data/nyt/archive_<YYYY-MM>_<company>.txt``
These are identical in format to the existing ProQuest dumps, so
``build_weekly_panel.py`` will ingest them automatically.
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
import textwrap
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
NYT_DIR = REPO_ROOT / "data" / "nyt"
SYN_PATH = REPO_ROOT / "data" / "text_lists" / "company_synonyms.txt"

SEP_LINE = "_" * 60  # same as split_articles


# ----------------- Synonym helpers (copied from fetch_nyt_api) --------------


def _parse_synonyms_line(line: str) -> List[str]:
    s = line.strip()
    if not s or s.startswith("#"):
        return []
    if s.startswith("(") and ")" in s:
        s = s[1 : s.find(")")]
    s = s.split(" AND ")[0].split(" and ")[0]
    s = s.split(" NOT ")[0].split(" not ")[0]
    parts = [p.strip() for p in s.replace(" OR ", " or ").split(" or ") if p.strip()]
    return [p.strip('"') for p in parts]


def _simple_key(name: str) -> str:
    return "".join(ch for ch in (name or "") if ch.isalnum()).lower()


def load_company_tokens(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            toks = _parse_synonyms_line(line)
            if not toks:
                continue
            out[_simple_key(toks[0])] = toks
    return out


# ----------------- Archive API helpers --------------------------------------


API_URL_TMPL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"


def month_iter(start_ym: str, end_ym: str):
    y0, m0 = map(int, start_ym.split("-"))
    y1, m1 = map(int, end_ym.split("-"))
    cur_y, cur_m = y0, m0
    while (cur_y, cur_m) <= (y1, m1):
        yield cur_y, cur_m
        # increment month
        if cur_m == 12:
            cur_y += 1
            cur_m = 1
        else:
            cur_m += 1


def matches_company(doc: dict, tokens: List[str]) -> bool:
    haystack = " ".join(
        [
            str(doc.get("headline", {}).get("main", "")),
            str(doc.get("snippet", "")),
            str(doc.get("abstract", "")),
        ]
    ).lower()
    return any(tok.lower() in haystack for tok in tokens if tok)


def render_block(doc: dict) -> str:
    headline = (doc.get("headline") or {}).get("main") or ""
    pub_date = doc.get("pub_date", "")
    if pub_date:
        pub_date = pub_date.split("T")[0]
    url = doc.get("web_url", "")
    lead = doc.get("lead_paragraph") or doc.get("abstract") or doc.get("snippet") or ""
    full_text = lead.replace("\n", " ").strip()
    return "\n".join(
        [
            f"Title: {headline}",
            "",
            f"Publication date: {pub_date}",
            f"URL: {url}",
            "",
            f"Full text: {full_text}",
            "",
        ]
    )


def fetch_month(year: int, month: int, api_key: str) -> list[dict]:
    url = API_URL_TMPL.format(year=year, month=month)
    params = {"api-key": api_key}
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Archive API {year}-{month:02d} failed: {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    return data.get("response", {}).get("docs", []) or []


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Download NYT monthly archives and split into per-company txt files.",
        epilog=textwrap.dedent(
            """
            Example:
              NYT_API_KEY=abc python -m code.fetch_nyt_archive --start 2024-02 --end 2024-12 --companies amazon apple
            """
        ),
    )
    p.add_argument("--start", required=True, help="YYYY-MM inclusive start month")
    p.add_argument("--end", required=True, help="YYYY-MM inclusive end month")
    p.add_argument("--companies", nargs="*", help="Subset of companies to keep (keys from synonyms file)")
    p.add_argument("--synonyms", default=str(SYN_PATH), help="Path to company_synonyms.txt")
    p.add_argument("--api-key", default=os.environ.get("NYT_API_KEY"), help="NYT API key")
    args = p.parse_args()

    if not args.api_key:
        p.error("NYT_API_KEY env var or --api-key is required")

    try:
        datetime.strptime(args.start, "%Y-%m")
        datetime.strptime(args.end, "%Y-%m")
    except ValueError:
        p.error("--start/--end must be YYYY-MM")

    tokens_map = load_company_tokens(Path(args.synonyms))
    targets = set(args.companies or tokens_map.keys())

    NYT_DIR.mkdir(parents=True, exist_ok=True)

    for year, month in month_iter(args.start, args.end):
        print(f"Fetching archive {year}-{month:02d}…", flush=True)
        docs = fetch_month(year, month, args.api_key)
        if not docs:
            print("  no docs returned, skipping")
            continue

        per_company: Dict[str, List[str]] = {k: [] for k in targets}
        for doc in docs:
            for company in targets:
                toks = tokens_map.get(company)
                if not toks:
                    continue
                if matches_company(doc, toks):
                    per_company[company].append(render_block(doc))

        # write out
        for company, blocks in per_company.items():
            if not blocks:
                continue
            out_path = NYT_DIR / f"{company}_{year}{month:02d}.txt"
            with out_path.open("w", encoding="utf-8") as f:
                for b in blocks:
                    f.write(b)
                    f.write("\n" + SEP_LINE + "\n\n")
            print(f"  {company}: {len(blocks)} articles -> {out_path}")

        # polite rate-limit: 1 request/sec as per docs
        time.sleep(1.1)


if __name__ == "__main__":
    main()
