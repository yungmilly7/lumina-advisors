"""News ingestion via Google News RSS (no API key needed) plus a lightweight
lexicon-based sentiment/event tagger.

This is deliberately not a heavy NLP pipeline: it's a transparent, auditable
keyword scorer so every score can be explained in one sentence ("headline
contains 'beats estimates' and 'record revenue'"). Swap in a real sentiment
model or an LLM classifier later without changing the rest of the pipeline.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from app.dataclients.httpjson import get_text

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LuminaAdvisorsBot/1.0)"}
RSS_URL = "https://news.google.com/rss/search"

POSITIVE_WORDS = {
    "beats": 1.0, "beat": 1.0, "record": 0.8, "surge": 0.9, "soar": 1.0,
    "jumps": 0.8, "rally": 0.8, "upgrade": 0.9, "upgraded": 0.9, "raises": 0.6,
    "raised guidance": 1.0, "strong demand": 0.8, "outperform": 0.7,
    "buyback": 0.5, "partnership": 0.4, "deal": 0.3, "launch": 0.4,
    "launches": 0.4, "approval": 0.7, "approved": 0.7, "expands": 0.4,
    "growth": 0.5, "profit": 0.4, "wins": 0.5, "acquire": 0.3, "acquires": 0.3,
}
NEGATIVE_WORDS = {
    "misses": -1.0, "miss": -1.0, "plunge": -1.0, "plunges": -1.0,
    "falls": -0.6, "fall": -0.6, "drop": -0.6, "drops": -0.6, "slump": -0.8,
    "downgrade": -0.9, "downgraded": -0.9, "cuts": -0.6, "cut guidance": -1.0,
    "recall": -0.8, "lawsuit": -0.7, "sues": -0.6, "sued": -0.6,
    "investigation": -0.7, "probe": -0.6, "layoffs": -0.7, "job cuts": -0.7,
    "warns": -0.6, "warning": -0.6, "weak demand": -0.8, "shortage": -0.5,
    "delay": -0.4, "delays": -0.4, "fraud": -1.0, "fine": -0.5, "fined": -0.5,
    "underperform": -0.7, "sell-off": -0.8, "selloff": -0.8, "bankruptcy": -1.2,
}
EVENT_PATTERNS = {
    "earnings": r"\b(earnings|eps|quarterly results|q[1-4] results)\b",
    "product_launch": r"\b(unveils|launch(es)?|debuts|introduces|announces new)\b",
    "product_delay": r"\b(delay(s|ed)?|pushback|postpone)\b",
    "deal_partnership": r"\b(partners? with|deal with|agreement|joint venture|teams up)\b",
    "ma": r"\b(acquir(e|es|ed|ing)|merger|buyout|takeover|to buy)\b",
    "regulatory": r"\b(regulator|antitrust|fda|sec probe|investigation|lawsuit|fine[ds]?)\b",
    "guidance": r"\b(guidance|forecast|outlook)\b",
    "analyst_action": r"\b(upgrade[ds]?|downgrade[ds]?|price target|initiat(e|es|ed) coverage)\b",
    "layoffs": r"\b(layoffs?|job cuts|workforce reduction)\b",
    "recall": r"\bre-?call\b",
}


def fetch_headlines(ticker: str, company_name: str, limit: int = 15) -> list[dict]:
    query = f'"{company_name}" OR {ticker} stock'
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    text = get_text(RSS_URL, params=params, headers=_HEADERS)
    root = ET.fromstring(text)

    rows = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate")
        source_el = item.find("source")
        source = source_el.text if source_el is not None else None
        try:
            published = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat()
        except Exception:
            published = datetime.now(timezone.utc).isoformat()
        sentiment, tags = score_headline(title)
        rows.append(
            {
                "published": published,
                "headline": title,
                "source": source,
                "url": link,
                "sentiment": sentiment,
                "event_tags": tags,
            }
        )
    return rows


def score_headline(headline: str) -> tuple[float, list[str]]:
    text = headline.lower()
    score = 0.0
    hits = 0
    for phrase, weight in {**POSITIVE_WORDS, **NEGATIVE_WORDS}.items():
        if phrase in text:
            score += weight
            hits += 1
    normalized = max(-1.0, min(1.0, score / 2.0)) if hits else 0.0

    tags = [tag for tag, pattern in EVENT_PATTERNS.items() if re.search(pattern, text)]
    return round(normalized, 3), tags
