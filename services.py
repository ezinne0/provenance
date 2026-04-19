"""
Helpers for calling Claude to infer product fields from a URL string alone,
and for fetching review-adjacent snippets via SerpAPI.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
_MAX_REVIEW_TEXT_CHARS = 100_000

# Helper function to parse the assistant's JSON response
def _parse_assistant_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Model did not return a JSON object.")
    return data


def _parse_assistant_json_array(text: str) -> list[Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Model did not return a JSON array.")
    return data


def get_product_name(url: str) -> dict[str, str]:
    """
    Send only the URL to Claude and ask it to infer ``product_name``, ``brand``,
    and ``category`` from the URL path, hostname, and query structure.
    Returns the same dict shape as before (JSON-shaped).
    """
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    url = (url or "").strip()
    if not url:
        raise ValueError("url is required.")

    model = (os.getenv("ANTHROPIC_MODEL") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    client = Anthropic(api_key=api_key)
    system = (
        "You infer product information from a product listing URL alone — no page content is provided. "
        "Use the domain, path segments, slug words, SKU-like segments, and query parameters as clues. "
        "Respond with a single JSON object and nothing else — no markdown fences. "
        'Keys: "product_name", "brand", "category" (all strings). '
        "Use empty string if unknown. Category should be short (e.g. sneakers, kettle)."
    )
    user = f"Product page URL:\n{url}"

    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = ""
    for block in msg.content:
        if hasattr(block, "text"):
            text += block.text

    data = _parse_assistant_json(text)
    return {
        "product_name": str(data.get("product_name", "")).strip(),
        "brand": str(data.get("brand", "")).strip(),
        "category": str(data.get("category", "")).strip(),
    }

# Helper function to get productreviews from across Google
# this is the entire public internet's opinion on the product, surfaced through Google Search
def get_reviews(product_name: str, brand: str) -> str:
    """
    Search Google via SerpAPI for ``{brand} {product_name} reviews`` and return
    all organic result snippets concatenated with newlines.
    """
    api_key = (os.getenv("SERP_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("SERP_API_KEY is not set.")

    pn = (product_name or "").strip()
    br = (brand or "").strip()
    q = f"{br} {pn} reviews".strip()
    if not q or q == "reviews":
        raise ValueError("product_name and brand cannot both be empty.")

    resp = requests.get(
        "https://serpapi.com/search",
        params={"q": q, "api_key": api_key, "engine": "google"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        raise ValueError(str(payload["error"]))

    organic = payload.get("organic_results") or []
    snippets: list[str] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        snippet = (item.get("snippet") or "").strip()
        if snippet:
            snippets.append(snippet)

    return "\n".join(snippets)


def get_product_image(product_name: str, brand: str) -> str:
    """
    Search Google Images via SerpAPI for ``{brand} {product_name}`` and return
    the ``original`` URL of the first image result.
    """
    api_key = (os.getenv("SERP_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("SERP_API_KEY is not set.")

    pn = (product_name or "").strip()
    br = (brand or "").strip()
    q = f"{br} {pn}".strip()
    if not q:
        raise ValueError("product_name and brand cannot both be empty.")

    resp = requests.get(
        "https://serpapi.com/search",
        params={
            "q": q,
            "api_key": api_key,
            "engine": "google_images",
            "num": 1,
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        raise ValueError(str(payload["error"]))

    images = payload.get("images_results") or []
    if not images:
        raise ValueError("No image results returned for this product.")

    first = images[0]
    if not isinstance(first, dict):
        raise ValueError("Unexpected image result format.")

    original = (first.get("original") or "").strip()
    if not original:
        raise ValueError("First image result had no original URL.")

    return original


# Helper function to normalize the keyword list
def _normalize_keyword_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        s = str(x).strip().lower()
        s = re.sub(r"[^a-z0-9\-]+", "", s)
        if s and s not in out:
            out.append(s)
        if len(out) >= 5:
            break
    while len(out) < 5:
        out.append("unknown")
    return out[:5]


def _normalize_star_rating(raw: Any) -> float:
    """Clamp to [1.0, 5.0] and round to nearest 0.5."""
    try:
        x = float(raw)
    except (TypeError, ValueError):
        x = 3.0
    x = max(1.0, min(5.0, x))
    return round(x * 2) / 2


# Helper function to synthesize the review text
def synthesize(review_text: str, product_name: str, brand: str) -> dict[str, Any]:
    """
    Send aggregated review text to Claude and return ``star_rating``, ``fit``,
    ``durability``, ``quality``, and ``keywords`` (5 items).
    """
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    rt = (review_text or "").strip()
    if len(rt) > _MAX_REVIEW_TEXT_CHARS:
        rt = rt[:_MAX_REVIEW_TEXT_CHARS]

    pn = (product_name or "").strip()
    br = (brand or "").strip()
    model = (os.getenv("ANTHROPIC_MODEL") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    client = Anthropic(api_key=api_key)
    system = (
        "You synthesize shopper-relevant insights from aggregated web search snippets about a product. "
        "Respond with a single JSON object only — no markdown, no code fences, no explanation or text outside the JSON. "
        "Exactly these five keys:\n"
        '- "star_rating": number (float) between 1.0 and 5.0 inclusive, rounded to the nearest 0.5, '
        "representing overall sentiment implied by the snippets (1 = very negative, 5 = very positive). "
        "Use values like 3.5 or 4.0 only — step by 0.5.\n"
        '- "fit": string, exactly one sentence on sizing, cut, comfort, and how it wears.\n'
        '- "durability": string, exactly one sentence on longevity — wear, materials, construction.\n'
        '- "quality": string, exactly one sentence on build quality, finish, materials, or defects.\n'
        '- "keywords": array of exactly 5 strings — concrete descriptor words people use (e.g. soft, creasing, bulky). '
        "No common stop words (the, and, very, good, bad as filler alone).\n"
        "If evidence is thin, say so briefly in the relevant string fields and lean toward a middling star_rating (around 3.0)."
    )
    user = (
        f"Product name: {pn}\n"
        f"Brand: {br}\n\n"
        f"Review / search snippet text:\n{rt}"
    )

    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        # set to 0 for deterministic output; tbd if im keeping this
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = ""
    for block in msg.content:
        if hasattr(block, "text"):
            text += block.text

    data = _parse_assistant_json(text)

    return {
        "star_rating": _normalize_star_rating(data.get("star_rating")),
        "fit": str(data.get("fit", "")).strip(),
        "durability": str(data.get("durability", "")).strip(),
        "quality": str(data.get("quality", "")).strip(),
        "keywords": _normalize_keyword_list(data.get("keywords")),
    }


def _normalize_one_liner(raw: Any, max_chars: int = 220) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    for sep in (". ", "! ", "? "):
        if sep in s:
            s = s.split(sep, 1)[0] + sep[0]
            break
    if len(s) > max_chars:
        s = s[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return s


def _normalize_similar_item(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Each similar product must be a JSON object.")
    return {
        "product_name": str(raw.get("product_name", "")).strip(),
        "brand": str(raw.get("brand", "")).strip(),
        "star_rating": _normalize_star_rating(raw.get("star_rating")),
        "one_liner": _normalize_one_liner(raw.get("one_liner")),
    }


def get_similar_items(product_name: str, brand: str, category: str) -> list[dict[str, Any]]:
    """
    Call Claude to suggest exactly three similar products.

    Returns a list of three dicts with keys ``product_name``, ``brand``,
    ``star_rating`` (1.0–5.0 step 0.5), and ``one_liner`` (one short sentence).
    """
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    pn = (product_name or "").strip()
    br = (brand or "").strip()
    cat = (category or "").strip()
    if not pn and not br and not cat:
        raise ValueError("At least one of product_name, brand, or category is required.")

    model = (os.getenv("ANTHROPIC_MODEL") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    client = Anthropic(api_key=api_key)
    system = (
        "You recommend similar retail products a shopper might compare to the reference item. "
        "Output must be a single JSON array and nothing else — no markdown, no code fences, "
        "no commentary before or after the array. "
        "The array must contain exactly 3 elements. Each element is one JSON object with exactly these keys:\n"
        '- "product_name" (string)\n'
        '- "brand" (string)\n'
        '- "star_rating" (number): a plausible overall rating between 1.0 and 5.0 inclusive, '
        "rounded to the nearest 0.5 only (e.g. 4.0, 3.5, 2.0).\n"
        '- "one_liner" (string): at most one sentence on why it is a sensible alternative or what it is known for.\n'
        "Use realistic product names and brands when possible; ratings are illustrative."
    )
    user = (
        f"Reference product_name: {pn}\n"
        f"Reference brand: {br}\n"
        f"Category: {cat}\n"
    )

    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = ""
    for block in msg.content:
        if hasattr(block, "text"):
            text += block.text

    items = _parse_assistant_json_array(text)
    if len(items) < 3:
        raise ValueError(f"Expected 3 similar products, got {len(items)}.")
    normalized = [_normalize_similar_item(x) for x in items[:3]]
    return normalized


# # Testing get_product_name() function with a list of URLs
# if __name__ == "__main__":
#     urls = [
#         "https://www.zara.com/us/en/regular-fit-textured-weave-suit-pT9960345005.html?v1=539928691",
#         "https://www.depop.com/products/merchoutlet08-brand-new-air-force-one-d8f6/",
#         "https://poshmark.com/listing/Levis-80s-Mom-Shorts-69e2720424d2e76a97493d38"
#     ]
#     for url in urls:
#         print(get_product_name(url))

# Testing get_reviews() function with a list of product names and brands
# if __name__ == "__main__":
#     product_names = [
#         "Air Force 1",
#         "Textured Weave Suit",
#         "Levi's 80s Mom Shorts"
#     ]
#     brands = [
#         "Nike",
#         "Zara",
#         "Levi's"
#     ]
#     for product_name, brand in zip(product_names, brands):
#         print(get_reviews(product_name, brand))

# Testing synthesize() function with a list of fake reviews
# if __name__ == "__main__":
#     fake_reviews = """
#     The Nike Air Force 1 fits true to size but runs large. 
#     Creases badly after a few wears especially at the toe box.
#     Durable leather upper holds up well over time but the sole can yellow.
#     Comfortable for casual wear but not for long walks.
#     Bulky silhouette, iconic design, pairs well with everything.
#     Some users report the laces fray quickly.
#     Great value for the price, very versatile shoe.
#     """
#     result = synthesize(fake_reviews, "Air Force 1", "Nike")
#     print(result)

if __name__ == "__main__":
    result = get_product_image("Air Force 1", "Nike")
    print(result)
