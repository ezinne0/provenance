"""
Helpers for calling Claude to infer product fields from a URL string alone.
"""

from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


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

if __name__ == "__main__":
    urls = [
        "https://www.zara.com/us/en/regular-fit-textured-weave-suit-pT9960345005.html?v1=539928691",
        "https://www.depop.com/products/merchoutlet08-brand-new-air-force-one-d8f6/",
        "https://poshmark.com/listing/Levis-80s-Mom-Shorts-69e2720424d2e76a97493d38"
    ]
    for url in urls:
        print(get_product_name(url))
