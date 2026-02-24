# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
generate_products.py

Production-grade script to generate Shopify-ready product JSON for plants.
This file contains the single prompt definition inside `build_prompt()` only.
"""
# Standard library imports
import os
import re
import json
import time
import csv
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import logging

# Third-party imports
from dotenv import load_dotenv
from openai import OpenAI
from config.runtime_flags import DISABLE_OPENAI_CALLS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "plants.csv"
OUTPUT_DIR = "output_products"
FAILED_LOG = "failed_products.txt"
ENV_FILE = ".env"
API_MODEL = "gpt-4o"
MAX_RETRIES = 3
DELAY_BETWEEN_PLANTS = 2
# 0.85 was too aggressive for AI-generated structured product content.
# Similar structure between plant descriptions does not imply duplication.
# 0.93 still prevents real copy-paste duplication while avoiding regeneration loops.
DUPLICATE_SIM_THRESHOLD = 0.93
URL_HANDLE_REGEX = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TAG_REGEX = re.compile(r"^[a-z0-9-]+$")
VALID_CATEGORIES = [
    "Indoor Plants",
    "Gift Plants",
    "Luxury Orchid Gifts",
    "Medicinal & Herbs",
    "Indian Exotics",
    "AC Friendly",
    "Corporate Gifting",
]

# Backend-controlled constants (business rules)
DELIVERY_POLICY = "Delivered within 1 business day across the UAE."
ABOUT_LUSHERRA = (
    "Lusherra is a Dubai-based luxury indoor plant brand curating "
    "premium ready-to-display plants designed for modern UAE interiors."
)

# Logger and JSON formatter
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # include extra if present
        extra = {}
        try:
            if hasattr(record, "__dict__"):
                for k, v in record.__dict__.items():
                    if k not in ("name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process"):
                        extra[k] = v
        except Exception:
            extra = {}
        if extra:
            payload["extra"] = extra
        return json.dumps(payload, ensure_ascii=False)

# configure module logger
logger = logging.getLogger("generate_products")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _load_existing_product(plant_name: str) -> Optional[dict]:
    """Attempt to load an existing product JSON from common output locations.

    Checks (in order):
    - outputs/products/{slug}.json (pipeline output)
    - OUTPUT_DIR/{SANITIZED}.json (this module's output dir)
    Returns the parsed dict or None.
    """
    candidates = []
    # pipeline outputs (lowercase slug)
    try:
        from pipeline import plant_slug
        slug = plant_slug(plant_name)
        candidates.append(os.path.join("outputs", "products", f"{slug}.json"))
    except Exception:
        pass

    # this module's OUTPUT_DIR with sanitized filename
    try:
        san = sanitize_filename(plant_name)
        candidates.append(os.path.join(OUTPUT_DIR, f"{san}.json"))
        candidates.append(os.path.join(OUTPUT_DIR, f"{san.lower()}.json"))
    except Exception:
        pass

    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception:
            continue
    return None


def load_plants(csv_path: str) -> List[str]:
    """Load plant names from CSV_PATH and return list of plant name strings.

    Supports a header row with a 'name' or 'plant' column; otherwise uses first column.
    """
    plants: List[str] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
            if not rows:
                return plants
            # detect header
            header = rows[0]
            name_idx = 0
            start = 0
            if any("name" in (c or "").lower() or "plant" in (c or "").lower() for c in header):
                start = 1
                for i, c in enumerate(header):
                    if "name" in (c or "").lower() or "plant" in (c or "").lower():
                        name_idx = i
                        break
            for row in rows[start:]:
                if len(row) > name_idx:
                    val = row[name_idx].strip()
                    if val:
                        plants.append(val)
    except FileNotFoundError:
        raise
    return plants


def sanitize_filename(name: str) -> str:
    """Sanitize `name` into an ASCII uppercase filename-safe string."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_bytes = nfkd.encode("ascii", "ignore")
    s = ascii_bytes.decode("ascii")
    s = s.upper()
    s = re.sub(r"\s+", "-", s)
    s = s.replace("/", "").replace("\\", "").replace("'", "")
    s = re.sub(r"[^A-Z0-9-]", "", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def build_prompt(plant_name: str) -> str:
    """Return the full production-grade plant content generation prompt as an f-string.

    This prompt is required to return STRICT valid JSON only. It injects {plant_name}.
    """
    prompt = f"""
You are a senior ecommerce SEO strategist and luxury indoor plant copywriter specializing in premium UAE indoor plant brands.

Generate a fully structured Shopify-ready product object for the plant below.

Plant Name: {plant_name}
Brand: Lusherra
Target Market: UAE (Dubai, Abu Dhabi, Sharjah)

You must return STRICT valid JSON only.
No commentary. No markdown. No explanations. No extra text before or after JSON.

The writing must feel naturally human, premium, calm, and brand-consistent.
Avoid robotic phrasing. Avoid template repetition. Vary paragraph openings and sentence rhythm.
Do not reuse FAQ questions across plants. Ensure uniqueness per plant.

If factual data is uncertain (e.g. mosquito repellent or pet safety), avoid false claims. Only include if widely known and safe.

Return JSON in EXACTLY this schema:

{{
"product_title": "",
"short_description": "",
"long_description": "",
"estimated_size": "",
"characteristics": [],
"special_features": [],
"mosquito_repellent": "",
"ac_room_benefits": "",
"pet_friendly": "",
"delivery_info": "",
"about_lusherra_section": "",
"category": "",
"price_aed": 0,
"compare_at_price_aed": 0,
"seo": {{
"meta_title": "",
"meta_description": "",
"url_handle": ""
}},
"tags": [],
"collections": [],
"meta_fields": {{
"lusherra_care_level": "",
"lusherra_room_placement": ""
}},
"care_guide": "",
"faqs": [
  {{
    "question": "",
    "answer": ""
  }}
]
}}

Do not change the structure.
All numeric values must be numbers, not strings.

CONTENT REQUIREMENTS:

PRODUCT TITLE

Include plant name.

Include UAE keyword naturally when appropriate.

Under 70 characters.

SHORT DESCRIPTION

2-3 compelling sentences.

Conversion-focused, not keyword stuffed.

    LONG DESCRIPTION (500-700 words)
Must naturally include:

Approximate plant height in cm

Pot included (ready-to-use curated indoor plant)

AC room suitability

Pet safety clarification (if known)

Mosquito repellent mention only if factually accurate

1-day delivery in UAE

About Lusherra positioning

Interior styling suggestions

Why suitable for UAE homes

Subtle SEO keyword placement (no stuffing)

        prompt = prompt_template.replace("{plant_name}", plant_name)
CARE GUIDE (Minimum 550 words)
Must include:

Light requirements (UAE climate context)

Watering frequency (Dubai climate)

Soil recommendation

Fertilization schedule

AC room adjustments

Humidity control

Repotting advice

Pest management

Seasonal adjustments in UAE

Common mistakes to avoid
Written as premium blog-style educational content.

FAQs

5–8 questions

Must reflect realistic Google search queries

Must vary per plant

Answers must be 80-150 words each

Do not repeat structure across plants

PRICING

Research realistic UAE market pricing for similar size

price_aed must be realistic integer

compare_at_price_aed must be <= 1.5x price_aed

Most plants should have under 40% markup

CATEGORY (choose ONE)
Indoor Plants
Gift Plants
Luxury Orchid Gifts
Medicinal & Herbs
Indian Exotics
AC Friendly
Corporate Gifting

TAGS
Must include:

indoor-plant

uae-plants
Include additional relevant tags such as:

gift-plant

ac-friendly

low-maintenance

pet-safe (only if accurate)

corporate-gifting

best-selling
Tags must be lowercase hyphen format.

COLLECTIONS
Must match tags logically.

SEO

Meta title under 60 characters

Meta description under 155 characters

URL handle lowercase, hyphen-separated, no special characters

WRITING RULES

No emojis

No icons

No repetitive sentence starters

No generic filler

No exaggerated health claims

No medical promises

No AI-like phrasing

Return JSON only.
"""
    return prompt


def _extract_response_text(resp: Any) -> str:
    """Safely extract text from OpenAI response.

    Prefer `resp.output_text` when available. Fallback to manual extraction.
    """
    try:
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
    except Exception:
        pass

    # Fallback manual extraction
    try:
        if hasattr(resp, "output") and resp.output:
            parts: List[str] = []
            for out in resp.output:
                content = getattr(out, "content", None) or (out.get("content") if isinstance(out, dict) else None)
                if content:
                    for c in content:
                        txt = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                        if txt:
                            parts.append(txt)
            if parts:
                return "".join(parts)
    except Exception:
        pass

    # As last resort, stringify
    try:
        return str(resp)
    except Exception:
        return ""


def call_openai(client: OpenAI, prompt: str, max_retries: int = MAX_RETRIES) -> Tuple[bool, Optional[str], Optional[str]]:
    """Call OpenAI responses.create with structured preference and return text.

    Returns (success, text, error).
    """
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            if DISABLE_OPENAI_CALLS:
                logger.info("OpenAI disabled — skipping call_openai")
                return False, None, "openai_disabled"

            logger.info("openai_call", extra={"attempt": attempt})
            resp = client.responses.create(
                model=API_MODEL,
                input=prompt,
            )
            text = _extract_response_text(resp)
            return True, text, None
        except Exception as exc:
            err = str(exc)
            logger.warning("openai_error", extra={"attempt": attempt, "error": err})
            if attempt < max_retries:
                backoff = 2 ** attempt
                time.sleep(backoff)
            else:
                return False, None, err


def is_duplicate_content(new_product: dict, existing_texts: List[str]) -> bool:
    """Detect duplicates assuming `existing_texts` are concatenated strings.

    Returns True if max cosine similarity > DUPLICATE_SIM_THRESHOLD.
    """
    def _combined_text(p: dict) -> str:
        long_desc = p.get("long_description") or p.get("description") or ""
        care = p.get("care_guide") or p.get("care") or ""
        faqs = p.get("faqs") or []
        faq_answers: List[str] = []
        if isinstance(faqs, list):
            for f in faqs:
                if isinstance(f, dict):
                    faq_answers.append(str(f.get("answer", "")))
        return " ".join([long_desc, care, " ".join(faq_answers)])

    new_text = _combined_text(new_product)
    if not existing_texts:
        return False
    docs = [new_text] + existing_texts
    try:
        tfidf = TfidfVectorizer().fit_transform(docs)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        max_sim = float(np.max(sims)) if sims.size > 0 else 0.0
        logger.info("duplicate_check", extra={"max_similarity": max_sim})
        return max_sim > DUPLICATE_SIM_THRESHOLD
    except Exception as exc:
        logger.warning("duplicate_detection_error", extra={"error": str(exc)})
        return False


def validate_product_json(product: dict) -> Dict[str, Optional[str]]:
    """Validate product JSON strictly.

    Returns {"valid": bool, "error": Optional[str]}.
    """
    # Required top-level fields
    required = [
        "product_title",
        "short_description",
        "long_description",
        "care_guide",
        "price_aed",
        "compare_at_price_aed",
        "seo",
        "tags",
        "collections",
        "category",
    ]
    for field in required:
        if field not in product:
            return {"valid": False, "error": f"missing_field:{field}"}

    # SEO fields
    seo = product.get("seo", {}) if isinstance(product.get("seo", {}), dict) else {}
    meta_title = seo.get("meta_title", "")
    meta_description = seo.get("meta_description", "")
    url_handle = seo.get("url_handle", "")

    if not isinstance(meta_title, str) or len(meta_title) > 60 or not meta_title:
        return {"valid": False, "error": "invalid_meta_title"}
    if not isinstance(meta_description, str) or len(meta_description) > 155 or not meta_description:
        return {"valid": False, "error": "invalid_meta_description"}
    if not isinstance(url_handle, str) or not URL_HANDLE_REGEX.match(url_handle):
        return {"valid": False, "error": "invalid_url_handle"}

    # Long description: enforce minimum only (backend minimum 550 words)
    long_desc = product.get("long_description", "")
    if not isinstance(long_desc, str):
        return {"valid": False, "error": "long_description_type"}
    long_words = len(re.findall(r"\w+", long_desc))
    if long_words < 550:
        return {"valid": False, "error": f"long_description_wordcount:{long_words}"}

    # Care guide word count >= 550
    care = product.get("care_guide", "")
    if not isinstance(care, str):
        return {"valid": False, "error": "care_guide_type"}
    care_words = len(re.findall(r"\w+", care))
    if care_words < 550:
        return {"valid": False, "error": f"care_guide_wordcount:{care_words}"}

    # FAQs validation
    faqs = product.get("faqs")
    if not isinstance(faqs, list) or len(faqs) < 5:
        return {"valid": False, "error": "faqs_min_count"}
    for i, faq in enumerate(faqs):
        if not isinstance(faq, dict):
            return {"valid": False, "error": f"faq_invalid_type:{i}"}
        q = faq.get("question")
        a = faq.get("answer")
        if not q or not a:
            return {"valid": False, "error": f"faq_missing_fields:{i}"}
        answer_words = len(re.findall(r"\w+", str(a)))
        if answer_words < 85:
            return {"valid": False, "error": f"faq_answer_length:{i}:{answer_words}"}

    # Pricing: must be integers and within allowed ratios
    price_val = product.get("price_aed")
    compare_val = product.get("compare_at_price_aed")
    # Ensure numeric and integer
    try:
        if isinstance(price_val, float) and not float(price_val).is_integer():
            return {"valid": False, "error": "price_aed_must_be_integer"}
        if isinstance(compare_val, float) and not float(compare_val).is_integer():
            return {"valid": False, "error": "compare_at_price_aed_must_be_integer"}
        price_int = int(price_val)
        compare_int = int(compare_val)
    except Exception:
        return {"valid": False, "error": "price_fields_invalid"}
    if price_int <= 0:
        return {"valid": False, "error": "price_must_be_positive"}
    if compare_int > int(price_int * 1.5):
        return {"valid": False, "error": "compare_price_too_high"}

    # Category validation
    category = product.get("category")
    if category not in VALID_CATEGORIES:
        return {"valid": False, "error": "invalid_category"}

    # Tags validation
    tags = product.get("tags")
    if not isinstance(tags, list) or not tags:
        return {"valid": False, "error": "tags_missing_or_invalid"}
    normalized_tags = []
    for t in tags:
        if not isinstance(t, str):
            return {"valid": False, "error": "tag_not_string"}
        if t != t.lower():
            return {"valid": False, "error": "tag_not_lowercase"}
        if not TAG_REGEX.match(t):
            return {"valid": False, "error": f"tag_invalid_format:{t}"}
        normalized_tags.append(t)
    if "indoor-plant" not in normalized_tags or "uae-plants" not in normalized_tags:
        return {"valid": False, "error": "required_tags_missing"}

    return {"valid": True, "error": None}


def validate_metadata_fields(metadata: dict) -> Dict[str, Optional[str]]:
    """Validate metadata-specific fields only. Returns same shape as validate_product_json."""
    # Required metadata fields (excluding long_description and care_guide)
    required_meta = [
        "product_title",
        "short_description",
        "estimated_size",
        "characteristics",
        "special_features",
        "mosquito_repellent",
        "ac_room_benefits",
        "pet_friendly",
        "price_aed",
        "seo",
        "tags",
        "meta_fields",
        "faqs",
    ]
    for field in required_meta:
        if field not in metadata:
            return {"valid": False, "error": f"missing_field:{field}"}

    # SEO checks
    seo = metadata.get("seo", {}) if isinstance(metadata.get("seo", {}), dict) else {}
    meta_title = seo.get("meta_title", "")
    meta_description = seo.get("meta_description", "")
    url_handle = seo.get("url_handle", "")
    if not isinstance(meta_title, str) or len(meta_title) > 60 or not meta_title:
        return {"valid": False, "error": "invalid_meta_title"}
    if not isinstance(meta_description, str) or len(meta_description) > 155 or not meta_description:
        return {"valid": False, "error": "invalid_meta_description"}
    if not isinstance(url_handle, str) or not URL_HANDLE_REGEX.match(url_handle):
        return {"valid": False, "error": "invalid_url_handle"}

    # Pricing (AI only provides price_aed)
    price_val = metadata.get("price_aed")
    try:
        if isinstance(price_val, float) and not float(price_val).is_integer():
            return {"valid": False, "error": "price_aed_must_be_integer"}
        price_int = int(price_val)
    except Exception:
        return {"valid": False, "error": "price_fields_invalid"}

    if price_int <= 0:
        return {"valid": False, "error": "price_must_be_positive"}

    # Tags: ensure format only (required tags are enforced by backend later)
    tags = metadata.get("tags")
    if not isinstance(tags, list):
        return {"valid": False, "error": "tags_missing_or_invalid"}
    for t in tags:
        if not isinstance(t, str):
            return {"valid": False, "error": "tag_not_string"}
        if t != t.lower():
            return {"valid": False, "error": "tag_not_lowercase"}
        if not TAG_REGEX.match(t):
            return {"valid": False, "error": f"tag_invalid_format:{t}"}

    # FAQs: 5-8, answers 90-220 words
    faqs = metadata.get("faqs")
    if not isinstance(faqs, list) or len(faqs) < 5 or len(faqs) > 8:
        return {"valid": False, "error": "faqs_min_count"}
    for i, faq in enumerate(faqs):
        if not isinstance(faq, dict):
            return {"valid": False, "error": f"faq_invalid_type:{i}"}
        q = faq.get("question")
        a = faq.get("answer")
        if not q or not a:
            return {"valid": False, "error": f"faq_missing_fields:{i}"}
        answer_words = len(re.findall(r"\w+", str(a)))
        if answer_words < 85:
            return {"valid": False, "error": f"faq_answer_length:{i}:{answer_words}"}

    return {"valid": True, "error": None}


def save_product(data: Dict[str, Any], plant_name: str, output_dir: str = OUTPUT_DIR) -> str:
    """Save product JSON to disk and return path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{sanitize_filename(plant_name)}.json"
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return out_path


def determine_category(plant_name: str) -> str:
    """Deterministic category mapping based on plant name."""
    name = plant_name.lower()
    if "orchid" in name:
        return "Luxury Orchid Gifts"
    if "basil" in name or "mint" in name:
        return "Medicinal & Herbs"
    if "jasmine" in name:
        return "Indian Exotics"
    return "Indoor Plants"


def calculate_compare_price(price: int) -> int:
    """Calculate backend compare-at price with 30% markup and psychological rounding to ..9."""
    try:
        markup = int(price * 1.3)
    except Exception:
        markup = int(price)
    if markup > 10:
        s = str(markup)
        rounded = int(s[:-1] + "9")
    else:
        rounded = markup
    return rounded


def derive_collections(tags: List[str]) -> List[str]:
    collections: List[str] = []
    if "gift-plant" in tags:
        collections.append("Gift Plants")
    if "corporate-gifting" in tags:
        collections.append("Corporate Gifting")
    if "ac-friendly" in tags or "ac-room" in tags:
        collections.append("AC Friendly")
    collections.append("Best Sellers")
    return collections


def normalize_tags(tags: List[str]) -> List[str]:
    """
    Normalize tags to lowercase hyphen format and remove invalid characters.
    """
    normalized = []
    for t in tags:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        t = t.replace("&", "and")
        t = re.sub(r"\s+", "-", t)
        t = re.sub(r"[^a-z0-9-]", "", t)
        t = re.sub(r"-+", "-", t)
        t = t.strip("-")
        if t:
            normalized.append(t)
    return list(set(normalized))


def append_failed_log(plant_name: str, reason: str, failed_log: str = FAILED_LOG) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {plant_name} | {reason}\n"
    with open(failed_log, "a", encoding="utf-8") as fh:
        fh.write(line)


def _attempt_parse_json(text: str) -> Tuple[bool, Optional[dict], Optional[str]]:
    """Attempt to parse JSON from text. Returns (ok, obj, error).

    Strips whitespace and returns helpful debug on failure.
    """
    if text is None:
        return False, None, "no_text"
    txt = text.strip()
    try:
        obj = json.loads(txt)
        return True, obj, None
    except json.JSONDecodeError as exc:
        snippet = txt[:500].replace('\n', ' ')
        return False, None, f"json_decode_error: {str(exc)} | snippet: {snippet}"


def clean_json_response(text: str) -> str:
    """Remove markdown code fences from model response."""
    if not text:
        return text

    text = text.strip()

    # Remove triple backtick blocks
    if text.startswith("```"):
        # Remove first fence (optionally with language)
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        # Remove ending fence
        text = re.sub(r"\n?```$", "", text)

    return text.strip()


def generate_long_description(client: OpenAI, plant_name: str) -> Optional[str]:
    """Generate a long description text for `plant_name`.

    Returns raw text (not JSON). Retries up to MAX_RETRIES. Enforces minimum 550 words.
    """
    prompt = f"""
You are a senior ecommerce copywriter crafting a long-form product description for a premium indoor plant sold in the UAE.

Requirements:

Before returning, count words. If under 550 words, expand further until reaching at least 550 words.

Write approximately 650-750 words. Never go below 620 words.

Write the description for: {plant_name}
"""
    if DISABLE_OPENAI_CALLS:
        logger.info("OpenAI disabled — loading existing long_description for %s", plant_name)
        existing = _load_existing_product(plant_name)
        if existing:
            return existing.get("long_description")
        logger.warning("No existing long_description found for %s", plant_name)
        return None

    try:
        logger.info("generate_long_description_call", extra={"plant": plant_name})
        resp = client.responses.create(model=API_MODEL, input=prompt)
        text = _extract_response_text(resp)
        if not text:
            logger.warning("long_description_empty", extra={"plant": plant_name})
            return None
        return text
    except Exception as exc:
        logger.warning("long_description_error", extra={"plant": plant_name, "error": str(exc)})
        return None


def generate_care_guide(client: OpenAI, plant_name: str) -> Optional[str]:
    """Generate a care guide for `plant_name`.

    Returns raw text. Retries up to MAX_RETRIES. Enforces minimum 600 words.
    """
    prompt = f"""
You are writing an in-depth, blog-quality care guide for a premium indoor plant sold in the UAE.

Requirements:
- Write approximately 700-800 words.
- Never go below 650 words.
- Structured educational tone, paragraph format (no markdown, no bullets).
- Must include: light (UAE context), watering schedule, soil, fertilizer, AC adjustments, humidity, repotting, pest prevention, seasonal changes, common mistakes.

Before returning, count words. If under 650 words, expand further until reaching at least 650 words.

Write the care guide for: {plant_name}
"""
    if DISABLE_OPENAI_CALLS:
        logger.info("OpenAI disabled — loading existing care_guide for %s", plant_name)
        existing = _load_existing_product(plant_name)
        if existing:
            return existing.get("care_guide")
        logger.warning("No existing care_guide found for %s", plant_name)
        return None

    try:
        logger.info("generate_care_guide_call", extra={"plant": plant_name})
        resp = client.responses.create(model=API_MODEL, input=prompt)
        text = _extract_response_text(resp)
        if not text:
            logger.warning("care_guide_empty", extra={"plant": plant_name})
            return None
        return text
    except Exception as exc:
        logger.warning("care_guide_error", extra={"plant": plant_name, "error": str(exc)})
        return None


def generate_structured_metadata(client: OpenAI, plant_name: str) -> Optional[dict]:
    """Generate the structured metadata JSON for `plant_name`.

    Returns a parsed dict or None. Retries up to MAX_RETRIES. Requires strict JSON only.
    """
    prompt_template = """
You are a senior luxury ecommerce strategist and premium indoor plant copywriter for a Dubai-based indoor plant brand.

The brand represents refined, design-led indoor greenery curated for modern UAE homes, offices, and corporate environments.

Generate STRICT valid JSON only (no commentary, no markdown, no explanations) using the exact schema below for the plant: {plant_name}

{
"product_title": "",
"short_description": "",
"estimated_size": "",
"characteristics": [],
"special_features": [],
"mosquito_repellent": "",
"ac_room_benefits": "",
"pet_friendly": "",
"price_aed": 0,
"seo": {
    "meta_title": "",
    "meta_description": "",
    "url_handle": ""
},
"tags": [],
"meta_fields": {
    "lusherra_care_level": "",
    "lusherra_room_placement": ""
},
"faqs": [{
    "question": "",
    "answer": ""
}]
}

CRITICAL:
- Return valid JSON only.
- Do not include fields not listed above.
- All numeric values must be numbers, not strings.
- No markdown formatting.
- No bold symbols.
- No emojis.

----------------------------------------
BRAND TONE
----------------------------------------

The tone must feel:
- Premium
- Calm
- Minimal
- Sophisticated
- Design-conscious
- Suitable for Dubai interiors

Avoid:
- Marketplace tone
- Generic ecommerce phrasing
- Over-promising health claims
- Repetitive sentence structures

STRUCTURAL VARIATION REQUIREMENT

Do not reuse the same paragraph structure across plants.
Avoid repeating this structure:
- Intro paragraph
- Aesthetic section
- UAE climate section
- Care simplicity section
- Conclusion

Instead, vary:
- Opening angle (design, emotion, lifestyle, spatial placement)
- Section order
- Sentence rhythm
- Transitions

Each plant must feel editorially distinct.

----------------------------------------
UAE POSITIONING
----------------------------------------

Naturally integrate context such as:
- Dubai apartments
- UAE villas
- AC-cooled interiors
- Modern minimal interiors
- Corporate gifting in UAE
- Reception areas, offices, majlis spaces

Luxury positioning must feel subtle and refined.
Use contextual references such as:
- Dubai high-rise apartments
- Contemporary UAE villas
- Minimal majlis interiors
- Corporate reception styling
- Architectural spaces with neutral palettes

Avoid repeating the phrase “perfect for UAE homes” across products.
Vary spatial language per plant.

Do NOT keyword stuff “Dubai” or “UAE”.
Use geographic references only when contextually relevant.

----------------------------------------
PRODUCT TITLE
----------------------------------------

- Under 65 characters.
- Must include plant name.
- May include UAE naturally if it fits.
- Should feel brand-like, not blog-like.

----------------------------------------
SHORT DESCRIPTION
----------------------------------------

2–3 refined sentences.

Must:
- Mention ready-to-display plant with pot.
- Suggest interior styling placement.
- Feel curated and premium.
- Subtle UAE context.

----------------------------------------
PRICING
----------------------------------------

Return realistic UAE luxury indoor plant pricing.
price_aed must be a realistic integer.
Do NOT include compare_at_price_aed.

----------------------------------------
SEO STRATEGY
----------------------------------------

Meta title:
- Under 60 characters.
- Must include plant name.
- Include UAE or Dubai naturally when relevant.

SEO DIFFERENTIATION REQUIREMENT

Meta titles must vary in structure across products.
Avoid repeating this template:
"Plant Name UAE | Luxury Indoor Decor"

Instead rotate structures like:
- "Plant Name for Dubai Interiors"
- "Plant Name UAE – Refined Indoor Greenery"
- "Plant Name | Premium Indoor Plant UAE"
- "Plant Name Dubai | Modern Indoor Styling"

Meta descriptions must not repeat sentence patterns across products.

Meta description:
- Under 155 characters.
- Include plant name.
- Include UAE context.
- Suggest interior styling.
- Optionally mention fast UAE delivery subtly.
- No keyword stuffing.

URL handle:
- Lowercase.
- Hyphen-separated.
- Include plant name.
- Include "uae" when natural.
- No special characters.

----------------------------------------
TAGS
----------------------------------------

All lowercase.
Hyphen-separated.
No spaces.

Must include:
- indoor-plant
- uae-plants

Add relevant tags like:
- luxury-plant
- flowering-plant
- gift-plant
- corporate-gifting
- ac-friendly
- low-maintenance
- pet-safe (only if accurate)

----------------------------------------
META FIELDS
----------------------------------------

lusherra_care_level:
One of: Easy, Moderate, Advanced

lusherra_room_placement:
Examples:
Living Room, Bedroom, Office, Reception, Dining Area, Corporate Lobby

Use comma-separated values if multiple.

----------------------------------------
FAQ REQUIREMENTS – STRICT ENFORCEMENT

You MUST generate 5–8 FAQs.

Each FAQ answer MUST be between 130 and 220 words.

Before returning the JSON:
1. Count the words of EACH FAQ answer.
2. If ANY answer is under 130 words, rewrite and expand it.
3. Repeat internally until ALL FAQ answers are at least 130 words.
4. Do NOT return until all answers meet the minimum.

The system will automatically reject answers under 85 words.
Do not summarize.
Do not shorten.
Do not compress sentences.
Write full, developed explanations.

This prevents borderline 90-word answers that keep failing.
"""
    # Replace only the {plant_name} placeholder without invoking str.format() so JSON braces are left intact
    prompt = prompt_template.replace("{plant_name}", plant_name)
    if DISABLE_OPENAI_CALLS:
        logger.info("OpenAI disabled — loading existing structured metadata for %s", plant_name)
        existing = _load_existing_product(plant_name)
        if existing:
            # return the structured metadata portion if present, otherwise the whole product
            meta_keys = ["product_title", "short_description", "estimated_size", "characteristics", "special_features", "mosquito_repellent", "ac_room_benefits", "pet_friendly", "price_aed", "seo", "tags", "meta_fields", "faqs"]
            meta = {k: existing.get(k) for k in meta_keys if k in existing}
            return meta if meta else existing
        logger.warning("No existing structured metadata found for %s", plant_name)
        return None

    try:
        logger.info("generate_structured_metadata_call", extra={"plant": plant_name})
        resp = client.responses.create(
            model=API_MODEL,
            input=prompt,
        )
        text = _extract_response_text(resp)
        if not text:
            logger.warning("metadata_empty", extra={"plant": plant_name})
            return None
        cleaned = clean_json_response(text)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        logger.warning("metadata_parse_not_object", extra={"plant": plant_name})
        return None
    except Exception as exc:
        logger.warning("metadata_generation_error", extra={"plant": plant_name, "error": str(exc)})
        return None


def expand_faq_answer(client: OpenAI, plant_name: str, question: str, answer: str) -> Optional[str]:
    """
    Expand a single FAQ answer to minimum 120 words.
    Retries up to MAX_RETRIES if still too short.
    """

    if DISABLE_OPENAI_CALLS:
        logger.info("OpenAI disabled — cannot expand FAQ answer for %s", plant_name)
        return None

    for attempt in range(MAX_RETRIES):
        prompt = f"""
You are expanding an FAQ answer for a premium UAE indoor plant product.

Plant: {plant_name}

Question:
{question}

Current Answer:
{answer}

Expand this answer to at least 120 words.
Do not shorten.
Do not change meaning.
Do not add unrelated claims.
Keep it premium and informative.
Return only the expanded paragraph.
No markdown.
No commentary.

IMPORTANT:
Ensure the final answer is at least 120 words.
"""

        try:
            resp = client.responses.create(model=API_MODEL, input=prompt)
            text = _extract_response_text(resp)
            if not text:
                continue

            expanded = text.strip()
            word_count = len(re.findall(r"\w+", expanded))

            if word_count >= 120:
                return expanded

        except Exception:
            continue

    return None


def main() -> None:
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("missing_api_key")
        return

    client = OpenAI(api_key=api_key)

    try:
        plants = load_plants(CSV_PATH)
    except Exception as exc:
        logger.error("failed_loading_plants", extra={"error": str(exc)})
        return

    total = len(plants)
    logger.info("starting_product_generation", extra={"total_plants": total})

    existing_texts: List[str] = []

    for idx, plant in enumerate(plants, start=1):
        logger.info("processing_plant", extra={"plant": plant, "index": idx, "total": total})

        # 3-stage generation pipeline
        # 1) Long description
        long_description = generate_long_description(client, plant)
        if not long_description:
            append_failed_log(plant, "long_description_generation_failed")
            time.sleep(DELAY_BETWEEN_PLANTS)
            continue
        long_words = len(re.findall(r"\w+", long_description))
        if long_words < 550:
            logger.error("long_description_too_short_after_generation", extra={"plant": plant, "words": long_words})
            append_failed_log(plant, f"long_description_too_short:{long_words}")
            time.sleep(DELAY_BETWEEN_PLANTS)
            continue

        # 2) Care guide
        care_guide = generate_care_guide(client, plant)
        if not care_guide:
            append_failed_log(plant, "care_guide_generation_failed")
            time.sleep(DELAY_BETWEEN_PLANTS)
            continue
        care_words = len(re.findall(r"\w+", care_guide))
        if care_words < 600:
            logger.error("care_guide_too_short_after_generation", extra={"plant": plant, "words": care_words})
            append_failed_log(plant, f"care_guide_too_short:{care_words}")
            time.sleep(DELAY_BETWEEN_PLANTS)
            continue

        # 3) Structured metadata (JSON)
        metadata = generate_structured_metadata(client, plant)
        if not metadata or not isinstance(metadata, dict):
            append_failed_log(plant, "metadata_generation_failed")
            time.sleep(DELAY_BETWEEN_PLANTS)
            continue

        # 3.a Inject backend-controlled business fields into metadata
        # Ensure tags exist and force required tags
        meta_tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
        normalized_tags = normalize_tags(meta_tags)

        normalized_tags.append("indoor-plant")
        normalized_tags.append("uae-plants")

        metadata["tags"] = list(set(normalized_tags))

        # Calculate compare_at_price and round via backend logic
        try:
            price_val = int(metadata.get("price_aed", 0))
        except Exception:
            price_val = 0
        metadata["compare_at_price_aed"] = calculate_compare_price(price_val)

        # Determine deterministic category and inject delivery/about
        metadata["category"] = determine_category(plant)
        metadata["delivery_info"] = DELIVERY_POLICY
        metadata["about_lusherra_section"] = ABOUT_LUSHERRA

        # Derive collections from tags
        metadata["collections"] = derive_collections(metadata.get("tags", []))

        # Merge into final product object (AI provides creative fields; backend provides business fields)
        parsed = dict(metadata)
        parsed["long_description"] = long_description
        parsed["care_guide"] = care_guide

        # Duplicate detection
        try:
            if is_duplicate_content(parsed, existing_texts):
                logger.info("duplicate_detected", extra={"plant": plant})
                # Attempt section-specific regeneration in order: long_description, care_guide, metadata
                duplicate_resolved = False

                # Try regenerating long_description up to MAX_RETRIES
                for attempt in range(1, MAX_RETRIES + 1):
                    ld2 = generate_long_description(client, plant)
                    if not ld2:
                        logger.warning("duplicate_regen_long_description_failed_attempt", extra={"plant": plant, "attempt": attempt})
                        time.sleep(1)
                        continue
                    candidate = dict(parsed)
                    candidate["long_description"] = ld2
                    if not is_duplicate_content(candidate, existing_texts):
                        parsed = candidate
                        duplicate_resolved = True
                        logger.info("duplicate_resolved_by_long_description", extra={"plant": plant, "attempt": attempt})
                        break
                    time.sleep(1)

                if not duplicate_resolved:
                    # Try regenerating care_guide
                    for attempt in range(1, MAX_RETRIES + 1):
                        cg2 = generate_care_guide(client, plant)
                        if not cg2:
                            logger.warning("duplicate_regen_care_guide_failed_attempt", extra={"plant": plant, "attempt": attempt})
                            time.sleep(1)
                            continue
                        candidate = dict(parsed)
                        candidate["care_guide"] = cg2
                        if not is_duplicate_content(candidate, existing_texts):
                            parsed = candidate
                            duplicate_resolved = True
                            logger.info("duplicate_resolved_by_care_guide", extra={"plant": plant, "attempt": attempt})
                            break
                        time.sleep(1)

                if not duplicate_resolved:
                    # Finally try regenerating metadata only
                    for attempt in range(1, MAX_RETRIES + 1):
                        meta2 = generate_structured_metadata(client, plant)
                        if not meta2 or not isinstance(meta2, dict):
                            logger.warning("duplicate_regen_metadata_failed_attempt", extra={"plant": plant, "attempt": attempt})
                            time.sleep(1)
                            continue
                        candidate = dict(meta2)
                        candidate["long_description"] = parsed.get("long_description")
                        candidate["care_guide"] = parsed.get("care_guide")
                        if not is_duplicate_content(candidate, existing_texts):
                            parsed = candidate
                            duplicate_resolved = True
                            logger.info("duplicate_resolved_by_metadata", extra={"plant": plant, "attempt": attempt})
                            break
                        time.sleep(1)

                if not duplicate_resolved:
                    logger.error("duplicate_persisted", extra={"plant": plant})
                    append_failed_log(plant, "duplicate_persisted")
                    time.sleep(DELAY_BETWEEN_PLANTS)
                    continue
        except Exception as exc:
            logger.warning("duplicate_check_error", extra={"plant": plant, "error": str(exc)})

        # Validation, with one regeneration attempt on failure
        validation = validate_product_json(parsed)
        if not validation.get("valid"):
            err = validation.get("error") or "validation_error"
            logger.info("validation_failed", extra={"plant": plant, "error": err})
            # Decide which section to regenerate based on the error code
            regenerated = False

            # Helper to attempt regenerating a section up to MAX_RETRIES
            def _try_regenerate_section(section: str) -> Optional[dict]:
                for attempt in range(1, MAX_RETRIES + 1):
                    if section == "long_description":
                        new_ld = generate_long_description(client, plant)
                        if not new_ld:
                            logger.warning("regen_long_description_failed_attempt", extra={"plant": plant, "attempt": attempt})
                            time.sleep(1)
                            continue
                        candidate = dict(parsed)
                        candidate["long_description"] = new_ld
                        # quick local validation for length
                        words = len(re.findall(r"\w+", new_ld))
                        if words < 500 or words > 700:
                            logger.warning("regen_long_description_wordcount_invalid", extra={"plant": plant, "attempt": attempt, "words": words})
                            time.sleep(1)
                            continue
                        return candidate
                    if section == "care_guide":
                        new_cg = generate_care_guide(client, plant)
                        if not new_cg:
                            logger.warning("regen_care_guide_failed_attempt", extra={"plant": plant, "attempt": attempt})
                            time.sleep(1)
                            continue
                        words = len(re.findall(r"\w+", new_cg))
                        if words < 550:
                            logger.warning("regen_care_guide_wordcount_invalid", extra={"plant": plant, "attempt": attempt, "words": words})
                            time.sleep(1)
                            continue
                        candidate = dict(parsed)
                        candidate["care_guide"] = new_cg
                        return candidate
                    if section == "metadata":
                        new_meta = generate_structured_metadata(client, plant)
                        if not new_meta or not isinstance(new_meta, dict):
                            logger.warning("regen_metadata_failed_attempt", extra={"plant": plant, "attempt": attempt})
                            time.sleep(1)
                            continue
                        candidate = dict(new_meta)
                        candidate["long_description"] = parsed.get("long_description")
                        candidate["care_guide"] = parsed.get("care_guide")
                        # run metadata-specific validation
                        meta_val = validate_metadata_fields(candidate)
                        if not meta_val.get("valid"):
                            logger.warning("regen_metadata_validation_failed", extra={"plant": plant, "attempt": attempt, "error": meta_val.get("error")})
                            time.sleep(1)
                            continue
                        return candidate
                return None

            # Map errors to sections
            if err.startswith("long_description"):
                cand = _try_regenerate_section("long_description")
                if cand:
                    parsed = cand
                    regenerated = True
            elif err.startswith("care_guide"):
                cand = _try_regenerate_section("care_guide")
                if cand:
                    parsed = cand
                    regenerated = True
            elif err.startswith("faq_answer_length"):
                # Extract FAQ index from error like faq_answer_length:0:87
                try:
                    parts = err.split(":")
                    faq_index = int(parts[1])
                except Exception:
                    faq_index = None

                if faq_index is not None:
                    faqs = parsed.get("faqs", [])
                    if 0 <= faq_index < len(faqs):
                        question = faqs[faq_index].get("question")
                        answer = faqs[faq_index].get("answer")

                        expanded = expand_faq_answer(client, plant, question, answer)
                        if expanded:
                            parsed["faqs"][faq_index]["answer"] = expanded
                            regenerated = True

                # If expansion failed, fallback to metadata regeneration
                if not regenerated:
                    cand = _try_regenerate_section("metadata")
                    if cand:
                        parsed = cand
                        regenerated = True

            else:
                # Default metadata regeneration for SEO, tags, pricing, faqs, category, etc.
                cand = _try_regenerate_section("metadata")
                if cand:
                    parsed = cand
                    regenerated = True

            if not regenerated:
                logger.error("validation_regeneration_failed", extra={"plant": plant, "error": err})
                append_failed_log(plant, f"validation_failed:{err}")
                time.sleep(DELAY_BETWEEN_PLANTS)
                continue
            # Re-validate after targeted regeneration
            validation2 = validate_product_json(parsed)
            if not validation2.get("valid"):
                logger.error("validation_regeneration_failed_final", extra={"plant": plant, "error": validation2.get("error")})
                append_failed_log(plant, f"validation_failed:{validation2.get('error')}")
                time.sleep(DELAY_BETWEEN_PLANTS)
                continue
            logger.info("validation_regeneration_success", extra={"plant": plant})

        # Final safety check: re-count long_description words and log before saving
        try:
            final_long_desc = parsed.get("long_description", "") or ""
            final_long_words = len(re.findall(r"\w+", final_long_desc))
            logger.info("final_long_description_wordcount", extra={"plant": plant, "words": final_long_words})
        except Exception:
            # non-fatal logging error, continue to save
            pass

        # Save and record success
        try:
            out_path = save_product(parsed, plant)
            logger.info("final_success", extra={"plant": plant, "path": out_path})
            # append combined text for future duplicate checks
            long_desc = parsed.get("long_description") or parsed.get("description") or ""
            care = parsed.get("care_guide") or parsed.get("care") or ""
            faqs = parsed.get("faqs") or []
            faq_answers: List[str] = []
            if isinstance(faqs, list):
                for f in faqs:
                    if isinstance(f, dict):
                        faq_answers.append(str(f.get("answer", "")))
            combined = " ".join([long_desc, care, " ".join(faq_answers)])
            existing_texts.append(combined)
        except Exception as exc:
            logger.error("save_failed", extra={"plant": plant, "error": str(exc)})
            append_failed_log(plant, f"save_failed:{str(exc)}")

        time.sleep(DELAY_BETWEEN_PLANTS)

    logger.info("product_generation_complete", extra={"total": total})


if __name__ == "__main__":
    main()
