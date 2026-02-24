#!/usr/bin/env python3
"""
generate_images.py

Main script to bulk-generate product images for an ecommerce plant store.

Features:
- Reads `plants.csv` (column: plant_name)
- Rotates between 3 templates: studio, lifestyle, macro
- Uses OpenAI Images API (`gpt-image-1`) via modern OpenAI Python SDK
- Produces 2400x2400 PNG images
- Batches requests (10 per batch) with pauses and per-call delays
- Retries failed API calls up to 3 times and logs failures to `failed_log.txt`
- Resume-safe: skips existing output files
- Reads API key from `.env` via `python-dotenv`

Usage: python generate_images.py
"""

import os
import csv
import time
import re
import base64
from PIL import Image
import io
import logging
import unicodedata
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from config.runtime_flags import DISABLE_OPENAI_CALLS

from templates import TEMPLATES

# -----------------------------
# Configuration
# -----------------------------
PLANTS_CSV = "plants.csv"
OUTPUT_DIR = "output"
FAILED_LOG = "failed_log.txt"
ENV_FILE = ".env"
IMAGE_SIZE = "2400x2400"
MODEL_NAME = "gpt-image-1"
BATCH_SIZE = 10
PER_CALL_DELAY = 2  # seconds between calls
POST_BATCH_DELAY = 10  # seconds pause after each batch
MAX_RETRIES = 3

# Rotation order for templates
TEMPLATE_ORDER = ["studio", "lifestyle", "macro", "luxury_livingroom",  "macro_hydrated",]

# Configure logging for console-friendly output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_plants(csv_path: str = PLANTS_CSV) -> List[str]:
    """Load plant names from a CSV file with header `plant_name`.

    Returns a list of plant name strings in the order found.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    plants = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "plant_name" not in reader.fieldnames:
            raise ValueError("CSV file must contain a 'plant_name' column")
        for row in reader:
            name = (row.get("plant_name") or "").strip()
            if name:
                plants.append(name)
    return plants


def sanitize_filename(name: str) -> str:
    """Sanitize a plant name to the required filename format.

    Rules applied:
    - Normalize unicode and strip accents
    - Uppercase
    - Replace spaces with hyphens
    - Remove slashes, apostrophes, and other special characters
    - Only allow A-Z, 0-9, and hyphens
    """
    # Normalize and remove accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_bytes = nfkd.encode("ascii", "ignore")
    ascii_str = ascii_bytes.decode("ascii")

    # Uppercase
    s = ascii_str.upper()

    # Replace whitespace with single hyphen
    s = re.sub(r"\s+", "-", s)

    # Remove slashes and apostrophes explicitly
    s = s.replace("/", "").replace("\\", "").replace("'", "")

    # Remove any character not A-Z, 0-9, or hyphen
    s = re.sub(r"[^A-Z0-9-]", "", s)

    # Collapse multiple hyphens
    s = re.sub(r"-+", "-", s)

    # Trim leading/trailing hyphens
    s = s.strip("-")

    return s


def build_prompt(template_key: str, plant_name: str) -> str:
    """Build the prompt text for a given template and plant name.

    Templates live in `templates.TEMPLATES` and are formatted with `{plant_name}`.
    """
    template = TEMPLATES.get(template_key)
    if not template:
        raise ValueError(f"Unknown template key: {template_key}")
    return template.format(plant_name=plant_name)


def generate_image(client: OpenAI, prompt: str, output_path: str, max_retries: int = MAX_RETRIES) -> bool:
    """Generate an image via the OpenAI Images API and save to `output_path` at 2400x2400.

    Behavior:
    - Request image at 1024x1024 from the API (supported size)
    - Decode base64, then upscale locally to 2400x2400 using Pillow LANCZOS
    - Retry on failure up to `max_retries`

    Returns True on success, False on permanent failure.
    """
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            if DISABLE_OPENAI_CALLS:
                logger.info("OpenAI disabled — skipping generate_image for prompt: %s", prompt[:80])
                return False
            logger.info("Requesting image (attempt %d)...", attempt)
            # Request at supported size (1024x1024), then upscale locally
            resp = client.images.generate(
                model=MODEL_NAME,
                prompt=prompt,
                size="1024x1024",
            )

            # The modern response contains base64 image data at data[0].b64_json
            img_b64 = resp.data[0].b64_json
            img_bytes = base64.b64decode(img_b64)

            # Load image into Pillow from bytes
            with io.BytesIO(img_bytes) as img_buffer:
                with Image.open(img_buffer) as im:
                    # Convert to RGBA to preserve any alpha, then to RGB for PNG
                    im_converted = im.convert("RGBA")

                    # Upscale to 2400x2400 using high-quality LANCZOS resampling
                    target_size = (2400, 2400)
                    im_upscaled = im_converted.resize(target_size, resample=Image.LANCZOS)

                    # Ensure output is saved in PNG-compatible mode (RGBA is fine)
                    im_upscaled.save(output_path, format="PNG")

            logger.info("Saved upscaled image to %s", output_path)
            return True

        except Exception as exc:
            logger.warning("Image generation failed on attempt %d: %s", attempt, exc)
            if attempt < max_retries:
                backoff = 2 ** attempt
                logger.info("Retrying after %d seconds...", backoff)
                time.sleep(backoff)
            else:
                logger.error("All retries failed for prompt. Error: %s", exc)
                return False


def append_failed_log(plant_name: str, template_key: str, reason: str, failed_log: str = FAILED_LOG) -> None:
    """Append an entry to the failed log file (safe to call from anywhere)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {plant_name} | {template_key} | {reason}\n"
    with open(failed_log, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    """Main execution: load plants, rotate templates, and generate images in batches."""
    # Load environment
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment. Please set it in %s", ENV_FILE)
        return

    # Initialize OpenAI client with provided API key
    client = OpenAI(api_key=api_key)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load plants
    try:
        plants = load_plants(PLANTS_CSV)
    except Exception as exc:
        logger.error("Failed to load plants: %s", exc)
        return

    total = len(plants)
    if total == 0:
        logger.info("No plants found in %s. Exiting.", PLANTS_CSV)
        return

    logger.info("Starting generation for %d plants", total)

    processed = 0
    batch_number = 1

    for idx, plant in enumerate(plants, start=1):
        template_key = TEMPLATE_ORDER[(idx - 1) % len(TEMPLATE_ORDER)]
        sanitized = sanitize_filename(plant)
        if not sanitized:
            logger.warning("Skipping empty/invalid plant name at row %d: %r", idx, plant)
            append_failed_log(plant, template_key, "Invalid sanitized filename")
            continue

        filename = f"{sanitized}-{template_key}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)

        # Resume capability: skip if file exists
        if os.path.exists(out_path):
            logger.info("Skipping existing file (%s) [%d/%d]", filename, idx, total)
            processed += 1
        else:
            prompt = build_prompt(template_key, plant)
            logger.info("Generating (%s) for plant: %s [%d/%d] — Batch %d", template_key, plant, idx, total, batch_number)

            success = generate_image(client, prompt, out_path, max_retries=MAX_RETRIES)
            if not success:
                append_failed_log(plant, template_key, "API generation failed after retries")
            # Respect per-call delay
            time.sleep(PER_CALL_DELAY)

            processed += 1

        # Batch handling
        if processed > 0 and processed % BATCH_SIZE == 0:
            logger.info("Completed batch %d (processed %d images). Pausing %d seconds...", batch_number, processed, POST_BATCH_DELAY)
            batch_number += 1
            time.sleep(POST_BATCH_DELAY)

        # Print estimated progress
        logger.info("Progress: %d/%d (%.1f%%)", idx, total, (idx / total) * 100)

    logger.info("Generation run complete. Processed: %d. See %s for failures.", processed, FAILED_LOG)


if __name__ == "__main__":
    main()
