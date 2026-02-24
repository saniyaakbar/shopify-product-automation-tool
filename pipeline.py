#!/usr/bin/env python3
"""pipeline.py

Orchestrator that runs image generation and product metadata generation
for each plant in `plants.csv`. Reuses existing functions from
`generate_images.py` and `generate_products.py` and creates a single
OpenAI client to avoid duplicated setup.

Usage: python pipeline.py
"""
from __future__ import annotations

import os
import json
import time
import logging
from typing import List, Dict, Any
import base64
import io
from PIL import Image

from dotenv import load_dotenv
from openai import OpenAI
from config.runtime_flags import DISABLE_OPENAI_CALLS

import generate_images as img_mod
import generate_products as prod_mod
from utils.html_formatter import format_description_to_html
try:
    from shopify_uploader import create_product, upload_images, get_installed_shops, ShopifyClient
except Exception:
    create_product = None
    upload_images = None
    get_installed_shops = None
    ShopifyClient = None

# Output layout
BASE_OUTPUT = "outputs"
IMAGES_DIR = os.path.join(BASE_OUTPUT, "images")
PRODUCTS_DIR = os.path.join(BASE_OUTPUT, "products")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")

# When True, existing product JSON files will be uploaded to Shopify again
FORCE_SHOPIFY_UPLOAD = True


def ensure_dirs() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(PRODUCTS_DIR, exist_ok=True)


def plant_slug(name: str) -> str:
    """Create a filesystem-safe slug for a plant.

    Reuse existing `sanitize_filename` but normalize to lowercase for paths.
    """
    s = img_mod.sanitize_filename(name)
    return s.lower() if isinstance(s, str) else name.replace(" ", "-").lower()


def generate_images_for_plant(client: OpenAI, plant: str, out_dir: str, save_to_disk: bool = True) -> List[Dict[str, str]]:
    """Generate images for a plant and return structured image objects.

    Returns a list of dicts with keys:
      - filename: suggested filename
      - b64: base64-encoded PNG bytes

    If `save_to_disk` is True the images are also written to `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)
    images: List[Dict[str, str]] = []

    templates = getattr(img_mod, "TEMPLATE_ORDER", None) or ["studio", "lifestyle", "macro"]

    for template_key in templates:
        filename = f"{plant}-{template_key}.png"
        output_path = os.path.join(out_dir, filename)

        # If file exists and save_to_disk is True, load and return existing bytes
        if save_to_disk and os.path.exists(output_path):
            logger.info("Loading existing image from disk: %s", output_path)
            try:
                with open(output_path, "rb") as fh:
                    b = fh.read()
                b64 = base64.b64encode(b).decode("ascii")
                images.append({"filename": filename, "b64": b64})
                continue
            except Exception:
                logger.exception("Failed to read existing image file, will regenerate: %s", output_path)

        prompt = img_mod.build_prompt(template_key, plant)
        try:
            # Directly call OpenAI Images API to obtain base64 payload (mirrors generate_images.generate_image)
            if DISABLE_OPENAI_CALLS:
                logger.info("OpenAI disabled — skipping image generation for %s (%s)", plant, template_key)
                img_mod.append_failed_log(plant, template_key, "openai_disabled")
                continue

            resp = client.images.generate(model=getattr(img_mod, "MODEL_NAME", "gpt-image-1"), prompt=prompt, size="1024x1024")
            img_b64 = resp.data[0].b64_json
            if not img_b64:
                logger.warning("Empty image response for %s (%s)", plant, template_key)
                img_mod.append_failed_log(plant, template_key, "empty_response")
                continue

            # Upscale locally to 2400x2400 to match prior behavior
            img_bytes = base64.b64decode(img_b64)
            with io.BytesIO(img_bytes) as img_buffer:
                with Image.open(img_buffer) as im:
                    im_converted = im.convert("RGBA")
                    target_size = (2400, 2400)
                    im_upscaled = im_converted.resize(target_size, resample=Image.LANCZOS)
                    with io.BytesIO() as out_buf:
                        im_upscaled.save(out_buf, format="PNG")
                        out_bytes = out_buf.getvalue()

            out_b64 = base64.b64encode(out_bytes).decode("ascii")
            images.append({"filename": filename, "b64": out_b64})

            # Optionally save to disk for resume / local inspection
            if save_to_disk:
                try:
                    with open(output_path, "wb") as fh:
                        fh.write(out_bytes)
                    logger.info("Saved image to %s", output_path)
                except Exception:
                    logger.exception("Failed to save image to disk: %s", output_path)

        except Exception as exc:
            logger.exception("Exception during image generation for %s: %s", plant, exc)
            img_mod.append_failed_log(plant, template_key, f"exception:{exc}")

        # polite delay between image calls
        time.sleep(getattr(img_mod, "PER_CALL_DELAY", 1))

    return images


def generate_product_for_plant(client: OpenAI, plant: str, images: List[Dict[str, str]]) -> Dict[str, Any]:
    """Generate product metadata for a plant and return the structured dict.

    This reuses the three-stage pipeline in `generate_products.py` and then
    injects deterministic backend fields and image references. It does NOT
    alter the original generator functions.
    """
    result: Dict[str, Any] = {}

    # Long description
    long_desc = prod_mod.generate_long_description(client, plant)
    if not long_desc:
        prod_mod.append_failed_log(plant, "long_description_generation_failed")
        raise RuntimeError("long_description_generation_failed")

    # Care guide
    care = prod_mod.generate_care_guide(client, plant)
    if not care:
        prod_mod.append_failed_log(plant, "care_guide_generation_failed")
        raise RuntimeError("care_guide_generation_failed")

    # Structured metadata (JSON)
    metadata = prod_mod.generate_structured_metadata(client, plant)
    if not metadata or not isinstance(metadata, dict):
        prod_mod.append_failed_log(plant, "metadata_generation_failed")
        raise RuntimeError("metadata_generation_failed")

    # Merge creative fields with backend deterministic fields
    metadata["long_description"] = long_desc
    metadata["care_guide"] = care
    # Also provide Shopify-ready HTML version of the long description
    try:
        metadata["body_html"] = format_description_to_html(long_desc)
    except Exception:
        logger.exception("Failed to format long_description to HTML for %s", plant)
        metadata["body_html"] = metadata["long_description"]

    # Ensure tags normalized and required tags present
    meta_tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
    normalized = prod_mod.normalize_tags(meta_tags)
    normalized.extend(["indoor-plant", "uae-plants"])
    metadata["tags"] = list(dict.fromkeys(normalized))

    # Business fields injection
    try:
        price_val = int(metadata.get("price_aed", 0))
    except Exception:
        price_val = 0
    metadata["compare_at_price_aed"] = prod_mod.calculate_compare_price(price_val)
    metadata["category"] = prod_mod.determine_category(plant)
    metadata["delivery_info"] = prod_mod.DELIVERY_POLICY if hasattr(prod_mod, "DELIVERY_POLICY") else ""
    metadata["about_lusherra_section"] = prod_mod.ABOUT_LUSHERRA if hasattr(prod_mod, "ABOUT_LUSHERRA") else ""
    metadata["collections"] = prod_mod.derive_collections(metadata.get("tags", []))

    # Inject image references (structured objects with filename and base64 payload)
    metadata["images"] = images

    # Final validation (reuse existing validator)
    validation = prod_mod.validate_product_json(metadata)
    if not validation.get("valid"):
        err = validation.get("error")
        logger.warning("Product validation failed for %s: %s", plant, err)
        # Do not raise; return metadata with validation error flagged
        metadata["_validation_error"] = err

    return metadata


def save_product_output(metadata: Dict[str, Any], plant_slug_str: str) -> str:
    """Save metadata dict to outputs/products/{plant_slug}.json and return path."""
    os.makedirs(PRODUCTS_DIR, exist_ok=True)
    out_path = os.path.join(PRODUCTS_DIR, f"{plant_slug_str}.json")
    # Avoid embedding large base64 image payloads in the on-disk JSON.
    disk_copy = dict(metadata)
    imgs = disk_copy.get("images")
    if isinstance(imgs, list):
        # Replace structured image objects with filenames for on-disk storage
        try:
            disk_copy["images"] = [i.get("filename") if isinstance(i, dict) else str(i) for i in imgs]
        except Exception:
            disk_copy["images"] = []

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(disk_copy, fh, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment (.env)")
        return

    if DISABLE_OPENAI_CALLS:
        logger.warning("OPENAI CALLS DISABLED — SAFE MODE ENABLED")

    ensure_dirs()

    client = None
    if not DISABLE_OPENAI_CALLS:
        client = OpenAI(api_key=api_key)

    # Load plants using the more flexible loader from product generator
    try:
        plants = prod_mod.load_plants(prod_mod.CSV_PATH if hasattr(prod_mod, "CSV_PATH") else "plants.csv")
    except Exception as exc:
        logger.exception("Failed to load plants: %s", exc)
        return

    total = len(plants)
    logger.info("Pipeline starting for %d plants", total)

    results: List[Dict[str, Any]] = []

    for idx, plant in enumerate(plants, start=1):
        logger.info("Processing %d/%d: %s", idx, total, plant)
        slug = plant_slug(plant)

        product_path = os.path.join(PRODUCTS_DIR, f"{slug}.json")
        if os.path.exists(product_path):
            logger.info("Product JSON exists: %s", product_path)
            try:
                with open(product_path, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                    results.append(metadata)
            except Exception:
                logger.exception("Failed to read existing product file: %s", product_path)
                # If we couldn't read the file, proceed to regenerate
                metadata = None

            if not FORCE_SHOPIFY_UPLOAD:
                logger.info("Skipping product generation for %s (FORCE_SHOPIFY_UPLOAD disabled)", plant)
                continue

            logger.info("Re-upload mode enabled — uploading existing product for %s", plant)
        else:
            metadata = None

        plant_images_dir = os.path.join(IMAGES_DIR, slug)

        # Generate images (do not fail entire run if images fail)
        try:
            images = generate_images_for_plant(client, plant, plant_images_dir)
        except Exception:
            logger.exception("Image generation step failed for %s", plant)
            images = []

        # Generate product metadata
        try:
            metadata = generate_product_for_plant(client, plant, images)
        except Exception:
            logger.exception("Product generation failed for %s", plant)
            prod_mod.append_failed_log(plant, "pipeline_product_generation_failed")
            # continue to next plant without saving
            time.sleep(1)
            continue

        # Save product JSON
        try:
            saved = save_product_output(metadata, slug)
            logger.info("Saved product JSON: %s", saved)
            results.append(metadata)
        except Exception:
            logger.exception("Failed to save product JSON for %s", plant)
            prod_mod.append_failed_log(plant, "save_product_failed")

        # Optional Shopify upload: upload product to each connected store
        if get_installed_shops and ShopifyClient:
            try:
                try:
                    shops = get_installed_shops()
                except Exception:
                    shops = []

                if not shops:
                    logger.info("No installed Shopify stores found; skipping uploads for %s", plant)
                else:
                    for shop in shops:
                        store = shop.get("store")
                        token = shop.get("token")
                        api_ver = shop.get("api_version")
                        try:
                            shop_client = ShopifyClient(store=store, token=token, api_version=api_ver)
                            if not shop_client.is_configured():
                                logger.info("Skipping shop %s due to missing credentials", store)
                                continue

                            product_id = shop_client.create_product(metadata)
                            try:
                                shop_client.upload_images(product_id, metadata.get("images", []))
                                # Upload additional product metafields (care guide, faqs)
                                try:
                                    shop_client.upload_metafields(product_id, metadata)
                                except Exception:
                                    logger.exception("Failed uploading metafields for %s to %s", plant, store)

                                logger.info("Shopify upload succeeded for %s to store %s (id=%s)", plant, store, product_id)
                            except Exception as exc:
                                logger.exception("Shopify image upload failed for %s to %s: %s", plant, store, exc)
                                prod_mod.append_failed_log(plant, f"shopify_image_upload_failed:{store}:{exc}")
                        except Exception as exc:
                            logger.exception("Shopify upload failed for %s to %s: %s", plant, store, exc)
                            prod_mod.append_failed_log(plant, f"shopify_upload_failed:{store}:{exc}")
            except Exception as exc:
                logger.exception("Unexpected error during Shopify uploads for %s: %s", plant, exc)
                prod_mod.append_failed_log(plant, f"shopify_upload_unexpected:{exc}")
        elif create_product and upload_images:
            # Backwards-compatible single-store upload (env-configured)
            try:
                try:
                    product_id = create_product(metadata)
                except EnvironmentError:
                    logger.info("Shopify credentials missing; skipping upload for %s", plant)
                    product_id = None

                if product_id:
                    try:
                        upload_images(product_id, metadata.get("images", []))
                        logger.info("Shopify upload succeeded for %s (id=%s)", plant, product_id)
                    except Exception as exc:
                        logger.exception("Shopify image upload failed for %s: %s", plant, exc)
                        prod_mod.append_failed_log(plant, f"shopify_image_upload_failed:{exc}")
            except Exception as exc:
                logger.exception("Shopify upload failed for %s: %s", plant, exc)
                prod_mod.append_failed_log(plant, f"shopify_upload_failed:{exc}")

        # Small delay between plants to be polite
        time.sleep(getattr(prod_mod, "DELAY_BETWEEN_PLANTS", 1))

    logger.info("Pipeline completed. Processed: %d products", len(results))


if __name__ == "__main__":
    main()
