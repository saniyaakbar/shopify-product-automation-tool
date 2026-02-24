#!/usr/bin/env python3
"""shopify_uploader.py

Lightweight Shopify Admin REST API uploader used by the pipeline.

Functions:
- create_product(metadata: dict) -> int
- upload_images(product_id: int, images: list)

Environment variables (loaded via dotenv):
- SHOPIFY_STORE (e.g. myshop.myshopify.com)
- SHOPIFY_TOKEN (Admin API access token)
"""
from __future__ import annotations

import os
import json

import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
import requests
from utils.html_formatter import format_description_to_html
from utils.shopify_richtext import to_shopify_richtext
try:
    from backend.auth.shop_repository import get_shop_token
except Exception:
    get_shop_token = None

logger = logging.getLogger("shopify_uploader")


# --- Storage abstraction for shop credentials ---
class ShopService:
    """Abstract shop credential service.

    Subclasses should implement `get_credentials(store: Optional[str]) -> Dict[str, Optional[str]]`
    returning a dict with keys: store, token, api_version.
    """

    def get_credentials(self, store: Optional[str] = None) -> Dict[str, Optional[str]]:
        raise NotImplementedError()


class EnvShopService(ShopService):
    """Simple ShopService that reads credentials from environment variables.

    Expects `SHOPIFY_STORE`, `SHOPIFY_TOKEN`, and optional `SHOPIFY_API_VERSION`.
    """

    def get_credentials(self, store: Optional[str] = None) -> Dict[str, Optional[str]]:
        load_dotenv()
        cfg_store = store or os.getenv("SHOPIFY_STORE")
        # Token will be sourced from DB (OAuth-installed shops); do not read SHOPIFY_TOKEN from env
        cfg_token = None
        cfg_api = os.getenv("SHOPIFY_API_VERSION", "2023-10")
        return {"store": cfg_store, "token": cfg_token, "api_version": cfg_api}


class ShopifyClient:
    """Lightweight Shopify Admin API client.

    Instantiate with explicit `store` and `token` to support multiple stores.
    """

    def __init__(self, store: Optional[str] = None, token: Optional[str] = None, api_version: Optional[str] = None):
        load_dotenv()
        self.store = store or os.getenv("SHOPIFY_STORE")
        self.token = token or os.getenv("SHOPIFY_TOKEN")
        self.api_version = api_version or os.getenv("SHOPIFY_API_VERSION", "2023-10")

        if not self.store or not self.token:
            # Do not raise here; callers can check availability via `is_configured()`
            logger.info("ShopifyClient initialized without full credentials")

    def is_configured(self) -> bool:
        return bool(self.store and self.token)

    def base_url(self) -> str:
        if not self.store:
            raise EnvironmentError("SHOPIFY_STORE not set")
        return f"https://{self.store}/admin/api/{self.api_version}"

    def headers(self) -> Dict[str, str]:
        if not self.token:
            raise EnvironmentError("SHOPIFY_TOKEN not set")
        return {
            "X-Shopify-Access-Token": self.token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def create_product(self, metadata: Dict[str, Any]) -> int:
        """Create a Shopify product (draft) and return its product_id."""
        if not self.is_configured():
            raise EnvironmentError("Shopify credentials missing")

        title = metadata.get("title") or metadata.get("product_title") or metadata.get("product_title")
        body_html = (
            metadata.get("body_html")
            or metadata.get("long_description")
            or metadata.get("short_description")
            or ""
        )
        category = metadata.get("category") or ""
        tags = metadata.get("tags") or []
        tags_str = ",".join([t for t in tags if isinstance(t, str)])

        price = metadata.get("price_aed")
        compare = metadata.get("compare_at_price_aed")
        try:
            price_str = f"{float(price):.2f}" if price is not None else "0.00"
        except Exception:
            price_str = "0.00"
        try:
            compare_str = f"{float(compare):.2f}" if compare is not None else None
        except Exception:
            compare_str = None

        payload = {
            "product": {
                "title": title,
                "body_html": body_html,
                "vendor": "Lusherra",
                "product_type": category,
                "tags": tags_str,
                "variants": [{"price": price_str}],
                "status": "draft",
            }
        }

        if compare_str:
            payload["product"]["variants"][0]["compare_at_price"] = compare_str

        url = f"{self.base_url()}/products.json"
        resp = requests.post(url, headers=self.headers(), data=json.dumps(payload), timeout=30)
        if resp.status_code not in (200, 201):
            logger.error("Shopify create product failed: %s %s", resp.status_code, resp.text)
            raise RuntimeError(f"Shopify create product failed: {resp.status_code}")

        data = resp.json()
        product = data.get("product")
        if not product:
            logger.error("Shopify response missing product data: %s", data)
            raise RuntimeError("Shopify response missing product data")

        product_id = int(product.get("id"))
        logger.info("Created Shopify product %s (id=%d)", title, product_id)
        return product_id

    def create_metafield(
        self,
        product_id: int,
        namespace: str,
        key: str,
        value: str,
        field_type: str = "multi_line_text_field",
    ) -> Dict[str, Any]:
        """Create a metafield for a product.

        Posts to /admin/api/{api_version}/metafields.json with the provided
        namespace, key, owner_id (product_id), owner_resource, type, and value.
        Raises RuntimeError if the response status is not 200/201.
        Returns the created metafield dict from the Shopify response.
        """
        if not self.is_configured():
            raise EnvironmentError("Shopify credentials missing")

        payload = {
            "metafield": {
                "namespace": namespace,
                "key": key,
                "owner_id": product_id,
                "owner_resource": "product",
                "type": field_type,
                "value": value,
            }
        }

        url = f"{self.base_url()}/metafields.json"
        logger.info("Creating metafield %s.%s for product %s", namespace, key, product_id)
        resp = requests.post(url, headers=self.headers(), json=payload, timeout=30)
        if resp.status_code not in (200, 201):
            logger.error("Shopify create metafield failed: %s %s", resp.status_code, resp.text)
            raise RuntimeError(f"Shopify create metafield failed: {resp.status_code}")

        data = resp.json()
        mf = data.get("metafield") or data
        logger.info("Created metafield %s.%s for product %s", namespace, key, product_id)
        return mf

    def upload_images(self, product_id: int, images: List[Dict[str, str]]) -> int:
        """Upload base64 images to product via Shopify Product Image API.

        Uploads images one-by-one (required by Shopify). Continues on individual
        failures and returns the number of successfully uploaded images.
        """
        if not self.is_configured():
            raise EnvironmentError("Shopify credentials missing")

        if not images:
            logger.info("No images to upload for product %s", product_id)
            return 0

        success_count = 0
        url = f"{self.base_url()}/products/{product_id}/images.json"

        for img in images:
            if not isinstance(img, dict):
                continue
            b64 = img.get("b64")
            fname = img.get("filename") or img.get("name") or "image.png"
            if not b64:
                logger.warning("Skipping image %s: missing base64 data", fname)
                continue

            payload = {"image": {"attachment": b64, "filename": fname}}

            logger.info("Uploading image %s to product %s", fname, product_id)
            try:
                resp = requests.post(url, headers=self.headers(), json=payload, timeout=60)
            except Exception as exc:
                logger.error("Shopify image upload exception for %s: %s", fname, exc)
                # continue with next image
                continue

            if resp.status_code not in (200, 201):
                logger.error("Shopify image upload failed for %s: %s", fname, resp.text)
                # do not raise; continue uploading remaining images
                continue

            logger.info("Uploaded image %s for product_id=%s (status=%s)", fname, product_id, resp.status_code)
            success_count += 1

        logger.info("Finished uploading images for product_id=%s: %d succeeded", product_id, success_count)
        return success_count

    def upload_metafields(self, product_id: int, metadata: Dict[str, Any]) -> None:
        """Upload product metafields: care_guide and faqs.

        Does not raise on individual failures; logs outcomes and continues.
        """
        if not self.is_configured():
            raise EnvironmentError("Shopify credentials missing")

        # Use product-scoped metafields endpoint
        url = f"{self.base_url()}/products/{product_id}/metafields.json"

        # Care guide metafield: format markdown headings into HTML before upload
        if metadata.get("care_guide"):
            try:
                formatted = format_description_to_html(metadata["care_guide"])
            except Exception:
                logger.exception("Failed to format care_guide to HTML for product %s", product_id)
                formatted = metadata.get("care_guide")

            payload = {
                "metafield": {
                    "namespace": "custom",
                    "key": "care_guide",
                    "type": "rich_text_field",
                    "value": json.dumps(to_shopify_richtext(formatted), ensure_ascii=False),
                    "owner_id": product_id,
                    "owner_resource": "product",
                }
            }

            product_metafield_url = f"https://{self.store}/admin/api/{self.api_version}/products/{product_id}/metafields.json"

            try:
                logger.info("Uploading care guide metafield for product %s...", product_id)
                resp = requests.post(product_metafield_url, headers=self.headers(), json=payload, timeout=30)
                logger.info(
                    "Care guide metafield upload status=%s response=%s",
                    resp.status_code,
                    resp.text,
                )
            except Exception as exc:
                logger.exception("Exception uploading care_guide metafield for product %s: %s", product_id, exc)

        # FAQs metafield (store as JSON string)
        try:
            faq_value = json.dumps(metadata.get("faqs", []), ensure_ascii=False)
        except Exception:
            faq_value = "[]"

        faq_payload = {
            "metafield": {
                "namespace": "custom",
                "key": "faqs",
                "type": "json",
                "owner_id": product_id,
                "owner_resource": "product",
                "value": faq_value,
            }
        }

        logger.info("Uploading faqs metafield for product %s...", product_id)
        try:
            resp = requests.post(url, headers=self.headers(), json=faq_payload, timeout=30)
        except Exception as exc:
            logger.exception("Exception uploading faqs metafield for product %s: %s", product_id, exc)
        else:
            if resp.status_code in (200, 201):
                logger.info("FAQs uploaded successfully for product %s", product_id)
            else:
                logger.error("FAQs metafield upload failed for product %s: %s", product_id, resp.text)


# Backwards-compatible convenience functions that use env-configured client
def _default_client() -> ShopifyClient:
    # Default client: read SHOPIFY_STORE from env and fetch access token from DB
    load_dotenv()
    store = os.getenv("SHOPIFY_STORE")
    api_version = os.getenv("SHOPIFY_API_VERSION", "2023-10")

    if not store:
        logger.error("SHOPIFY_STORE not set in environment")
        raise RuntimeError("SHOPIFY_STORE not set")

    token = None
    if get_shop_token:
        try:
            token = get_shop_token(store)
            if token:
                logger.info("Loaded Shopify token for store %s from database", store)
            else:
                logger.warning("No access token found in database for store %s", store)
                raise RuntimeError(f"Shop {store} not installed or token missing")
        except Exception as exc:
            logger.exception("Error loading shop token for %s: %s", store, exc)
            raise RuntimeError(f"Error loading shop token for {store}: {exc}")
    else:
        logger.error("shop_repository.get_shop_token not available; cannot load tokens from DB")
        raise RuntimeError("shop repository unavailable")

    logger.info("Creating ShopifyClient for store %s (api_version=%s)", store, api_version)
    return ShopifyClient(store=store, token=token, api_version=api_version)


def create_product(metadata: Dict[str, Any]) -> int:
    client = _default_client()
    return client.create_product(metadata)


def upload_images(product_id: int, images: List[Dict[str, str]]) -> None:
    client = _default_client()
    return client.upload_images(product_id, images)


def get_installed_shops(service: Optional[ShopService] = None) -> List[Dict[str, Optional[str]]]:
    """Return a list of installed shops with credentials.

    The returned list contains dicts with keys: `store`, `token`, `api_version`.
    By default this uses `EnvShopService` which yields a single shop when
    `SHOPIFY_STORE` and `SHOPIFY_TOKEN` are set in the environment.
    """
    # Prefer reading installed shops from the SQLite backend (OAuth installs)
    try:
        from backend.auth.database import get_connection

        conn = get_connection()
        try:
            cur = conn.cursor()
            # alias shop_domain -> shop so we can access as 'shop' below
            cur.execute("SELECT shop_domain AS shop, access_token FROM shops")
            rows = cur.fetchall()
            shops: List[Dict[str, Optional[str]]] = []
            for r in rows:
                # r is sqlite3.Row thanks to get_connection() row_factory
                store = r["shop"]
                token = r["access_token"]
                shops.append({"store": store, "token": token, "api_version": "2024-10"})

            logger.info("Loaded %d installed shops from database", len(shops))
            if shops:
                return shops
        finally:
            conn.close()
    except Exception:
        logger.debug("Could not load installed shops from DB, falling back to service", exc_info=True)

    # Fallback to provided service (EnvShopService by default)
    svc = service or EnvShopService()
    creds = svc.get_credentials()
    store = creds.get("store")
    token = creds.get("token")
    api_version = creds.get("api_version") or "2024-10"
    if store and token:
        logger.info("Loaded 1 installed shop from environment variables")
        return [{"store": store, "token": token, "api_version": api_version}]

    logger.info("No installed shops found")
    return []
