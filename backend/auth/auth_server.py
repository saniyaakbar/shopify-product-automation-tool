import os
import logging
import secrets
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
import requests
from dotenv import load_dotenv

from .shop_repository import save_shop

load_dotenv()

logger = logging.getLogger("auth_server")
logging.basicConfig(level=logging.INFO)

SHOPIFY_CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID")
SHOPIFY_CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")

app = FastAPI()

# simple in-memory state store (for demo). In production, persist this.
_STATE_STORE = {}


@app.get("/install")
def install(shop: Optional[str] = None):
    if not shop:
        raise HTTPException(status_code=400, detail="missing shop parameter")
    if not SHOPIFY_CLIENT_ID:
        raise HTTPException(status_code=500, detail="server not configured")

    state = secrets.token_urlsafe(16)
    _STATE_STORE[state] = shop

    redirect_uri = f"{APP_BASE_URL}/auth/callback"
    scopes = os.getenv("SHOPIFY_SCOPES", "write_products,write_product_listings")
    install_url = (
        f"https://{shop}/admin/oauth/authorize?client_id={SHOPIFY_CLIENT_ID}"
        f"&scope={scopes}&redirect_uri={redirect_uri}&state={state}"
    )
    logger.info("Redirecting install for shop=%s state=%s", shop, state)
    return RedirectResponse(install_url)


@app.get("/auth/callback")
def callback(request: Request, shop: Optional[str] = None, code: Optional[str] = None, state: Optional[str] = None):
    if not shop or not code:
        raise HTTPException(status_code=400, detail="missing parameters")

    # validate state loosely
    if state and state in _STATE_STORE:
        expected = _STATE_STORE.pop(state, None)
        if expected and expected != shop:
            logger.warning("State shop mismatch: expected=%s got=%s", expected, shop)

    token_url = f"https://{shop}/admin/oauth/access_token"
    payload = {
        "client_id": SHOPIFY_CLIENT_ID,
        "client_secret": SHOPIFY_CLIENT_SECRET,
        "code": code,
    }

    try:
        resp = requests.post(token_url, json=payload, timeout=15)
    except Exception as exc:
        logger.exception("Error exchanging code for token: %s", exc)
        raise HTTPException(status_code=500, detail="token exchange failed")

    if resp.status_code != 200:
        logger.error("Token exchange failed: %s %s", resp.status_code, resp.text)
        raise HTTPException(status_code=500, detail="token exchange failed")

    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        logger.error("No access_token in response: %s", data)
        raise HTTPException(status_code=500, detail="no access token")

    # persist to database
    try:
        save_shop(shop, access_token)
    except Exception:
        logger.exception("Failed to save shop token for %s", shop)
        raise HTTPException(status_code=500, detail="failed to save token")

    html = HTMLResponse(content=f"<html><body>App installed for {shop}. You may close this window.</body></html>")
    logger.info("Installed shop=%s", shop)
    return html


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
