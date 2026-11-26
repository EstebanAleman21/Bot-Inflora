"""
WhatsApp Business Order Bot — Vercel Serverless (FastAPI + Upstash Redis)
=========================================================================
Entry point for all WhatsApp Cloud API webhook events.

Flow overview:
  Staff sends a message → Meta POSTs to /api/webhook → process_message()
  routes to the correct state-machine handler → handler sends a WhatsApp
  reply and updates the session in Redis.

Environment variables (see .env.example):
  WHATSAPP_VERIFY_TOKEN, WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID,
  WHATSAPP_API_VERSION, SHOPIFY_STORE_DOMAIN, SHOPIFY_ADMIN_TOKEN,
  SHOPIFY_API_VERSION, ALLOWED_STAFF_PHONES,
  UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, timedelta
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from upstash_redis.asyncio import Redis

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()  # no-op in production (Vercel injects env vars directly)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

WA_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
WA_TOKEN: str = os.getenv("WHATSAPP_TOKEN", "")
WA_PHONE_ID: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WA_API_VERSION: str = os.getenv("WHATSAPP_API_VERSION", "v21.0")

SHOPIFY_DOMAIN: str = os.getenv("SHOPIFY_STORE_DOMAIN", "")
SHOPIFY_VERSION: str = os.getenv("SHOPIFY_API_VERSION", "2025-01")
# New Dev Dashboard apps use client_credentials (expiring tokens).
# Legacy admin-created apps can still set SHOPIFY_ADMIN_TOKEN as a static fallback.
SHOPIFY_CLIENT_ID: str = os.getenv("SHOPIFY_CLIENT_ID", "")
SHOPIFY_CLIENT_SECRET: str = os.getenv("SHOPIFY_CLIENT_SECRET", "")
SHOPIFY_STATIC_TOKEN: str = os.getenv("SHOPIFY_ADMIN_TOKEN", "")

# Parse allowlist once at startup; store as a set for O(1) lookup.
_raw_phones = os.getenv("ALLOWED_STAFF_PHONES", "")
ALLOWED_PHONES: set[str] = {p.strip() for p in _raw_phones.split(",") if p.strip()}

# Derived URLs
WA_API_URL = f"https://graph.facebook.com/{WA_API_VERSION}/{WA_PHONE_ID}/messages"
SHOPIFY_GQL_URL = (
    f"https://{SHOPIFY_DOMAIN}/admin/api/{SHOPIFY_VERSION}/graphql.json"
)

# Redis TTLs (seconds)
SESSION_TTL = 1800   # 30 min — session expires after inactivity
CATALOG_TTL = 1800   # 30 min — catalog cache

# ---------------------------------------------------------------------------
# Redis client (singleton, lazily initialised)
# ---------------------------------------------------------------------------
# Vercel's Upstash Storage integration injects KV-prefixed variable names.
# We read those explicitly and fall back to the plain names for local dev.

_redis: Redis | None = None


def get_redis() -> Redis:
    """Return the shared async Redis client, creating it on first call."""
    global _redis
    if _redis is None:
        url = (
            os.getenv("UPSTASH_REDIS_REST_KV_REST_API_URL")  # Vercel Storage integration
            or os.getenv("UPSTASH_REDIS_REST_URL", "")        # local dev fallback
        )
        token = (
            os.getenv("UPSTASH_REDIS_REST_KV_REST_API_TOKEN")
            or os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
        )
        _redis = Redis(url=url, token=token)
    return _redis


# ===========================================================================
# SECTION 1 — WhatsApp Cloud API helpers
# ===========================================================================

async def _wa_post(payload: dict) -> dict:
    """
    POST a payload to the WhatsApp Cloud API messages endpoint.
    Returns the parsed JSON response (or raises on HTTP errors).
    """
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(WA_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def send_text(to: str, body: str) -> None:
    """Send a plain-text WhatsApp message to *to* (international phone number)."""
    await _wa_post({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    })


async def send_buttons(to: str, body: str, buttons: list[dict]) -> None:
    """
    Send an interactive reply-buttons message.

    Meta limits:
      - Maximum 3 buttons per message.
      - Button title max 20 characters (enforced by truncation).
      - Each button id must be unique.

    Each button dict: {"id": str, "title": str}
    """
    # Enforce Meta limits
    buttons = buttons[:3]
    safe_buttons = [
        {
            "type": "reply",
            "reply": {
                "id": btn["id"],
                "title": btn["title"][:20],
            },
        }
        for btn in buttons
    ]

    await _wa_post({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body},
            "action": {"buttons": safe_buttons},
        },
    })


async def send_list(
    to: str,
    body: str,
    button_text: str,
    section_title: str,
    rows: list[dict],
) -> None:
    """
    Send an interactive list message.

    Meta limits:
      - Maximum 10 rows total.
      - Row title max 24 characters.
      - Row description max 72 characters.

    Each row dict: {"id": str, "title": str, "description": str}

    The description is a good place to put the full product name + price so
    that truncated 24-char titles remain informative to staff.
    """
    rows = rows[:10]
    safe_rows = [
        {
            "id": row["id"],
            "title": row["title"][:24],
            "description": row.get("description", "")[:72],
        }
        for row in rows
    ]

    await _wa_post({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": body},
            "action": {
                "button": button_text[:20],
                "sections": [
                    {"title": section_title[:24], "rows": safe_rows}
                ],
            },
        },
    })


# ===========================================================================
# SECTION 2 — Shopify GraphQL helpers
# ===========================================================================

async def _get_shopify_token() -> str:
    """
    Return a valid Shopify Admin API access token.

    Strategy (in order):
      1. If SHOPIFY_ADMIN_TOKEN is set (legacy static token), use it directly.
      2. Otherwise use SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET to request a
         short-lived token via the OAuth2 client_credentials grant. The token
         is cached in Redis under "shopify:token" with a TTL 60s shorter than
         its actual expiry so we refresh it before it goes stale.
    """
    # Static token path (legacy admin-created apps)
    if SHOPIFY_STATIC_TOKEN:
        return SHOPIFY_STATIC_TOKEN

    redis = get_redis()

    # Check Redis cache first
    cached = await redis.get("shopify:token")
    if cached:
        return cached

    # Fetch a fresh token via client_credentials
    token_url = f"https://{SHOPIFY_DOMAIN}/admin/oauth/access_token"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            token_url,
            data={
                "client_id": SHOPIFY_CLIENT_ID,
                "client_secret": SHOPIFY_CLIENT_SECRET,
                "grant_type": "client_credentials",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    token: str = data["access_token"]
    # Cache with a 60s safety buffer before actual expiry (default ~24h)
    ttl = int(data.get("expires_in", 86400)) - 60
    await redis.set("shopify:token", token, ex=ttl)
    log.info("Fetched fresh Shopify access token (TTL %ds)", ttl)
    return token


async def _shopify_gql(query: str, variables: dict | None = None) -> dict:
    """
    Execute a Shopify Admin GraphQL request and return the parsed response.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    token = await _get_shopify_token()
    headers = {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(SHOPIFY_GQL_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Catalog caching
# ---------------------------------------------------------------------------

_PRODUCTS_QUERY = """
query GetProducts($cursor: String) {
  products(first: 250, after: $cursor, query: "status:active") {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        title
        collections(first: 10) {
          edges { node { id } }
        }
        variants(first: 1) {
          edges {
            node {
              id
              price
            }
          }
        }
      }
    }
  }
}
"""

_COLLECTIONS_QUERY = """
query GetCollections($cursor: String) {
  collections(first: 250, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        title
        productsCount { count }
      }
    }
  }
}
"""


async def get_catalog(force_refresh: bool = False) -> dict:
    """
    Return the product+collection catalog dict.

    Structure:
      {
        "products": {
          "<product_gid>": {
            "id": str,
            "title": str,
            "variantId": str,
            "price": str,
            "collectionIds": [str, ...]
          }
        },
        "collections": [
          {"id": str, "title": str, "productIds": [str, ...]}
          ...  sorted by productCount desc, top 8 used in menu
        ]
      }

    Data is cached in Redis under "shopify:catalog" for CATALOG_TTL seconds.
    Pass force_refresh=True to bypass the cache (used by /sync command).
    """
    redis = get_redis()

    # Try the cache first (unless a forced refresh was requested)
    if not force_refresh:
        cached = await redis.get("shopify:catalog")
        if cached:
            return json.loads(cached)

    log.info("Fetching fresh catalog from Shopify…")

    # ── Paginate through all active products ─────────────────────────────
    products: dict[str, dict] = {}
    # collection_id → set of product_ids (for building productIds list)
    col_product_map: dict[str, set[str]] = {}

    cursor = None
    while True:
        data = await _shopify_gql(_PRODUCTS_QUERY, {"cursor": cursor})
        page = data["data"]["products"]
        for edge in page["edges"]:
            node = edge["node"]
            pid = node["id"]
            variant_node = node["variants"]["edges"][0]["node"] if node["variants"]["edges"] else None
            if not variant_node:
                continue  # skip products with no variants

            col_ids = [c["node"]["id"] for c in node["collections"]["edges"]]
            products[pid] = {
                "id": pid,
                "title": node["title"],
                "variantId": variant_node["id"],
                "price": variant_node["price"],
                "collectionIds": col_ids,
            }
            for cid in col_ids:
                col_product_map.setdefault(cid, set()).add(pid)

        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]

    # ── Paginate through all collections ────────────────────────────────
    raw_collections: list[dict] = []
    cursor = None
    while True:
        data = await _shopify_gql(_COLLECTIONS_QUERY, {"cursor": cursor})
        page = data["data"]["collections"]
        for edge in page["edges"]:
            node = edge["node"]
            cid = node["id"]
            raw_collections.append({
                "id": cid,
                "title": node["title"],
                "productIds": list(col_product_map.get(cid, set())),
            })
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]

    # Sort collections by product count descending so the most stocked
    # categories appear first in the menu.
    raw_collections.sort(key=lambda c: len(c["productIds"]), reverse=True)

    catalog = {"products": products, "collections": raw_collections}

    # Persist to Redis
    await redis.set("shopify:catalog", json.dumps(catalog), ex=CATALOG_TTL)
    log.info(
        "Catalog cached: %d products, %d collections",
        len(products),
        len(raw_collections),
    )
    return catalog


# ---------------------------------------------------------------------------
# Shopify order creation
# ---------------------------------------------------------------------------

_DRAFT_ORDER_CREATE = """
mutation DraftOrderCreate($input: DraftOrderInput!) {
  draftOrderCreate(input: $input) {
    draftOrder {
      id
      name
      invoiceUrl
    }
    userErrors { field message }
  }
}
"""

_DRAFT_ORDER_COMPLETE = """
mutation DraftOrderComplete($id: ID!) {
  draftOrderComplete(id: $id) {
    draftOrder {
      order {
        id
        name
      }
    }
    userErrors { field message }
  }
}
"""


async def create_draft_order(session: dict) -> dict:
    """
    Create a Shopify Draft Order from the session data.

    Returns:
      {"draftOrderId": str, "name": str, "invoiceUrl": str}

    Raises ValueError if Shopify returns userErrors.
    """
    line_items = []
    for item in session.get("items", []):
        if item.get("is_custom"):
            line_items.append({
                "title": item["title"],
                "originalUnitPrice": str(item["price"]),
                "quantity": item["quantity"],
            })
        else:
            line_items.append({
                "variantId": item["variant_id"],
                "quantity": item["quantity"],
            })

    custom_attrs = [
        {"key": "Delivery Date", "value": session.get("delivery_date", "")},
        {"key": "Delivery Window", "value": session.get("delivery_window", "")},
        {"key": "Recipient Details", "value": session.get("recipient_details", "")},
    ]

    draft_input: dict[str, Any] = {
        "lineItems": line_items,
        "customAttributes": custom_attrs,
    }

    # Only set note when there's an actual card message
    card_msg = session.get("card_message", "none")
    if card_msg and card_msg.lower() != "none":
        draft_input["note"] = card_msg

    result = await _shopify_gql(_DRAFT_ORDER_CREATE, {"input": draft_input})
    payload = result["data"]["draftOrderCreate"]

    if payload["userErrors"]:
        errors = "; ".join(f"{e['field']}: {e['message']}" for e in payload["userErrors"])
        raise ValueError(f"Shopify draft order error: {errors}")

    draft = payload["draftOrder"]
    return {
        "draftOrderId": draft["id"],
        "name": draft["name"],
        "invoiceUrl": draft.get("invoiceUrl", ""),
    }


async def complete_draft_order(draft_order_id: str) -> str:
    """
    Complete (pay and finalise) a Shopify draft order.

    Returns the resulting order name (e.g. "#1045").
    Raises ValueError on userErrors.
    """
    result = await _shopify_gql(_DRAFT_ORDER_COMPLETE, {"id": draft_order_id})
    payload = result["data"]["draftOrderComplete"]

    if payload["userErrors"]:
        errors = "; ".join(f"{e['field']}: {e['message']}" for e in payload["userErrors"])
        raise ValueError(f"Shopify complete order error: {errors}")

    return payload["draftOrder"]["order"]["name"]


# ===========================================================================
# SECTION 3 — Session helpers (Upstash Redis)
# ===========================================================================

def _session_key(phone: str) -> str:
    return f"session:{phone}"


async def get_session(phone: str) -> dict | None:
    """Load session JSON from Redis. Returns None if no session exists."""
    redis = get_redis()
    raw = await redis.get(_session_key(phone))
    if raw is None:
        return None
    return json.loads(raw)


async def save_session(phone: str, data: dict) -> None:
    """Persist session JSON to Redis with SESSION_TTL."""
    redis = get_redis()
    await redis.set(_session_key(phone), json.dumps(data), ex=SESSION_TTL)


async def delete_session(phone: str) -> None:
    """Remove a session from Redis (order complete or cancelled)."""
    redis = get_redis()
    await redis.delete(_session_key(phone))


def new_session() -> dict:
    """Return a blank session dict at the first step."""
    return {
        "step": "SELECT_CATEGORY",
        "items": [],
        "filtered_product_ids": [],
        "page": 0,
        "selected_collection_id": None,
        "delivery_date": None,
        "delivery_window": None,
        "recipient_details": None,
        "card_message": None,
        "payment_mode": None,
    }


# ===========================================================================
# SECTION 4 — State-machine step handlers
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 1 — SELECT_CATEGORY
# ---------------------------------------------------------------------------

async def show_category_menu(phone: str, session: dict, catalog: dict) -> None:
    """
    Send the main category selection list.
    Shows top-8 Shopify collections + Search + Custom item rows.
    """
    rows = []

    # Build rows for the top 8 collections
    for col in catalog["collections"][:8]:
        # Strip the GID prefix to get a short numeric id for the row id
        col_short_id = col["id"].split("/")[-1]
        count = len(col["productIds"])
        rows.append({
            "id": f"COL_{col_short_id}",
            "title": col["title"][:24],
            "description": f"{count} products",
        })

    # Fixed action rows
    rows.append({
        "id": "ACT_SEARCH",
        "title": "Search by Name",
        "description": "Find any product by keyword",
    })
    rows.append({
        "id": "ACT_CUSTOM",
        "title": "Custom Item ($)",
        "description": "Enter a custom name & price",
    })

    await send_list(
        phone,
        body="Select a category or action:",
        button_text="Choose",
        section_title="Categories",
        rows=rows,
    )


async def handle_select_category(
    phone: str,
    session: dict,
    catalog: dict,
    reply_id: str,
) -> None:
    """
    Process the user's category selection.

    reply_id patterns:
      COL_<numeric_id>  → browse a collection
      ACT_SEARCH        → search by keyword
      ACT_CUSTOM        → enter a custom item
    """
    if reply_id.startswith("COL_"):
        # Map short id back to full GID
        short_id = reply_id[4:]  # strip "COL_"
        full_gid = f"gid://shopify/Collection/{short_id}"

        # Collect product ids for this collection
        product_ids = []
        for col in catalog["collections"]:
            if col["id"] == full_gid:
                product_ids = col["productIds"]
                break

        session["selected_collection_id"] = full_gid
        session["filtered_product_ids"] = product_ids
        session["page"] = 0
        session["step"] = "SELECT_PRODUCT"
        await save_session(phone, session)
        await show_product_page(phone, session, catalog)

    elif reply_id == "ACT_SEARCH":
        session["step"] = "SEARCH_QUERY"
        await save_session(phone, session)
        await send_text(phone, "Type a keyword to search all products:")

    elif reply_id == "ACT_CUSTOM":
        session["step"] = "CUSTOM_ITEM"
        await save_session(phone, session)
        await send_text(
            phone,
            "Enter the custom item in this format:\n"
            "*Item Name - Price*\n\n"
            "Example: Custom Bouquet - 1500",
        )


# ---------------------------------------------------------------------------
# Step 2a — SEARCH_QUERY
# ---------------------------------------------------------------------------

async def handle_search_query(
    phone: str,
    session: dict,
    catalog: dict,
    text: str,
) -> None:
    """
    Case-insensitive substring search across all cached product titles.
    If no matches, prompt again. Otherwise move to SELECT_PRODUCT.
    """
    keyword = text.strip().lower()
    matches = [
        pid
        for pid, prod in catalog["products"].items()
        if keyword in prod["title"].lower()
    ]

    if not matches:
        await send_text(
            phone,
            f'No products found for "{text}". Try a different keyword:',
        )
        return  # Stay in SEARCH_QUERY step

    session["filtered_product_ids"] = matches
    session["page"] = 0
    session["step"] = "SELECT_PRODUCT"
    await save_session(phone, session)
    await show_product_page(phone, session, catalog)


# ---------------------------------------------------------------------------
# Step 2b — CUSTOM_ITEM
# ---------------------------------------------------------------------------

async def handle_custom_item(phone: str, session: dict, text: str) -> None:
    """
    Parse "Item Name - Price" input.
    Validates that price is numeric, then adds to session items.
    """
    # Split on the LAST hyphen so item names can contain hyphens
    parts = text.rsplit("-", 1)
    if len(parts) != 2:
        await send_text(
            phone,
            "Format not recognised. Please use:\n*Item Name - Price*\n\nExample: Sunflower Mix - 850",
        )
        return

    item_name = parts[0].strip()
    price_str = parts[1].strip().replace(",", "").replace("$", "")

    try:
        price = float(price_str)
    except ValueError:
        await send_text(phone, f'"{parts[1].strip()}" is not a valid price. Please try again:')
        return

    session["items"].append({
        "title": item_name,
        "price": price,
        "quantity": 1,
        "is_custom": True,
    })
    session["step"] = "ITEM_ACTION"
    await save_session(phone, session)
    await show_item_action(phone, session)


# ---------------------------------------------------------------------------
# Step 3a — SELECT_PRODUCT (paginated)
# ---------------------------------------------------------------------------

async def show_product_page(phone: str, session: dict, catalog: dict) -> None:
    """
    Display a paginated product selection list (up to 9 products + optional Next Page row).
    """
    product_ids = session["filtered_product_ids"]
    page = session.get("page", 0)
    page_size = 9

    start = page * page_size
    end = start + page_size
    page_ids = product_ids[start:end]
    has_more = end < len(product_ids)

    rows = []
    for pid in page_ids:
        prod = catalog["products"].get(pid)
        if not prod:
            continue
        variant_short = prod["variantId"].split("/")[-1]
        rows.append({
            "id": f"PROD_{variant_short}",
            "title": prod["title"][:24],
            # Full title + price in description so truncated title is still clear
            "description": f"${prod['price']} — {prod['title']}"[:72],
        })

    if has_more:
        rows.append({
            "id": "PAGE_NEXT",
            "title": "Next Page",
            "description": f"Page {page + 2}",
        })

    total = len(product_ids)
    body = (
        f"Showing {start + 1}–{min(end, total)} of {total} products.\n"
        "Select a product to add to the order:"
    )

    await send_list(
        phone,
        body=body,
        button_text="Select",
        section_title="Products",
        rows=rows,
    )


async def handle_select_product(
    phone: str,
    session: dict,
    catalog: dict,
    reply_id: str,
) -> None:
    """
    Handle a product selection or pagination action from the product list.
    """
    if reply_id == "PAGE_NEXT":
        session["page"] = session.get("page", 0) + 1
        await save_session(phone, session)
        await show_product_page(phone, session, catalog)
        return

    if reply_id.startswith("PROD_"):
        variant_short = reply_id[5:]  # strip "PROD_"
        full_variant_gid = f"gid://shopify/ProductVariant/{variant_short}"

        # Look up the product by variant id
        product = None
        for prod in catalog["products"].values():
            if prod["variantId"] == full_variant_gid:
                product = prod
                break

        if not product:
            await send_text(phone, "Product not found. Please try again.")
            return

        session["items"].append({
            "variant_id": full_variant_gid,
            "title": product["title"],
            "price": product["price"],
            "quantity": 1,
            "is_custom": False,
        })
        session["step"] = "ITEM_ACTION"
        await save_session(phone, session)
        await show_item_action(phone, session)


# ---------------------------------------------------------------------------
# Step 3b — ITEM_ACTION
# ---------------------------------------------------------------------------

def _build_cart_summary(session: dict) -> str:
    """Return a formatted string listing all items currently in the cart."""
    lines = ["*Current cart:*"]
    total = 0.0
    for i, item in enumerate(session["items"], 1):
        price = float(item["price"])
        qty = item["quantity"]
        subtotal = price * qty
        total += subtotal
        lines.append(f"  {i}. {item['title']} x{qty} — ${subtotal:,.2f}")
    lines.append(f"\n*Total: ${total:,.2f}*")
    return "\n".join(lines)


async def show_item_action(phone: str, session: dict) -> None:
    """
    Send the item-action buttons after a product is added to the cart.
    Shows the current cart summary + 3 action buttons.
    """
    cart_text = _build_cart_summary(session)
    last_item = session["items"][-1] if session["items"] else {}
    body = (
        f"{cart_text}\n\n"
        f"Last added: *{last_item.get('title', '')}*\n"
        "What would you like to do?"
    )

    await send_buttons(
        phone,
        body=body,
        buttons=[
            {"id": "BTN_QTY_PLUS", "title": "Qty +1"},
            {"id": "BTN_ADD_MORE", "title": "Add Another"},
            {"id": "BTN_ITEMS_DONE", "title": "Done Items"},
        ],
    )


async def handle_item_action(
    phone: str,
    session: dict,
    catalog: dict,
    reply_id: str,
) -> None:
    """
    Handle the three item-action buttons.
    """
    if reply_id == "BTN_QTY_PLUS":
        # Increment quantity of the last item in the cart
        if session["items"]:
            session["items"][-1]["quantity"] += 1
        await save_session(phone, session)
        await show_item_action(phone, session)  # re-send with updated cart

    elif reply_id == "BTN_ADD_MORE":
        session["step"] = "SELECT_CATEGORY"
        await save_session(phone, session)
        catalog = await get_catalog()
        await show_category_menu(phone, session, catalog)

    elif reply_id == "BTN_ITEMS_DONE":
        session["step"] = "SELECT_DATE"
        await save_session(phone, session)
        await show_select_date(phone, session)


# ---------------------------------------------------------------------------
# Step 4 — SELECT_DATE / ENTER_DATE / SELECT_WINDOW
# ---------------------------------------------------------------------------

async def show_select_date(phone: str, session: dict) -> None:
    """Send the delivery date selection buttons."""
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    await send_buttons(
        phone,
        body="Select the delivery date:",
        buttons=[
            {"id": "DATE_TODAY", "title": f"Today ({today})"},
            {"id": "DATE_TOMORROW", "title": "Tomorrow"},
            {"id": "DATE_OTHER", "title": "Other Date"},
        ],
    )


async def handle_select_date(
    phone: str,
    session: dict,
    reply_id: str,
    text: str,
) -> None:
    """
    Handle date selection.

    reply_id is set when the user taps a button; text is set when they type
    a free-form date (ENTER_DATE step).
    """
    current_step = session.get("step")

    if current_step == "ENTER_DATE":
        # Staff typed a custom date string — save it and move on
        session["delivery_date"] = text.strip()
        session["step"] = "SELECT_WINDOW"
        await save_session(phone, session)
        await show_select_window(phone, session)
        return

    # Handling button reply in SELECT_DATE step
    if reply_id == "DATE_TODAY":
        session["delivery_date"] = date.today().isoformat()
        session["step"] = "SELECT_WINDOW"
        await save_session(phone, session)
        await show_select_window(phone, session)

    elif reply_id == "DATE_TOMORROW":
        session["delivery_date"] = (date.today() + timedelta(days=1)).isoformat()
        session["step"] = "SELECT_WINDOW"
        await save_session(phone, session)
        await show_select_window(phone, session)

    elif reply_id == "DATE_OTHER":
        session["step"] = "ENTER_DATE"
        await save_session(phone, session)
        await send_text(phone, "Type the delivery date (e.g. 2026-12-25 or Dec 25):")


async def show_select_window(phone: str, session: dict) -> None:
    """Send the delivery time-window selection list."""
    rows = [
        {"id": "WIN_1", "title": "9:00 AM - 12:00 PM",  "description": "Morning"},
        {"id": "WIN_2", "title": "12:00 PM - 3:00 PM",  "description": "Afternoon"},
        {"id": "WIN_3", "title": "3:00 PM - 6:00 PM",   "description": "Late afternoon"},
        {"id": "WIN_4", "title": "6:00 PM - 9:00 PM",   "description": "Evening"},
        {"id": "WIN_5", "title": "Store Pickup",         "description": "Customer picks up in store"},
    ]

    await send_list(
        phone,
        body=f"Delivery date: *{session['delivery_date']}*\n\nSelect a delivery window:",
        button_text="Choose window",
        section_title="Time Windows",
        rows=rows,
    )


_WINDOW_LABELS = {
    "WIN_1": "9:00 AM - 12:00 PM",
    "WIN_2": "12:00 PM - 3:00 PM",
    "WIN_3": "3:00 PM - 6:00 PM",
    "WIN_4": "6:00 PM - 9:00 PM",
    "WIN_5": "Store Pickup",
}


async def handle_select_window(phone: str, session: dict, reply_id: str) -> None:
    """Save the chosen delivery window and move to ENTER_RECIPIENT."""
    label = _WINDOW_LABELS.get(reply_id, reply_id)
    session["delivery_window"] = label
    session["step"] = "ENTER_RECIPIENT"
    await save_session(phone, session)
    await send_text(
        phone,
        "Reply with the recipient's *Name*, *Phone*, and *Delivery Address*:\n\n"
        "Example: Maria Lopez, 5218113456789, Av. Constitución 100, Monterrey",
    )


# ---------------------------------------------------------------------------
# Step 5a — ENTER_RECIPIENT
# ---------------------------------------------------------------------------

async def handle_enter_recipient(phone: str, session: dict, text: str) -> None:
    """Save recipient details and move to ENTER_CARD."""
    session["recipient_details"] = text.strip()
    session["step"] = "ENTER_CARD"
    await save_session(phone, session)
    await send_text(
        phone,
        "Reply with the *gift card message* (or type _none_ if no card):",
    )


# ---------------------------------------------------------------------------
# Step 5b — ENTER_CARD
# ---------------------------------------------------------------------------

async def handle_enter_card(phone: str, session: dict, text: str) -> None:
    """Save card message and move to SELECT_PAYMENT."""
    session["card_message"] = text.strip()
    session["step"] = "SELECT_PAYMENT"
    await save_session(phone, session)
    await send_buttons(
        phone,
        body="How is this order being paid?",
        buttons=[
            {"id": "PAY_LINK", "title": "Payment Link"},
            {"id": "PAY_PAID", "title": "Already Paid"},
        ],
    )


# ---------------------------------------------------------------------------
# Step 5c — SELECT_PAYMENT → CONFIRM
# ---------------------------------------------------------------------------

async def handle_select_payment(phone: str, session: dict, reply_id: str) -> None:
    """Save payment mode and show the final order confirmation summary."""
    session["payment_mode"] = reply_id  # "PAY_LINK" or "PAY_PAID"
    session["step"] = "CONFIRM"
    await save_session(phone, session)
    await show_order_summary(phone, session)


async def show_order_summary(phone: str, session: dict) -> None:
    """
    Build and send a comprehensive order summary with Confirm / Cancel buttons.
    """
    # Build items block
    item_lines = []
    total = 0.0
    for item in session["items"]:
        price = float(item["price"])
        qty = item["quantity"]
        sub = price * qty
        total += sub
        item_lines.append(f"  • {item['title']} x{qty} = ${sub:,.2f}")

    payment_label = "Payment Link" if session["payment_mode"] == "PAY_LINK" else "Already Paid"
    card_msg = session.get("card_message", "none")

    summary = (
        "*ORDER SUMMARY*\n\n"
        "*Items:*\n"
        + "\n".join(item_lines)
        + f"\n*Total: ${total:,.2f}*\n\n"
        f"*Delivery:* {session.get('delivery_date')} | {session.get('delivery_window')}\n"
        f"*Recipient:* {session.get('recipient_details')}\n"
        f"*Card message:* {card_msg}\n"
        f"*Payment:* {payment_label}\n\n"
        "Confirm and create the Shopify order?"
    )

    await send_buttons(
        phone,
        body=summary,
        buttons=[
            {"id": "CONFIRM_CREATE", "title": "Create Order"},
            {"id": "CONFIRM_CANCEL", "title": "Cancel"},
        ],
    )


# ---------------------------------------------------------------------------
# Step 5d — CONFIRM
# ---------------------------------------------------------------------------

async def handle_confirm(phone: str, session: dict, reply_id: str) -> None:
    """
    Final confirmation step.
    CONFIRM_CREATE → create Shopify draft order (+ complete if already paid).
    CONFIRM_CANCEL → cancel and clean up session.
    """
    if reply_id == "CONFIRM_CANCEL":
        await delete_session(phone)
        await send_text(phone, "Order cancelled. Type *new* to start again.")
        return

    if reply_id != "CONFIRM_CREATE":
        return

    # ── Create Shopify draft order ────────────────────────────────────────
    try:
        draft = await create_draft_order(session)
    except ValueError as exc:
        await send_text(phone, f"Shopify error: {exc}\n\nType *cancel* or try again.")
        return

    payment_mode = session.get("payment_mode")

    if payment_mode == "PAY_PAID":
        # Complete the draft into a finalised paid order
        try:
            order_name = await complete_draft_order(draft["draftOrderId"])
        except ValueError as exc:
            await send_text(phone, f"Shopify complete error: {exc}\n\nDraft saved as {draft['name']}.")
            await delete_session(phone)
            return

        confirmation = (
            f"Order *{order_name}* created and marked as paid in Shopify!\n\n"
            "Type *new* to start another order."
        )
    else:
        # PAY_LINK — share the invoice URL with the staff member
        confirmation = (
            f"Draft order *{draft['name']}* created.\n\n"
            f"Payment link:\n{draft['invoiceUrl']}\n\n"
            "Share this link with the customer. Type *new* to start another order."
        )

    await delete_session(phone)
    await send_buttons(
        phone,
        body=confirmation,
        buttons=[{"id": "BTN_NEW_ORDER", "title": "New Order"}],
    )


# ===========================================================================
# SECTION 5 — Main message dispatcher
# ===========================================================================

async def process_message(
    phone: str,
    msg_type: str,
    text: str,
    reply_id: str,
) -> None:
    """
    Central dispatcher for all incoming WhatsApp messages.

    Args:
      phone:    Sender's phone number (international, no +).
      msg_type: "text", "button_reply", or "list_reply".
      text:     Normalised message text (lowercase for text type, raw for others).
      reply_id: Interactive reply id (empty string for plain text).
    """
    # ── 1. Staff authorisation ────────────────────────────────────────────
    if phone not in ALLOWED_PHONES:
        log.warning("Ignored message from unauthorised number: %s", phone)
        return  # Silently ignore — do not reveal the bot to unknown senders

    # ── 2. Global commands (always available, bypass state machine) ───────
    text_lower = text.lower().strip()

    if text_lower in ("cancel", "salir"):
        await delete_session(phone)
        await send_text(phone, "Order cancelled. Type *new* to start again.")
        return

    if text_lower == "/sync":
        catalog = await get_catalog(force_refresh=True)
        product_count = len(catalog["products"])
        col_count = len(catalog["collections"])
        await send_text(
            phone,
            f"Catalog synced: *{product_count} products* and *{col_count} collections*.",
        )
        return

    # ── 3. Session init / reset triggers ─────────────────────────────────
    session = await get_session(phone)

    if text_lower in ("new", "orden", "hola") or reply_id == "BTN_NEW_ORDER":
        session = new_session()
        await save_session(phone, session)
        catalog = await get_catalog()
        await show_category_menu(phone, session, catalog)
        return

    # If there is no active session and the message is not an init trigger,
    # guide the staff member.
    if session is None:
        await send_text(
            phone,
            "Hi! Type *new* (or *orden* / *hola*) to start a new order.",
        )
        return

    # ── 4. Route to the correct step handler ─────────────────────────────
    step = session.get("step", "SELECT_CATEGORY")
    catalog = await get_catalog()  # always served from Redis cache (fast)

    if step == "SELECT_CATEGORY":
        await handle_select_category(phone, session, catalog, reply_id or text_lower)

    elif step == "SEARCH_QUERY":
        await handle_search_query(phone, session, catalog, text)

    elif step == "CUSTOM_ITEM":
        await handle_custom_item(phone, session, text)

    elif step == "SELECT_PRODUCT":
        await handle_select_product(phone, session, catalog, reply_id)

    elif step == "ITEM_ACTION":
        await handle_item_action(phone, session, catalog, reply_id)

    elif step in ("SELECT_DATE", "ENTER_DATE"):
        await handle_select_date(phone, session, reply_id, text)

    elif step == "SELECT_WINDOW":
        await handle_select_window(phone, session, reply_id)

    elif step == "ENTER_RECIPIENT":
        await handle_enter_recipient(phone, session, text)

    elif step == "ENTER_CARD":
        await handle_enter_card(phone, session, text)

    elif step == "SELECT_PAYMENT":
        await handle_select_payment(phone, session, reply_id)

    elif step == "CONFIRM":
        await handle_confirm(phone, session, reply_id)

    else:
        log.error("Unknown session step: %s", step)
        await send_text(phone, "Something went wrong. Type *cancel* to reset.")


# ===========================================================================
# SECTION 6 — FastAPI application & routes
# ===========================================================================

app = FastAPI(title="Inflora WhatsApp Order Bot")


@app.get("/api/health")
async def health(check: str = "") -> JSONResponse:
    """
    Health check endpoint.
    GET /api/health             → basic liveness (no external calls)
    GET /api/health?check=redis   → also pings Upstash Redis
    GET /api/health?check=shopify → also verifies Shopify token fetch
    """
    result: dict = {"status": "ok", "service": "inflora-bot"}

    if check == "redis":
        try:
            pong = await get_redis().ping()
            result["redis"] = "ok" if pong else "unexpected response"
        except Exception as exc:
            result["redis"] = f"error: {exc}"
            result["status"] = "degraded"

    elif check == "shopify":
        try:
            token = await _get_shopify_token()
            result["shopify"] = "ok" if token else "empty token"
        except Exception as exc:
            result["shopify"] = f"error: {exc}"
            result["status"] = "degraded"

    return JSONResponse(result)


@app.get("/api/webhook")
async def verify_webhook(request: Request) -> Response:
    """
    Meta webhook verification handshake.
    GET /api/webhook?hub.mode=subscribe&hub.verify_token=X&hub.challenge=Y
    If the verify token matches, respond with the challenge string.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WA_VERIFY_TOKEN:
        log.info("Webhook verified successfully.")
        return PlainTextResponse(challenge)

    log.warning("Webhook verification failed (bad token or mode).")
    return Response(status_code=403)


@app.post("/api/webhook")
async def receive_webhook(request: Request) -> JSONResponse:
    """
    Receive incoming WhatsApp Cloud API events.

    Always returns HTTP 200 so Meta does not retry the delivery.
    All errors are caught, logged, and (when possible) reported to the
    staff member via a WhatsApp text message.
    """
    phone = ""  # keep in scope for the except block
    try:
        body = await request.json()

        # Navigate the standard Meta webhook envelope
        # body → entry[0] → changes[0] → value → messages[0]
        entry = body.get("entry", [])
        if not entry:
            return JSONResponse({"status": "ok"})

        changes = entry[0].get("changes", [])
        if not changes:
            return JSONResponse({"status": "ok"})

        value = changes[0].get("value", {})

        # Ignore delivery/read status updates — only process incoming messages
        if "statuses" in value and "messages" not in value:
            return JSONResponse({"status": "ok"})

        messages = value.get("messages", [])
        if not messages:
            return JSONResponse({"status": "ok"})

        msg = messages[0]
        phone = msg.get("from", "")
        msg_type_raw = msg.get("type", "")  # "text", "interactive"

        # ── Normalise message type and extract text / reply_id ────────────
        text = ""
        reply_id = ""

        if msg_type_raw == "text":
            text = msg.get("text", {}).get("body", "")
            msg_type = "text"

        elif msg_type_raw == "interactive":
            interactive = msg.get("interactive", {})
            itype = interactive.get("type")  # "button_reply" or "list_reply"

            if itype == "button_reply":
                reply_id = interactive["button_reply"]["id"]
                text = interactive["button_reply"]["title"]
                msg_type = "button_reply"

            elif itype == "list_reply":
                reply_id = interactive["list_reply"]["id"]
                text = interactive["list_reply"]["title"]
                msg_type = "list_reply"

            else:
                return JSONResponse({"status": "ok"})  # unknown interactive type

        else:
            # Unsupported message type (image, audio, etc.) — ignore silently
            return JSONResponse({"status": "ok"})

        # ── Dispatch ──────────────────────────────────────────────────────
        await process_message(phone, msg_type, text, reply_id)

    except Exception as exc:
        log.exception("Unhandled error processing webhook")
        # Attempt to notify the staff member so they know something went wrong.
        # This best-effort send must not raise (we're already in the except block).
        if phone:
            try:
                await send_text(
                    phone,
                    "Something went wrong on our end. Please type *cancel* and try again.",
                )
            except Exception:
                pass  # noqa: S110

    # Always return 200 so Meta does not retry the webhook delivery.
    return JSONResponse({"status": "ok"})


# Vercel's FastAPI integration (declared in pyproject.toml) manages the ASGI
# bridge directly — no Mangum adapter needed.
