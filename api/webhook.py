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

    # Validate required config before attempting a network call
    missing = [
        name for name, val in [
            ("SHOPIFY_STORE_DOMAIN", SHOPIFY_DOMAIN),
            ("SHOPIFY_CLIENT_ID", SHOPIFY_CLIENT_ID),
            ("SHOPIFY_CLIENT_SECRET", SHOPIFY_CLIENT_SECRET),
        ] if not val
    ]
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")

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

_DRAFT_ORDER_CALCULATE = """
mutation DraftOrderCalculate($input: DraftOrderInput!) {
  draftOrderCalculate(input: $input) {
    calculatedDraftOrder {
      availableShippingRates {
        handle
        title
        price { amount }
      }
    }
    userErrors { field message }
  }
}
"""

_DRAFT_ORDER_COMPLETE = """
mutation DraftOrderComplete($id: ID!, $paymentPending: Boolean) {
  draftOrderComplete(id: $id, paymentPending: $paymentPending) {
    draftOrder {
      order {
        id
        name
        statusUrl
      }
    }
    userErrors { field message }
  }
}
"""

_CUSTOMER_SEARCH = """
query CustomerSearch($query: String!) {
  customers(first: 1, query: $query) {
    edges {
      node {
        id
      }
    }
  }
}
"""

_CUSTOMER_CREATE = """
mutation CustomerCreate($input: CustomerInput!) {
  customerCreate(input: $input) {
    customer {
      id
    }
    userErrors { field message }
  }
}
"""



async def find_or_create_customer(email: str, full_name: str) -> str | None:
    """
    Look up a Shopify customer by email.
    - Found: update their name to match the template, return their GID.
    - Not found: create them and return the new GID.
    Returns None if email is blank or on any error (order continues without customerId).
    """
    if not email:
        return None
    try:
        name_parts = full_name.strip().split(" ", 1)
        first_name = name_parts[0]
        last_name  = name_parts[1] if len(name_parts) > 1 else ""

        data = await _shopify_gql(_CUSTOMER_SEARCH, {"query": f"email:{email}"})
        if data.get("errors") or data.get("data") is None:
            log.warning("customerSearch GQL error: %s", data.get("errors"))
            return None
        edges = data["data"]["customers"]["edges"]
        if edges:
            return edges[0]["node"]["id"]

        # Not found — create
        customer_input: dict[str, Any] = {"email": email, "firstName": first_name}
        if last_name:
            customer_input["lastName"] = last_name

        result = await _shopify_gql(_CUSTOMER_CREATE, {"input": customer_input})
        payload = result["data"]["customerCreate"]
        if payload["userErrors"]:
            log.warning("customerCreate errors: %s", payload["userErrors"])
            return None
        return payload["customer"]["id"]
    except Exception:
        log.exception("find_or_create_customer failed for %s", email)
        return None


def _parse_mx_address(full_address: str) -> dict:
    """
    Split a free-text Mexican address into Shopify shippingAddress fields.

    Expected format (most common):
      "Street Number, Colonia, ZIP City, State[, Mexico]"
    Example:
      "Rio Mississippi 267, Del Valle, 66220 San Pedro Garza García, N.L., Mexico"
      → address1="Rio Mississippi 267, Del Valle"
        zip="66220"  city="San Pedro Garza García"  province="N.L."

    Uses the 5-digit postal code as the anchor.  If not found, the whole
    string goes into address1 and city/province/zip are left empty.
    """
    s = full_address.strip()

    # Remove trailing country label (", Mexico" / "Mexico")
    s = re.sub(r",?\s*M[eé]xico\s*$", "", s, flags=re.IGNORECASE).strip()

    zip_match = re.search(r"\b(\d{5})\b", s)
    if not zip_match:
        return {"address1": s, "city": "", "province": "Nuevo León", "zip": ""}

    zip_code = zip_match.group(1)

    # Everything before the zip → address1 (strip trailing comma/space)
    address1 = s[: zip_match.start()].strip().rstrip(", ").strip()

    # Everything after the zip → "City, State"
    after_zip = s[zip_match.end():].strip().lstrip(", ").strip()
    parts = [p.strip() for p in after_zip.split(",") if p.strip()]

    city     = parts[0] if parts else ""
    province = parts[1] if len(parts) > 1 else "Nuevo León"

    return {"address1": address1, "city": city, "province": province, "zip": zip_code}


async def get_shipping_rate(session: dict) -> dict | None:
    """
    Call draftOrderCalculate to get the first available shipping rate for the
    session's address. Returns {"title": str, "price": str} or None on any
    error / no rates found.
    """
    try:
        full_name = session.get("recipient_name", "").strip()
        name_parts = full_name.split(" ", 1)
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

        parsed_addr = _parse_mx_address(session.get("recipient_address", ""))
        calc_input: dict[str, Any] = {
            "lineItems": line_items,
            "shippingAddress": {
                "firstName": name_parts[0],
                "lastName": name_parts[1] if len(name_parts) > 1 else "",
                "address1": parsed_addr["address1"],
                "city": parsed_addr["city"],
                "province": parsed_addr["province"],
                "zip": parsed_addr["zip"],
                "countryCode": "MX",
            },
        }
        data = await _shopify_gql(_DRAFT_ORDER_CALCULATE, {"input": calc_input})
        if data.get("errors") or data.get("data") is None:
            log.warning("draftOrderCalculate error: %s", data.get("errors"))
            return None

        rates = (
            data["data"]["draftOrderCalculate"]
            ["calculatedDraftOrder"]["availableShippingRates"]
        )
        return [
            {"title": r["title"], "price": r["price"]["amount"]}
            for r in rates
        ]
    except Exception:
        log.exception("get_shipping_rates failed")
        return []


async def create_draft_order(session: dict) -> dict:
    """
    Create a Shopify Draft Order from the session data, matching the store's
    existing order structure (shippingAddress, DATE/SLOT custom attrs, tags).

    Returns:
      {"draftOrderId": str, "name": str, "invoiceUrl": str}

    Raises ValueError if Shopify returns userErrors.
    """
    is_pickup: bool = session.get("is_pickup", False)

    # ── Line items ────────────────────────────────────────────────────────
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

    # ── Custom attributes (matching existing store keys) ──────────────────
    custom_attrs = [
        {"key": "DELIVERY_METHOD", "value": "LOCAL_DELIVERY"},
        {"key": "DATE",            "value": session.get("delivery_date", "")},
        {"key": "SLOT",            "value": ""},   # always blank; time info goes in note
        {"key": "SLOT_ID",         "value": ""},
        {"key": "STORE_LOCATION",  "value": "Tienda Calzada del Valle"},
        {"key": "LOCATION_ID",     "value": "6615bbd3154a64d9bb1e328b"},
        {"key": "PIN_CODE",        "value": ""},
    ]

    # ── Shipping address (from structured recipient fields) ───────────────
    full_name = session.get("recipient_name", "").strip()
    name_parts = full_name.split(" ", 1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    parsed_addr = _parse_mx_address(session.get("recipient_address", ""))
    shipping_address = {
        "firstName": first_name,
        "lastName": last_name,
        "phone": session.get("recipient_phone", ""),
        "address1": parsed_addr["address1"],
        "city": parsed_addr["city"],
        "province": parsed_addr["province"],
        "zip": parsed_addr["zip"],
        "countryCode": "MX",
    }

    # ── Note (card message; prepend PICKUP info when applicable) ──────────
    card_msg = session.get("card_message", "none") or "none"
    if is_pickup:
        window = session.get("delivery_window", "")
        note_parts = [f"PICKUP - {window}"]
        if card_msg.lower() != "none":
            note_parts.append(card_msg)
        note = "\n\n".join(note_parts)
    else:
        note = card_msg if card_msg.lower() != "none" else ""

    # ── Tags ─────────────────────────────────────────────────────────────
    tags = ["Pickup"] if is_pickup else ["VENTA DOMICILIO"]

    # ── Resolve Shopify customer (find or create by email) ────────────────
    customer_email = session.get("customer_email", "").strip()
    customer_name  = session.get("customer_name", "").strip()
    customer_id    = await find_or_create_customer(customer_email, customer_name)

    # ── Assemble draft input ──────────────────────────────────────────────
    delivery_date = session.get("delivery_date", "")

    draft_input: dict[str, Any] = {
        "lineItems": line_items,
        "customAttributes": custom_attrs,
        "shippingAddress": shipping_address,
        "tags": tags,
        "metafields": [
            {
                "namespace": "custom",
                "key": "fecha_de_entrega-d0ef",
                "value": delivery_date,
                "type": "date",
            }
        ],
    }
    if note:
        draft_input["note"] = note
    if customer_email:
        draft_input["email"] = customer_email
    if customer_id:
        draft_input["customerId"] = customer_id
    shipping_rate = session.get("shipping_rate")
    if shipping_rate:
        draft_input["shippingLine"] = {
            "title": shipping_rate["title"],
            "price": shipping_rate["price"],
        }

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


async def complete_draft_order(draft_order_id: str) -> dict:
    """
    Complete (pay and finalise) a Shopify draft order.

    Returns {"name": str, "statusUrl": str}.
    Raises ValueError on userErrors.
    """
    result = await _shopify_gql(_DRAFT_ORDER_COMPLETE, {"id": draft_order_id, "paymentPending": True})
    if "data" not in result:
        log.error("draftOrderComplete GQL error: %s", result)
        raise ValueError(f"Shopify GQL error: {result.get('errors', result)}")
    payload = result["data"]["draftOrderComplete"]

    if payload["userErrors"]:
        errors = "; ".join(f"{e['field']}: {e['message']}" for e in payload["userErrors"])
        raise ValueError(f"Shopify complete order error: {errors}")

    order = payload["draftOrder"]["order"]
    return {
        "name": order["name"],
        "statusUrl": order.get("statusUrl", ""),
    }


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
        "step": "SEARCH_QUERY",
        "items": [],
        "filtered_product_ids": [],
        "page": 0,
        "selected_collection_id": None,
        "delivery_date": None,
        "delivery_window": None,   # time slot label e.g. "9:00 AM - 12:00 PM" or "Store Pickup"
        "is_pickup": False,        # True when delivery_window == "Store Pickup"
        # Recipient — collected as three separate fields to build shippingAddress
        "recipient_name": None,    # e.g. "Maria Lopez"
        "recipient_phone": None,   # e.g. "8119885392"
        "recipient_address": None, # e.g. "Av. Constitución 100, Monterrey, 64000"
        "customer_name": None,
        "customer_email": None,
        "card_message": None,
        "payment_mode": None,
        "shipping_rate": None,  # {"title": str, "price": str} or None
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

async def show_search_prompt(phone: str) -> None:
    """Send the initial search prompt with escape-hatch instructions."""
    await send_text(
        phone,
        "What product are you looking for? Type a keyword to search.\n\n"
        "_Or type_ *categories* _to browse by category, or_ *custom* _to add a custom item._",
    )


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

    # Escape hatches
    if keyword == "categories":
        session["step"] = "SELECT_CATEGORY"
        await save_session(phone, session)
        await show_category_menu(phone, session, catalog)
        return

    if keyword == "custom":
        session["step"] = "CUSTOM_ITEM"
        await save_session(phone, session)
        await send_text(
            phone,
            "Enter the custom item in this format:\n"
            "*Item Name - Price*\n\nExample: Custom Bouquet - 1500",
        )
        return

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
    page_size = 8

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

    rows.append({
        "id": "PAGE_BACK",
        "title": "Volver",
        "description": "Regresar a la busqueda",
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

    if reply_id == "PAGE_BACK":
        session["step"] = "SEARCH_QUERY"
        await save_session(phone, session)
        await show_search_prompt(phone)
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
        "What would you like to do?\n"
        "_Type_ *volver* _to remove last item._"
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
        session["step"] = "SEARCH_QUERY"
        await save_session(phone, session)
        await show_search_prompt(phone)

    elif reply_id == "BTN_ITEMS_DONE":
        session["step"] = "FILL_TEMPLATE"
        await save_session(phone, session)
        await show_order_template(phone)


# ---------------------------------------------------------------------------
# Step 4 — FILL_TEMPLATE (single-message order info collection)
# ---------------------------------------------------------------------------

_ORDER_TEMPLATE = (
    "Por favor completa los siguientes datos y envíalos en un solo mensaje:\n\n"
    "• Nombre del cliente:\n"
    "• Correo del cliente:\n"
    "• Fecha de entrega (YYYY-MM-DD):\n\n"
    "• Nombre de la persona que recibe el arreglo:\n"
    "• Dirección de envío completa (calle, número, colonia, municipio y código postal):\n"
    "• Teléfono de la persona que recibe el arreglo:\n\n"
    "• Mensaje de la tarjeta:"
)

_REQUIRED_TEMPLATE_FIELDS = {
    "delivery_date":    "Fecha de entrega",
    "recipient_name":   "Nombre de la persona que recibe",
    "recipient_phone":  "Teléfono de la persona que recibe",
    "recipient_address":"Dirección de envío",
}

# Label patterns in the order they appear in the template.
# Each pattern matches the label text + anything up to (and including) an optional colon.
# The value for a field is everything between its label's end and the next label's start.
_LABEL_PATTERNS: list[tuple[str, str]] = [
    ("customer_name",    r"nombre del cliente[^:\n]*[:：]?"),
    ("customer_email",   r"correo del cliente[^:\n]*[:：]?"),
    ("delivery_date",    r"fecha de entrega[^:\n]*[:：]?"),
    ("recipient_name",   r"nombre de la persona que recibe[^:\n]*[:：]?"),
    ("recipient_address",r"direcci[oó]n de env[ií][oó][^:\n]*[:：]?"),
    ("recipient_phone",  r"tel[eé]fono de la persona que recibe[^:\n]*[:：]?"),
    ("card_message",     r"mensaje de la tarjeta[^:\n]*[:：]?"),
]


async def show_order_template(phone: str) -> None:
    """Send the order info template for the staff to fill in."""
    await send_text(phone, _ORDER_TEMPLATE)


# Invisible / zero-width Unicode characters WhatsApp embeds in messages.
_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00a0]+")
# Fields whose values may legitimately span multiple lines.
_MULTILINE_FIELDS = {"card_message"}


def _clean_field(raw: str, field: str) -> str:
    """
    Clean a raw segment extracted from the template:
      1. Remove invisible/zero-width Unicode characters.
      2. Strip surrounding whitespace and leading bullet chars.
      3. For single-line fields, return only the first non-empty line so that
         stray bullets or newlines from the template don't bleed into values.
    """
    cleaned = _INVISIBLE_RE.sub(" ", raw)
    cleaned = cleaned.strip().lstrip("• ").strip()

    if field not in _MULTILINE_FIELDS:
        for line in cleaned.splitlines():
            line = line.strip().lstrip("• ").strip()
            if line:
                return line
        return ""

    return cleaned


def _parse_template(text: str) -> dict:
    """
    Segment-based template parser.

    Locates each known label in the text, then treats everything between
    that label's end and the next label's start as the field value.
    This correctly handles:
      - values on the same line as the label
      - values on the following line (blank after the colon)
      - multi-line values (e.g. card messages spanning several paragraphs)
      - completely blank values (the field is simply omitted from the result)
    """
    # Strip invisible chars from the whole text so they don't break label matching.
    # WhatsApp embeds U+2060 and similar inside copied text, not just at value edges.
    text = _INVISIBLE_RE.sub("", text)

    anchors: list[tuple[int, int, str]] = []
    for field, pattern in _LABEL_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            anchors.append((m.start(), m.end(), field))

    if not anchors:
        return {}

    anchors.sort(key=lambda x: x[0])

    result: dict[str, str] = {}
    for i, (_, label_end, field) in enumerate(anchors):
        next_start = anchors[i + 1][0] if i + 1 < len(anchors) else len(text)
        raw = text[label_end:next_start]
        value = _clean_field(raw, field)
        if value:
            result[field] = value

    return result


_MONTH_MAP: dict[str, int] = {
    # Spanish
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    # English
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _normalize_date(raw: str) -> str:
    """
    Convert a free-text date into YYYY-MM-DD.
    Handles:
      - Already normalized:          2026-11-27
      - Numeric DD/MM/YYYY:          27/11/2026  or  27-11-2026
      - Spanish/English named month: 27 Noviembre 2026
                                     27 de noviembre de 2026
                                     November 27, 2026
    Returns the original string unchanged if no pattern matches.
    """
    s = raw.strip()

    # Already YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.fullmatch(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", s)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

    # "27 Noviembre 2026" / "27 de Noviembre de 2026"
    m = re.search(r"(\d{1,2})\s+(?:de\s+)?([a-záéíóúñ]+)\s+(?:de\s+)?(\d{4})", s, re.IGNORECASE)
    if m:
        month = _MONTH_MAP.get(m.group(2).lower())
        if month:
            return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(1)):02d}"

    # "November 27, 2026" / "November 27 2026"
    m = re.search(r"([a-záéíóúñ]+)\s+(\d{1,2})[,\s]+(\d{4})", s, re.IGNORECASE)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(2)):02d}"

    return s  # fallback: return as-is


# ---------------------------------------------------------------------------
# Delivery fee lookup by zip code
# ---------------------------------------------------------------------------

_GUADALUPE_ZIPS = frozenset({
    "64834",
    "67100","67110","67112","67113","67115","67116","67117","67118","67119",
    "67120","67123","67124","67125","67126","67128","67129","67130","67132",
    "67133","67134","67140","67144","67150","67153","67154","67155","67160",
    "67163","67164","67165","67166","67167","67168","67169","67170","67173",
    "67174","67175","67176","67177","67178","67179","67180","67182","67183",
    "67184","67185","67186","67188","67189","67190","67192","67193","67194",
    "67195","67196","67197","67198","67199",
    "67200","67202","67203","67204","67205",
})

_ESCOBEDO_APODACA_ZIPS = frozenset({
    "65780",
    "66050","66051","66052","66053","66054","66055","66056","66057","66058",
    "66059","66060","66062","66063","66064","66065","66066","66070","66072",
    "66073","66074","66080","66083","66084","66085",
    "66600","66603","66604","66605","66606","66609","66610","66612","66613",
    "66614","66620","66623","66626","66627","66630","66632","66633","66634",
    "66635","66636","66637","66640","66643","66644","66645","66646","66647",
    "66648","66649",
})

_SAN_NICOLAS_ZIPS = frozenset({
    "66400","66410","66412","66413","66414","66415","66417","66418","66420",
    "66422","66423","66424","66425","66426","66427","66428","66430","66434",
    "66435","66436","66437","66438","66440","66444","66445","66446","66447",
    "66448","66449","66450","66454","66455","66457","66458","66459","66460",
    "66463","66464","66465","66466","66467","66468","66469","66470","66473",
    "66474","66475","66477","66478","66480","66482","66483","66484","66485",
    "66486","66490","66493","66494","66496",
})

_GARCIA_ZIPS = frozenset({
    "66000","66001","66003","66004","66005","66006","66007","66008","66009",
    "66010","66012","66013","66014","66017","66018","66020","66023","66024",
    "66025","66026","66027","66030","66033","66035","66036","66037","66040",
    "66042","66045","66046",
})

_CIENAGA_ZIPS = frozenset({
    "65550","65554","65555","65556","65558","65559","65560","65562","65563",
    "65564","65565","65567","65573","65580","65583",
    "67450","67451","67453","67455","67460","67463","67465","67470","67473",
    "67474","67475","67480","67483","67484","67485","67486","67488","67490",
    "67492","67497",
})


_ZONE_FEES: dict[str, int] = {
    "san pedro": 99, "san pedro garza garcia": 99, "garza garcia": 99,
    "monterrey": 99,
    "santa catarina": 99,
    "guadalupe": 149,
    "escobedo": 149, "general escobedo": 149,
    "apodaca": 149,
    "san nicolas": 149, "san nicolás": 149, "san nicolas de los garza": 149,
    "garcia": 299, "garcía": 299,
    "santiago": 299,
    "cienega": 349, "ciénega": 349, "cienega de flores": 349, "ciénega de flores": 349,
}

_ACCENT_MAP = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")


def _get_delivery_fee_by_zone(zone_text: str) -> int | None:
    """Return delivery fee for a free-text zone name, or None if not matched."""
    normalized = zone_text.lower().strip().translate(_ACCENT_MAP)
    for zone, fee in _ZONE_FEES.items():
        if zone.translate(_ACCENT_MAP) in normalized:
            return fee
    return None


def _get_delivery_fee(zip_code: str) -> int | None:
    """
    Return the delivery fee (MXN) for a zip code based on configured zones.
    Returns None if the zip is outside all delivery zones.
    Specific zip sets are checked before ranges to handle overlaps (e.g. 64834
    belongs to Guadalupe at $149, not Monterrey at $99).
    Monterrey Valle Alto ($129) is checked before general Monterrey ($99).
    """
    if not zip_code:
        return None

    # Specific zip sets (highest priority — override any range below)
    if zip_code in _GUADALUPE_ZIPS:       return 149
    if zip_code in _ESCOBEDO_APODACA_ZIPS: return 149
    if zip_code in _SAN_NICOLAS_ZIPS:     return 149
    if zip_code in _GARCIA_ZIPS:          return 299
    if zip_code in _CIENAGA_ZIPS:         return 349

    try:
        z = int(zip_code)
    except ValueError:
        return None

    if 64978 <= z <= 64997: return 129   # Monterrey Valle Alto — before general MTY
    if 64000 <= z <= 64997: return 99    # Monterrey
    if 66200 <= z <= 66296: return 99    # San Pedro Garza García
    if 66100 <= z <= 66390: return 99    # Santa Catarina
    if 67300 <= z <= 67344: return 299   # Santiago

    return None


async def handle_fill_template(phone: str, session: dict, text: str) -> None:
    """
    Parse the filled-in order template and advance to SELECT_PAYMENT.
    If required fields are missing, re-send the template with a note.
    """
    parsed = _parse_template(text)

    missing = [
        label for field, label in _REQUIRED_TEMPLATE_FIELDS.items()
        if not parsed.get(field)
    ]
    if missing:
        missing_str = ", ".join(missing)
        await send_text(
            phone,
            f"Faltan los siguientes datos: *{missing_str}*\n\n"
            "Por favor completa y envía el mensaje de nuevo:",
        )
        await show_order_template(phone)
        return

    session["delivery_date"]    = _normalize_date(parsed["delivery_date"])
    session["recipient_name"]   = parsed["recipient_name"]
    session["recipient_phone"]  = parsed["recipient_phone"]
    session["recipient_address"]= parsed["recipient_address"]
    session["customer_name"]    = parsed.get("customer_name", "")
    session["customer_email"]   = parsed.get("customer_email", "")
    session["card_message"]     = parsed.get("card_message", "none")
    session["delivery_window"]  = ""
    session["is_pickup"]        = False
    session["step"] = "REVIEW_TEMPLATE"
    await save_session(phone, session)
    await show_review_template(phone, session)


async def show_review_template(phone: str, session: dict) -> None:
    """Show the parsed order info for staff to verify before payment."""
    card_preview = session.get("card_message", "none") or "none"
    if len(card_preview) > 80:
        card_preview = card_preview[:80] + "…"

    customer_name  = session.get("customer_name", "")
    customer_email = session.get("customer_email", "")
    customer_line  = customer_name + (f" ({customer_email})" if customer_email else "")

    lines = ["*Verifica la información:*\n"]
    if customer_line:
        lines.append(f"Cliente: {customer_line}")
    lines.append(f"Fecha: {session.get('delivery_date', '')}")
    lines.append(f"Recibe: {session.get('recipient_name', '')}")
    lines.append(f"Tel: {session.get('recipient_phone', '')}")
    lines.append(f"Direccion: {session.get('recipient_address', '')}")
    lines.append(f"Tarjeta: {card_preview}")

    await send_buttons(
        phone,
        body="\n".join(lines),
        buttons=[
            {"id": "BTN_REVIEW_OK",   "title": "Confirmar"},
            {"id": "BTN_REVIEW_EDIT", "title": "Corregir"},
            {"id": "BTN_BACK",        "title": "Volver"},
        ],
    )


async def show_payment_buttons(phone: str) -> None:
    """Send the payment method selection buttons."""
    await send_buttons(
        phone,
        body="¿Cómo se paga este pedido?",
        buttons=[
            {"id": "PAY_LINK", "title": "Payment Link"},
            {"id": "PAY_PENDING", "title": "Pending Payment"},
            {"id": "BTN_BACK", "title": "Volver"},
        ],
    )


async def handle_review_template(phone: str, session: dict, reply_id: str) -> None:
    """Handle the review confirmation or edit request."""
    if reply_id == "BTN_REVIEW_OK":
        session["step"] = "SHIPPING_METHOD"
        await save_session(phone, session)
        await show_shipping_method(phone)
    elif reply_id == "BTN_REVIEW_EDIT":
        session["step"] = "FILL_TEMPLATE"
        await save_session(phone, session)
        await send_text(phone, "Sin problema, envía la información corregida:")
        await show_order_template(phone)


async def show_shipping_method(phone: str) -> None:
    """Ask staff whether this order is delivery or in-store pickup."""
    await send_buttons(
        phone,
        body="¿Cómo se va a entregar este pedido?",
        buttons=[
            {"id": "BTN_DELIVERY", "title": "Entrega local"},
            {"id": "BTN_PICKUP",   "title": "Recoger en tienda"},
            {"id": "BTN_BACK",     "title": "Volver"},
        ],
    )


async def handle_shipping_method(phone: str, session: dict, reply_id: str) -> None:
    """Route to delivery fee check or straight to payment for pickup."""
    if reply_id == "BTN_PICKUP":
        session["is_pickup"]       = True
        session["delivery_window"] = "Recoger en Tienda"
        session["shipping_rate"]   = None
        session["step"] = "SELECT_PAYMENT"
        await save_session(phone, session)
        await show_payment_buttons(phone)

    elif reply_id == "BTN_DELIVERY":
        session["is_pickup"] = False
        parsed = _parse_mx_address(session.get("recipient_address", ""))
        fee = _get_delivery_fee(parsed["zip"])

        if fee:
            session["step"] = "CONFIRM_SHIPPING"
            await save_session(phone, session)
            await send_buttons(
                phone,
                body=f"Cargo de envio calculado: *${fee:,}*\n\n¿Incluir en el pedido?",
                buttons=[
                    {"id": "BTN_SHIP_YES", "title": f"Si, ${fee:,}"},
                    {"id": "BTN_SHIP_NO",  "title": "No incluir"},
                    {"id": "BTN_BACK",     "title": "Volver"},
                ],
            )
        else:
            session["step"] = "ENTER_SHIPPING_COST"
            await save_session(phone, session)
            await send_text(
                phone,
                f"No encontramos el código postal *{parsed['zip'] or 'desconocido'}*.\n\n"
                "Escribe la zona de entrega (ej: *San Pedro*, *Escobedo*, *Apodaca*) "
                "o el código postal correcto:",
            )


async def handle_enter_shipping_cost(phone: str, session: dict, text: str) -> None:
    """Staff enters a zone name, zip code, or manual price to set the delivery fee."""
    raw = text.strip()

    # 1. Try zone name lookup
    fee = _get_delivery_fee_by_zone(raw)

    # 2. Try 5-digit zip code
    if fee is None:
        zip_match = re.search(r"\b(\d{5})\b", raw)
        if zip_match:
            fee = _get_delivery_fee(zip_match.group(1))

    if fee is not None:
        session["shipping_rate"] = {"title": "Envío", "price": str(fee)}
        session["step"] = "SELECT_PAYMENT"
        await save_session(phone, session)
        await send_text(phone, f"Cargo de envío calculado: *${fee:,}*")
        await show_payment_buttons(phone)
        return

    # 3. Fall back to manual numeric price
    cleaned = raw.replace(",", "").replace("$", "")
    try:
        manual_fee = float(cleaned)
    except ValueError:
        await send_text(
            phone,
            "No reconocimos la zona. Escribe la zona (ej: *San Pedro*, *Escobedo*, *Apodaca*), "
            "el código postal, o el monto directo (ej: *150* o *0* si no aplica):",
        )
        return

    if manual_fee > 0:
        session["shipping_rate"] = {"title": "Envío", "price": str(int(manual_fee))}
    else:
        session["shipping_rate"] = None

    session["step"] = "SELECT_PAYMENT"
    await save_session(phone, session)
    await show_payment_buttons(phone)


# ---------------------------------------------------------------------------
# Step 5 — SELECT_DATE / ENTER_DATE / SELECT_WINDOW  (kept for legacy sessions)
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
    """Save the chosen delivery window and move to ENTER_RECIPIENT_NAME."""
    label = _WINDOW_LABELS.get(reply_id, reply_id)
    session["delivery_window"] = label
    session["is_pickup"] = (label == "Store Pickup")
    session["step"] = "ENTER_RECIPIENT_NAME"
    await save_session(phone, session)
    await send_text(phone, "What is the recipient's *full name*?")


# ---------------------------------------------------------------------------
# Step 5a — ENTER_RECIPIENT_NAME / ENTER_RECIPIENT_PHONE / ENTER_RECIPIENT_ADDRESS
# ---------------------------------------------------------------------------

async def handle_enter_recipient_name(phone: str, session: dict, text: str) -> None:
    """Save recipient name and ask for phone."""
    session["recipient_name"] = text.strip()
    session["step"] = "ENTER_RECIPIENT_PHONE"
    await save_session(phone, session)
    await send_text(phone, "Recipient's *phone number*?")


async def handle_enter_recipient_phone(phone: str, session: dict, text: str) -> None:
    """Save recipient phone and ask for address (skip if pickup)."""
    session["recipient_phone"] = text.strip()
    if session.get("is_pickup"):
        # Pickup — no delivery address needed; go straight to card message
        session["recipient_address"] = ""
        session["step"] = "ENTER_CARD"
        await save_session(phone, session)
        await send_text(
            phone,
            "Reply with the *gift card message* (or type _none_ if no card):",
        )
    else:
        session["step"] = "ENTER_RECIPIENT_ADDRESS"
        await save_session(phone, session)
        await send_text(
            phone,
            "Delivery *address*?\n\nInclude street, city, and zip code.\n"
            "Example: Priv. Amazonas 551, Monterrey, 66220",
        )


async def handle_enter_recipient_address(phone: str, session: dict, text: str) -> None:
    """Save delivery address and move to ENTER_CARD."""
    session["recipient_address"] = text.strip()
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
            {"id": "PAY_PENDING", "title": "Pending Payment"},
        ],
    )


# ---------------------------------------------------------------------------
# Step 5c — SELECT_PAYMENT → CONFIRM
# ---------------------------------------------------------------------------

async def handle_select_payment(phone: str, session: dict, reply_id: str) -> None:
    """Save payment mode and proceed to order confirmation."""
    session["payment_mode"] = reply_id  # "PAY_LINK" or "PAY_PENDING"
    session["step"] = "CONFIRM"
    await save_session(phone, session)
    await show_order_summary(phone, session)


async def handle_confirm_shipping(phone: str, session: dict, reply_id: str) -> None:
    """Staff confirms or skips the delivery fee."""
    if reply_id == "BTN_SHIP_YES":
        parsed = _parse_mx_address(session.get("recipient_address", ""))
        fee = _get_delivery_fee(parsed["zip"])
        session["shipping_rate"] = {"title": "Envío", "price": str(fee)} if fee else None
    else:
        session["shipping_rate"] = None

    session["step"] = "SELECT_PAYMENT"
    await save_session(phone, session)
    await show_payment_buttons(phone)


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

    payment_label = "Payment Link" if session.get("payment_mode") == "PAY_LINK" else "Pending Payment"
    card_msg = session.get("card_message", "none")
    is_pickup = session.get("is_pickup", False)
    delivery_type = "Pickup" if is_pickup else "Delivery"

    def _s(key: str) -> str:
        """Get a session string value, stripped of invisible chars."""
        return _INVISIBLE_RE.sub("", session.get(key, "") or "").strip()

    customer_name  = _s("customer_name")
    customer_email = _s("customer_email")
    customer_line  = f"  {customer_name}" + (f" — {customer_email}" if customer_email else "") + "\n"

    recipient_lines = (
        f"  {_s('recipient_name')}\n"
        f"  {_s('recipient_phone')}\n"
        + (f"  {_s('recipient_address')}\n" if not is_pickup else "")
    )

    window_part = f" | {session.get('delivery_window')}" if session.get("delivery_window") else ""
    summary = (
        "*ORDER SUMMARY*\n\n"
        "*Items:*\n"
        + "\n".join(item_lines)
        + f"\n*Total: ${total:,.2f}*\n\n"
        + (f"*Customer:*\n{customer_line}" if customer_name else "")
        + f"*{delivery_type}:* {session.get('delivery_date')}{window_part}\n"
        f"*Recipient:*\n{recipient_lines}"
        f"*Card message:* {card_msg}\n"
        f"*Payment:* {payment_label}\n"
        + (f"*Envio:* ${float(session['shipping_rate']['price']):,.2f}\n"
           if session.get("shipping_rate") else "")
        + "\nConfirm and create the Shopify order?"
    )

    await send_buttons(
        phone,
        body=summary,
        buttons=[
            {"id": "CONFIRM_CREATE", "title": "Create Order"},
            {"id": "CONFIRM_CANCEL", "title": "Cancel"},
            {"id": "BTN_BACK",       "title": "Volver"},
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

    if payment_mode == "PAY_PENDING":
        # Complete the draft into a finalised paid order
        try:
            completed = await complete_draft_order(draft["draftOrderId"])
        except ValueError as exc:
            await send_text(phone, f"Shopify complete error: {exc}\n\nDraft saved as {draft['name']}.")
            await delete_session(phone)
            return

        confirmation = f"Orden *{completed['name']}* creada con pago pendiente en Shopify!"
        if completed.get("statusUrl"):
            confirmation += f"\n\n📦 *Link de seguimiento (cliente):*\n{completed['statusUrl']}"
        confirmation += "\n\nEscribe *new* para empezar otra orden."
    else:
        # PAY_LINK — share the invoice URL with the staff member
        confirmation = (
            f"Borrador *{draft['name']}* creado.\n\n"
            f"Envía este link al cliente para ver su orden y pagar:\n{draft['invoiceUrl']}\n\n"
            "Escribe *new* para empezar otra orden."
        )

    await delete_session(phone)
    await send_text(phone, confirmation)


# ===========================================================================
# SECTION 5 — Main message dispatcher
# ===========================================================================

async def handle_back(phone: str, session: dict) -> None:
    """Navigate back one step, undoing the last action where applicable."""
    step = session.get("step")

    if step == "SELECT_PRODUCT":
        session["step"] = "SEARCH_QUERY"
        await save_session(phone, session)
        await show_search_prompt(phone)

    elif step == "ITEM_ACTION":
        # Remove the last item that was just added
        if session.get("items"):
            session["items"].pop()
        if session.get("items"):
            await save_session(phone, session)
            await show_item_action(phone, session)
        else:
            session["step"] = "SEARCH_QUERY"
            await save_session(phone, session)
            await show_search_prompt(phone)

    elif step == "FILL_TEMPLATE":
        if session.get("items"):
            session["step"] = "ITEM_ACTION"
            await save_session(phone, session)
            await show_item_action(phone, session)
        else:
            session["step"] = "SEARCH_QUERY"
            await save_session(phone, session)
            await show_search_prompt(phone)

    elif step == "REVIEW_TEMPLATE":
        session["step"] = "FILL_TEMPLATE"
        await save_session(phone, session)
        await show_order_template(phone)

    elif step == "SHIPPING_METHOD":
        session["step"] = "REVIEW_TEMPLATE"
        await save_session(phone, session)
        await show_review_template(phone, session)

    elif step == "CONFIRM_SHIPPING":
        session["step"] = "SHIPPING_METHOD"
        await save_session(phone, session)
        await show_shipping_method(phone)

    elif step == "ENTER_SHIPPING_COST":
        session["step"] = "SHIPPING_METHOD"
        await save_session(phone, session)
        await show_shipping_method(phone)

    elif step == "SELECT_PAYMENT":
        session["step"] = "SHIPPING_METHOD"
        await save_session(phone, session)
        await show_shipping_method(phone)

    elif step == "CONFIRM":
        session["step"] = "SELECT_PAYMENT"
        await save_session(phone, session)
        await show_payment_buttons(phone)

    else:
        await send_text(phone, "No hay paso anterior. Escribe *cancel* para empezar de nuevo.")

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
    # ── 1. Global commands (always available, bypass state machine) ───────
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
        await show_search_prompt(phone)
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
    step = session.get("step", "SEARCH_QUERY")
    catalog = await get_catalog()  # always served from Redis cache (fast)

    # Global back — intercept before step routing
    if reply_id == "BTN_BACK" or text_lower in ("volver", "atras"):
        await handle_back(phone, session)
        return

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

    elif step == "FILL_TEMPLATE":
        await handle_fill_template(phone, session, text)

    elif step == "REVIEW_TEMPLATE":
        await handle_review_template(phone, session, reply_id)

    elif step == "SHIPPING_METHOD":
        await handle_shipping_method(phone, session, reply_id)

    elif step == "ENTER_SHIPPING_COST":
        await handle_enter_shipping_cost(phone, session, text)

    elif step in ("SELECT_DATE", "ENTER_DATE"):
        await handle_select_date(phone, session, reply_id, text)

    elif step == "SELECT_WINDOW":
        await handle_select_window(phone, session, reply_id)

    elif step == "ENTER_RECIPIENT_NAME":
        await handle_enter_recipient_name(phone, session, text)

    elif step == "ENTER_RECIPIENT_PHONE":
        await handle_enter_recipient_phone(phone, session, text)

    elif step == "ENTER_RECIPIENT_ADDRESS":
        await handle_enter_recipient_address(phone, session, text)

    elif step == "ENTER_CARD":
        await handle_enter_card(phone, session, text)

    elif step == "SELECT_PAYMENT":
        await handle_select_payment(phone, session, reply_id)

    elif step == "CONFIRM_SHIPPING":
        await handle_confirm_shipping(phone, session, reply_id)

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
    GET /api/health                   → basic liveness (no external calls)
    GET /api/health?check=redis       → also pings Upstash Redis
    GET /api/health?check=shopify     → also verifies Shopify token fetch
    GET /api/health?check=shopify_gql → fires a real GQL query (shop name + first product)
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

    elif check == "shopify_gql":
        try:
            data = await _shopify_gql(
                "{ shop { name } products(first: 1) { edges { node { title } } } }"
            )
            shop_name = data["data"]["shop"]["name"]
            edges = data["data"]["products"]["edges"]
            first_product = edges[0]["node"]["title"] if edges else "no products"
            result["shopify_gql"] = "ok"
            result["shop"] = shop_name
            result["first_product"] = first_product
        except Exception as exc:
            result["shopify_gql"] = f"error: {exc}"
            result["status"] = "degraded"

    return JSONResponse(result)


@app.get("/api/debug/order")
async def debug_order(name: str = "") -> JSONResponse:
    """
    Temporary debug endpoint — fetch a Shopify order by name and return its raw structure.
    GET /api/debug/order?name=1045   (pass the number without #)
    Remove this endpoint once the order field mapping is confirmed.
    """
    if not name:
        return JSONResponse({"error": "Pass ?name=<order_number> e.g. ?name=1045"}, status_code=400)

    query = """
    query GetOrder($query: String!) {
      orders(first: 1, query: $query) {
        edges {
          node {
            id
            name
            note
            phone
            shippingAddress {
              firstName lastName phone address1 address2 city province country zip
            }
            customAttributes { key value }
            lineItems(first: 20) {
              edges {
                node {
                  title quantity originalUnitPriceSet { shopMoney { amount } }
                  variant { id }
                }
              }
            }
            tags
            createdAt
          }
        }
      }
    }
    """
    try:
        data = await _shopify_gql(query, {"query": f"name:#{name}"})
        edges = data["data"]["orders"]["edges"]
        if not edges:
            return JSONResponse({"error": f"Order #{name} not found"}, status_code=404)
        return JSONResponse(edges[0]["node"])
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/debug/shipping")
async def debug_shipping(zip: str = "", city: str = "San Pedro Garza García") -> JSONResponse:
    """
    Debug endpoint — returns all shipping rates Shopify/Scrollengine calculates
    for a given zip code.
    GET /api/debug/shipping?zip=66220
    GET /api/debug/shipping?zip=66220&city=Monterrey
    """
    if not zip:
        return JSONResponse({"error": "Pass ?zip=<postal_code> e.g. ?zip=66220"}, status_code=400)

    calc_input: dict = {
        "lineItems": [{"title": "Test Item", "originalUnitPrice": "1000", "quantity": 1}],
        "shippingAddress": {
            "firstName": "Test",
            "lastName": "Debug",
            "address1": "Calle Test 123",
            "city": city,
            "province": "Nuevo León",
            "zip": zip,
            "countryCode": "MX",
        },
    }
    try:
        data = await _shopify_gql(_DRAFT_ORDER_CALCULATE, {"input": calc_input})
        return JSONResponse({"zip": zip, "city": city, "raw": data})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/debug/delivery")
async def debug_delivery() -> JSONResponse:
    """
    Debug endpoint — fetches all Shopify delivery profiles with zones and rates.
    GET /api/debug/delivery
    """
    query = """
    query {
      deliveryProfiles(first: 10) {
        edges {
          node {
            id
            name
            profileLocationGroups {
              locationGroupZones(first: 30) {
                edges {
                  node {
                    zone {
                      id
                      name
                      countries {
                        name
                        code { countryCode }
                        provinces { name code }
                      }
                    }
                    methodDefinitions(first: 10) {
                      edges {
                        node {
                          id
                          name
                          active
                          rateProvider {
                            __typename
                            ... on DeliveryRateDefinition {
                              price { amount currencyCode }
                            }
                          }
                          methodConditions {
                            field
                            operator
                            conditionCriteria {
                              __typename
                              ... on MoneyV2 { amount currencyCode }
                              ... on Weight { value unit }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    try:
        data = await _shopify_gql(query)
        return JSONResponse(data)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


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

    except Exception:
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
