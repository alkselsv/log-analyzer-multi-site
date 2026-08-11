"""URL → page-type tokens for session embedding."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

# More specific patterns first.
DEFAULT_URL_PATTERNS = [
    (re.compile(r"/personal/order/make/"), "order_make"),
    (re.compile(r"/personal/cart/"), "cart"),
    (re.compile(r"^/personal/orders/"), "personal_orders"),
    (re.compile(r"^/personal/"), "personal"),
    (re.compile(r"^/search"), "search"),
    (re.compile(r"^/store"), "store"),
    (re.compile(r"^/here_to_help/"), "here_to_help"),
    (re.compile(r"^/culinary-world/"), "culinary_world"),
    (re.compile(r"^/auth"), "auth"),
    (re.compile(r"/filter/"), "catalog_filter"),
    (re.compile(r"^/catalog/[^/]+/[^/]+/[^/]+/[^/]+/?"), "catalog_4_levels"),
    (re.compile(r"^/catalog/[^/]+/[^/]+/[^/]+/?"), "catalog_3_levels"),
    (re.compile(r"^/catalog/[^/]+/[^/]+/?"), "catalog_2_levels"),
    (re.compile(r"^/catalog/[^/]+/?"), "catalog_1_levels"),
    (re.compile(r"^/catalog/?$"), "catalog"),
    (re.compile(r"^/$"), "home"),
]


def url_to_type(
    url: str,
    patterns: Sequence[tuple[re.Pattern[str], str]] = DEFAULT_URL_PATTERNS,
) -> Optional[str]:
    path = (url or "").split("?", 1)[0]
    for pattern, type_name in patterns:
        if pattern.search(path):
            return type_name
    return None


def uris_to_tokens(
    uris: Iterable[str],
    patterns: Sequence[tuple[re.Pattern[str], str]] = DEFAULT_URL_PATTERNS,
) -> List[str]:
    tokens: List[str] = []
    for uri in uris:
        type_name = url_to_type(uri, patterns)
        if type_name:
            tokens.append(type_name)
    return tokens


def build_token_features(
    session_uris: Mapping[str, Sequence[str]],
    session_ids: Optional[Iterable[str]] = None,
    patterns: Sequence[tuple[re.Pattern[str], str]] = DEFAULT_URL_PATTERNS,
) -> Dict[str, List[str]]:
    ids = list(session_ids) if session_ids is not None else list(session_uris.keys())
    return {
        str(session_id): uris_to_tokens(session_uris.get(str(session_id), []), patterns)
        for session_id in ids
    }
