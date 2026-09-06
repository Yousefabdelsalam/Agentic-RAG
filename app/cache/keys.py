"""Query normalisation and deterministic cache keys.

Normalisation decides what counts as "the same question". It is deliberately
conservative: differences a reader would call typographical are erased, and
everything else is kept. `What is 2+2?` and `what is 2 + 2` must not collide,
because they are not the same question to a calculator, so punctuation is
mapped to a canonical form rather than deleted.

Keys are pure functions of the normalised query and the versions in force, so
the same question under the same conditions always lands on the same key — in
this process, in another replica, and after a restart.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from app.cache.base import CacheVersions

KEY_PREFIX = "rag:answer"
INDEX_PREFIX = "rag:semantic"
KEY_VERSION = "v1"

#: Typographic variants that carry no meaning, mapped to their ASCII form.
#: Written as escapes because these characters are, by definition, the ones that
#: look like the characters they are not. NFKC runs first and already folds the
#: compatibility forms — non-breaking space, ellipsis — so only the quotes and
#: dashes it leaves alone need listing.
_PUNCTUATION_VARIANTS = str.maketrans(
    {
        "‘": "'",  # left single quotation mark
        "’": "'",  # right single quotation mark
        "‚": "'",  # single low-9 quotation mark
        "“": '"',  # left double quotation mark
        "”": '"',  # right double quotation mark
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus sign
    }
)

#: Sentence-final punctuation, which never changes what was asked.
_TRAILING_PUNCTUATION = "?!.,;:"

_WHITESPACE = re.compile(r"\s+")
_REPEATED_PUNCTUATION = re.compile(r"([?!.,;:])\1+")


def normalize_query(query: str) -> str:
    """Reduce a question to the form the cache keys on.

    Applied in order: compatibility-normalise unicode, fold typographic
    punctuation to ASCII, lowercase, collapse whitespace runs to one space,
    collapse repeated punctuation, and drop trailing sentence punctuation.
    """
    text = unicodedata.normalize("NFKC", query)
    text = text.translate(_PUNCTUATION_VARIANTS)
    text = text.casefold()
    text = _WHITESPACE.sub(" ", text).strip()
    text = _REPEATED_PUNCTUATION.sub(r"\1", text)
    return text.rstrip(_TRAILING_PUNCTUATION).strip()


def query_digest(normalized: str) -> str:
    """Hash a normalised query into a fixed-width, filesystem-safe token."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def answer_key(namespace: str, versions: CacheVersions, normalized: str) -> str:
    """Build the exact-cache key for a normalised query.

    The version fingerprint sits in the key, so an entry written under a
    superseded corpus, model, or prompt is unreachable rather than merely
    rejected after being read.
    """
    return ":".join(
        (KEY_PREFIX, KEY_VERSION, namespace, versions.fingerprint, query_digest(normalized))
    )


def index_namespace(namespace: str, versions: CacheVersions) -> str:
    """Build the semantic index namespace for the versions currently in force."""
    return ":".join((INDEX_PREFIX, KEY_VERSION, namespace, versions.fingerprint))


def knowledge_base_key(namespace: str) -> str:
    """Build the key holding the knowledge base version counter."""
    return ":".join((KEY_PREFIX, KEY_VERSION, namespace, "kb-version"))
