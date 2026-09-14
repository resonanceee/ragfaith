"""Claim decomposition, copied from rag_faithfulness_eval/decompose.py.

spaCy sentencizer is the benchmarked path (pip install ragfaith-proxy[spacy]).
Without spaCy we degrade to a regex sentence splitter and warn loudly; claim
offsets stay verbatim char-offsets into the text either way.
"""

import logging
import re
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_nlp = None

_SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")

# Markdown furniture: table rows/fragments, footnotes, thematic breaks,
# reference-style link definitions. These are not atomic claims; judging them
# wastes tokens and produces false flags (issue #76).
_FURNITURE_RE = re.compile(r"^\s*(?:\||\[\^|\^\S|---|\[[^\]\n]*\]:\s*<?https?://)")


def _strip_furniture(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _FURNITURE_RE.match(line))


def _get_nlp():  # lazy: spacy is an optional extra, the proxy must not need it
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _nlp = nlp
    except Exception as e:  # noqa: BLE001 - missing extra OR broken install; degrade, never crash
        # blank("en") + sentencizer is the benchmark path and needs no model
        # download — the [spacy] extra alone is enough. If this still fails,
        # the spaCy install itself is broken (e.g. missing system libraries);
        # surface the underlying error instead of a generic hint.
        logger.warning(
            "spaCy unavailable (%s: %s): using regex sentence splitter (degraded "
            "claim boundaries). 'pip install ragfaith-proxy[spacy]' is sufficient; "
            "no model download is required.",
            type(e).__name__,
            e,
        )

        def regex_nlp(text: str):
            sents = []
            for m in _SENT_RE.finditer(text):
                seg = m.group()
                stripped = seg.strip()
                if not stripped:
                    continue
                start = m.start() + len(seg) - len(seg.lstrip())
                sents.append(
                    SimpleNamespace(start_char=start, end_char=start + len(stripped), text=stripped)
                )
            return SimpleNamespace(sents=sents)

        _nlp = regex_nlp
    return _nlp


def split_claims(text: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, sentence)] with char offsets into the
    furniture-stripped text (markdown tables/footnotes/citations are dropped
    before segmentation and never become claims; issue #76)."""
    stripped = _strip_furniture(text)
    doc = _get_nlp()(stripped)
    return [(s.start_char, s.end_char, s.text) for s in doc.sents if s.text.strip()]
