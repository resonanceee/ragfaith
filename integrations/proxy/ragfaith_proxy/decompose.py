"""Claim decomposition, adapted from rag_faithfulness_eval/decompose.py.

spaCy sentencizer is the benchmarked path (pip install ragfaith-proxy[spacy]).
Without spaCy we degrade to a regex sentence splitter and warn loudly.

Markdown replies are structurally pre-segmented before sentence splitting
(issue #76): furniture (tables, footnotes, headings, code blocks, reference
link definitions) is dropped, list items / blockquote lines become atomic
claim candidates, meta-confidence lines are skipped, and no claim exceeds
MAX_CLAIM_CHARS (longer ones are re-split at clause boundaries, then
dropped) so mega-claims never reach the judge. Claim offsets are verbatim
char-offsets into the joined candidate text.
"""

import logging
import re
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_nlp = None

_SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")

# Longest claim handed to the judge (nudge display truncates at 200; longer
# blobs degrade judge JSON compliance and make nudges un-actionable).
MAX_CLAIM_CHARS = 300

# heading lines (## ... ) — labels, not assertions
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")

# table rows/fragments, footnotes, reference-style link definitions
_FURNITURE_RE = re.compile(r"^\s*(?:\||\[\^|\^\S|\[[^\]\n]*\]:\s*<?https?://)")

# thematic breaks: ---  ***  ___  :---  (must be the whole line)
_HR_RE = re.compile(r"^\s*:?[*_-]{3,}:?\s*$")

# meta-confidence lines: "Overall confidence: High." / "tl;dr: ..." /
# "spoiler: ..." — never assertions. Requires : or - right after the label
# so "Confidence interval was 95%" survives.
_META_RE = re.compile(
    r"^\s*(?:tl\s*;\s*dr|spoiler(?:\s+alert)?|(?:overall\s+)?confidence|certainty)\s*[:\-]",
    re.IGNORECASE,
)

# whole-line bold/italic labels: "**Who she is**" / "*Summary*" — sub-headers,
# not claims (content after the label keeps the line as a claim candidate)
_LABEL_RE = re.compile(r"^\s*\*{1,3}[^*\n]{1,60}\*{1,3}\s*[:.]?\s*$")

# list bullets / numbered markers: content becomes its own atomic candidate
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")

# blockquote markers
_QUOTE_RE = re.compile(r"^\s*>\s?")

# clause boundaries for over-long claims: after ; : — –
_CLAUSE_RE = re.compile(r"(?<=[;:\u2014\u2013])\s+")


def _claim_segments(text: str) -> list[str]:
    """Markdown-structural pre-pass (issue #76): one candidate per source line
    so whole blocks never fuse into mega-claims."""
    segs: list[str] = []
    fence = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue  # blank line = hard boundary; candidates never fuse
        if fence:
            if s.startswith(("```", "~~~")):
                fence = False
            continue  # code lines are not claims
        if s.startswith(("```", "~~~")):
            fence = True
            continue
        if _HR_RE.match(s) or _HEADING_RE.match(s) or _FURNITURE_RE.match(s):
            continue
        if _META_RE.match(s) or _LABEL_RE.match(s):
            continue
        s = _LIST_RE.sub("", s, count=1)
        s = _QUOTE_RE.sub("", s, count=1)
        if s:
            segs.append(s)
    return segs


def _claim_texts(sentence: str) -> list[str]:
    """Enforce MAX_CLAIM_CHARS: re-split at clause boundaries, drop remnants
    that are still over-long (never pass a blob to the judge)."""
    if len(sentence) <= MAX_CLAIM_CHARS:
        return [sentence]
    out = [p.strip() for p in _CLAUSE_RE.split(sentence) if p.strip()]
    return [p for p in out if len(p) <= MAX_CLAIM_CHARS]


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
    """Return [(start, end, claim)] with char offsets into the joined
    candidate text. Markdown furniture/labels/headings/code are never claims;
    list items and blockquotes are atomic; no claim exceeds MAX_CLAIM_CHARS
    (issue #76)."""
    nlp = _get_nlp()
    claims: list[tuple[int, int, str]] = []
    base = 0
    for seg in _claim_segments(text):
        for s in nlp(seg).sents:
            t = s.text.strip()
            if not t:
                continue
            for c in _claim_texts(t):
                claims.append((base + s.start_char, base + s.start_char + len(t), c))
        base += len(seg) + 1
    return claims
