"""Chunking and metadata classification pipeline for ingesting the committed GitBook docs into ChromaDB.

Reads the scraped Markdown documents from `./data/gitbook_docs/`, splits on `##`, `###`,
and `####` headers, and tags every chunk with:
- `phase`: "entry", "mid", "exit", or "general"
- `symptom`: "understeer", "oversteer", "tire_wear", "bottoming", or "general"
- `component`: "arb", "springs", "dampers_low", "dampers_fast", "diff",
               "camber_toe", "brake_bias", "aero_rake"
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

VALID_PHASES = ("entry", "mid", "exit", "general")
VALID_SYMPTOMS = ("understeer", "oversteer", "tire_wear", "bottoming", "general")
VALID_COMPONENTS = (
    "arb",
    "springs",
    "dampers_low",
    "dampers_fast",
    "diff",
    "camber_toe",
    "brake_bias",
    "aero_rake",
)

PHASE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "entry": (
        "entry",
        "turn-in",
        "turn in",
        "braking",
        "trail-braking",
        "trail braking",
        "coast",
        "deceleration",
        "off power",
    ),
    "mid": ("mid-corner", "mid corner", "middle", "apex", "steady-state", "steady state", "center of the corner"),
    "exit": ("exit", "throttle", "on power", "acceleration", "traction", "track-out", "wheelspin", "drive ramp"),
}

SYMPTOM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "understeer": ("understeer", "push", "tight", "plowing", "front wash"),
    "oversteer": ("oversteer", "loose", "snap", "spin", "rear instability", "rotation"),
    "tire_wear": ("tire wear", "tyre wear", "wear", "degradation", "shoulder", "scrubbing", "blistering", "temperature"),
    "bottoming": ("bottoming", "bottom out", "splitter", "porpoising", "grounding"),
}

COMPONENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "arb": ("anti-roll bar", "arb", "swaybar", "sway bar", "roll bar"),
    "springs": ("spring", "heave", "third spring", "coil", "torsion bar"),
    "dampers_low": (
        "low-speed",
        "low speed",
        "shock",
        "dampening",
        "damping",
        "compression",
        "rebound",
        "pitch",
        "roll rate",
    ),
    "dampers_fast": ("high-speed", "high speed", "fast bump", "fast rebound", "fast damper", "curb", "kerb", "bump"),
    "diff": ("differential", "diff", "preload", "coast ramp", "cost ramp", "drive ramp", "power ramp", "clutch"),
    "camber_toe": ("camber", "toe", "toe-in", "toe-out", "caster", "alignment"),
    "brake_bias": ("brake bias", "brake balance", "brake cylinder", "master cylinder", "bias"),
    "aero_rake": ("aero", "rake", "ride height", "wing", "splitter", "downforce", "center of pressure"),
}


def _score_keywords(text: str, mapping: dict[str, tuple[str, ...]], default: str) -> str:
    lowered = text.lower()
    best_label = default
    best_score = 0
    for label, keywords in mapping.items():
        score = sum(lowered.count(kw) for kw in keywords)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def classify_chunk_metadata(header: str, body: str, source_file: str = "") -> dict[str, str]:
    """Classify a markdown chunk into phase, symptom, and component metadata."""
    section_only = header.split(" - ", 1)[-1] if " - " in header else header
    weighted_text = f"{section_only} {section_only} {section_only} {header} {body} {source_file}"
    phase = _score_keywords(weighted_text, PHASE_KEYWORDS, default="general")
    symptom = _score_keywords(weighted_text, SYMPTOM_KEYWORDS, default="general")
    component = _score_keywords(weighted_text, COMPONENT_KEYWORDS, default="springs")
    return {
        "phase": phase,
        "symptom": symptom,
        "component": component,
        "header": header.strip(),
        "source": source_file,
    }


def chunk_markdown(markdown_text: str, source_file: str = "doc.md") -> list[dict[str, Any]]:
    """Split a Markdown document on `##`, `###`, and `####` headers and attach structured metadata."""
    lines = markdown_text.splitlines()
    doc_title = source_file
    raw_sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_lines: list[str] = []

    header_pattern = re.compile(r"^(#{2,4})\s+(.+)$")
    h1_pattern = re.compile(r"^#\s+(.+)$")

    for line in lines:
        h1_match = h1_pattern.match(line.strip())
        if h1_match:
            doc_title = h1_match.group(1).strip()
            continue

        match = header_pattern.match(line.strip())
        if match:
            if current_lines and "".join(current_lines).strip():
                raw_sections.append((current_header or doc_title, current_lines))
            current_header = f"{doc_title} - {match.group(2).strip()}"
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines and "".join(current_lines).strip():
        raw_sections.append((current_header or doc_title, current_lines))

    chunks: list[dict[str, Any]] = []
    for idx, (header, body_lines) in enumerate(raw_sections):
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        full_text = f"{header}\n\n{body}"
        metadata = classify_chunk_metadata(header, body, source_file=source_file)
        chunk_id = hashlib.sha1(f"{source_file}:{idx}:{header}".encode("utf-8")).hexdigest()[:16]
        chunks.append(
            {
                "id": chunk_id,
                "text": full_text,
                "metadata": metadata,
            }
        )

    return chunks


def load_and_chunk_directory(docs_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load all pre-scraped GitBook markdown files from `docs_dir` and chunk them."""
    target_dir = docs_dir or settings.gitbook_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"GitBook documentation directory not found: {target_dir}")

    all_chunks: list[dict[str, Any]] = []
    for md_file in sorted(target_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown(content, source_file=md_file.name))
    return all_chunks


def ingest_gitbook_to_vector_store(docs_dir: Path | None = None) -> int:
    """Ingest all pre-scraped GitBook markdown chunks into the persistent vector store."""
    from rag.vector_store import get_vector_store

    target_dir = docs_dir or settings.gitbook_dir
    chunks = load_and_chunk_directory(target_dir)
    store = get_vector_store()
    store.upsert_chunks(chunks)
    logger.info("Ingested %d chunks into vector store.", len(chunks))
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = ingest_gitbook_to_vector_store()
    print(f"Successfully ingested {count} GitBook setup chunks into {settings.chroma_path}")
