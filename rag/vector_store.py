"""Persistent ChromaDB vector store and query interface for iRacing setup RAG.

Stores embeddings in persistent ChromaDB using `all-MiniLM-L6-v2` and provides
`query_setup_advice(query: str, phase: str = None, symptom: str = None, top_k: int = 3) -> list[dict]`.

Includes a deterministic NumPy TF-IDF / n-gram cosine similarity fallback when
offline or when ChromaDB / HuggingFace model weights are unavailable, ensuring
consistent behavior across both full production and isolated test environments.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from config import settings
from scraper.ingest_rag import load_and_chunk_directory

logger = logging.getLogger(__name__)

COLLECTION_NAME = "iracing_setup_guide"


def _deterministic_hash_embedding(text: str, dim: int = 384) -> list[float]:
    """Compute a deterministic 384-d unit vector from word and character n-grams.

    Matches the 384-d dimensionality of `all-MiniLM-L6-v2` so ChromaDB or local
    stores can operate offline without downloading weights from HuggingFace Hub.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        vec[0] = 1.0
        return vec.tolist()

    # Unigrams + bigrams for semantic phrase capture
    features = list(tokens) + [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    for feat in features:
        h = hash(feat)
        idx = abs(h) % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


class SetupVectorStore:
    """Wrapper around persistent ChromaDB with automatic fallback to local JSON+NumPy."""

    def __init__(self, db_path: Path | None = None, model_name: str | None = None) -> None:
        self.db_path = db_path or settings.chroma_path
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._collection: Any = None
        self._encoder: Any = None
        self._local_chunks: list[dict[str, Any]] = []
        self._local_index_file = self.db_path / "fallback_chunks.json"
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.db_path))
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.debug("ChromaDB unavailable (%s); using local NumPy vector store.", exc)
            self._collection = None
            if self._local_index_file.exists():
                try:
                    self._local_chunks = json.loads(self._local_index_file.read_text(encoding="utf-8"))
                except Exception:
                    self._local_chunks = []

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is None:
            try:
                from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

                onnx_ef = ONNXMiniLM_L6_V2()
                if onnx_ef.DOWNLOAD_PATH.exists():
                    self._encoder = ("onnx", onnx_ef)
                else:
                    raise FileNotFoundError("ONNX model cache not present locally")
            except Exception:
                try:
                    from sentence_transformers import SentenceTransformer

                    self._encoder = ("st", SentenceTransformer(self.model_name, local_files_only=True))
                except Exception:
                    self._encoder = False  # Sentinel indicating offline deterministic embedding

        if isinstance(self._encoder, tuple):
            kind, model = self._encoder
            if kind == "onnx":
                return [[float(x) for x in vec] for vec in model(texts)]
            embeddings = model.encode(texts, normalize_embeddings=True)
            return [emb.tolist() for emb in embeddings]

        return [_deterministic_hash_embedding(t) for t in texts]

    def count(self) -> int:
        if self._collection is not None:
            return int(self._collection.count())
        return len(self._local_chunks)

    def ensure_populated(self) -> None:
        if self.count() == 0:
            chunks = load_and_chunk_directory()
            self.upsert_chunks(chunks)

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self._encode(documents)

        if self._collection is not None:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            by_id = {c["id"]: c for c in self._local_chunks}
            for chunk, emb in zip(chunks, embeddings):
                entry = dict(chunk)
                entry["embedding"] = emb
                by_id[chunk["id"]] = entry
            self._local_chunks = list(by_id.values())
            self._local_index_file.write_text(
                json.dumps(self._local_chunks, indent=2),
                encoding="utf-8",
            )

    def query(
        self,
        query_text: str,
        phase: str | None = None,
        symptom: str | None = None,
        component: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        self.ensure_populated()
        query_emb = np.array(self._encode([query_text])[0], dtype=np.float32)

        if self._collection is not None:
            where_clauses: list[dict[str, Any]] = []
            if phase:
                where_clauses.append({"phase": {"$in": [phase, "general"]}})
            if symptom:
                where_clauses.append({"symptom": {"$in": [symptom, "general"]}})
            if component:
                where_clauses.append({"component": component})

            where_filter: dict[str, Any] | None = None
            if len(where_clauses) == 1:
                where_filter = where_clauses[0]
            elif len(where_clauses) > 1:
                where_filter = {"$and": where_clauses}

            res = self._collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=max(1, top_k),
                where=where_filter,
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]

            results: list[dict[str, Any]] = []
            for doc, meta, dist in zip(docs, metas, dists):
                results.append(
                    {
                        "text": doc,
                        "phase": meta.get("phase", "general"),
                        "symptom": meta.get("symptom", "general"),
                        "component": meta.get("component", "general"),
                        "header": meta.get("header", ""),
                        "source": meta.get("source", ""),
                        "score": round(float(1.0 - dist), 4),
                    }
                )
            if results or not where_filter:
                return results
            # Fallback to unfiltered query if strict filter matched 0 chunks
            return self.query(query_text, phase=None, symptom=None, component=None, top_k=top_k)

        # Local NumPy cosine similarity search with metadata boosting/filtering
        candidates = list(self._local_chunks)
        filtered = [
            c
            for c in candidates
            if (not phase or c["metadata"].get("phase") in (phase, "general"))
            and (not symptom or c["metadata"].get("symptom") in (symptom, "general"))
            and (not component or c["metadata"].get("component") == component)
        ]
        pool = filtered if filtered else candidates

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in pool:
            emb = np.array(item.get("embedding") or self._encode([item["text"]])[0], dtype=np.float32)
            sim = float(np.dot(query_emb, emb))
            meta = item["metadata"]
            # Boost exact phase/symptom/component match so most relevant advice ranks first
            if phase and meta.get("phase") == phase:
                sim += 0.25
            if symptom and meta.get("symptom") == symptom:
                sim += 0.25
            if component and meta.get("component") == component:
                sim += 0.30
            scored.append((sim, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        output: list[dict[str, Any]] = []
        for sim, item in scored[: max(1, top_k)]:
            meta = item["metadata"]
            output.append(
                {
                    "text": item["text"],
                    "phase": meta.get("phase", "general"),
                    "symptom": meta.get("symptom", "general"),
                    "component": meta.get("component", "general"),
                    "header": meta.get("header", ""),
                    "source": meta.get("source", ""),
                    "score": round(sim, 4),
                }
            )
        return output


_STORE_INSTANCE: SetupVectorStore | None = None


def get_vector_store() -> SetupVectorStore:
    """Return singleton SetupVectorStore instance."""
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        _STORE_INSTANCE = SetupVectorStore()
    return _STORE_INSTANCE


def query_setup_advice(
    query: str,
    phase: str | None = None,
    symptom: str | None = None,
    top_k: int = 3,
    component: str | None = None,
) -> list[dict[str, Any]]:
    """Query the iRacing setup vector store for relevant setup adjustments.

    Args:
        query: Natural language query describing the handling balance issue or corner phase.
        phase: Optional corner phase filter ("entry", "mid", "exit", or "general").
        symptom: Optional handling symptom filter ("understeer", "oversteer", "tire_wear", "bottoming", or "general").
        top_k: Number of top matching setup chunks to return.
        component: Optional component filter ("arb", "springs", "dampers_low", "dampers_fast",
                   "diff", "camber_toe", "brake_bias", "aero_rake").

    Returns:
        List of dictionaries containing chunk `text`, `phase`, `symptom`, `component`, `header`, and `score`.
    """
    store = get_vector_store()
    normalized_phase = phase.lower().strip() if phase else None
    normalized_symptom = symptom.lower().strip() if symptom else None
    normalized_component = component.lower().strip() if component else None
    return store.query(
        query_text=query,
        phase=normalized_phase,
        symptom=normalized_symptom,
        component=normalized_component,
        top_k=top_k,
    )
