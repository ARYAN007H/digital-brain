"""
ChromaDB vector store for semantic search across neurons.

One collection per cortex region (amygdala is metadata-only).
Embeddings generated externally (by Ollama) and stored here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from brain.config import Brain, Paths

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB wrapper for neuron embeddings."""

    def __init__(self, persist_dir: Optional[Path] = None):
        import chromadb

        self._persist_dir = str(persist_dir or Paths.CHROMADB)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collections: dict = {}
        self._init_collections()

    def _init_collections(self):
        """Create or get all cortex region collections."""
        for region, name in Brain.CHROMA_COLLECTIONS.items():
            self._collections[region] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        logger.info(f"ChromaDB initialized: {len(self._collections)} collections")

    def _get_collection(self, region: str):
        """Get collection by region name."""
        if region not in self._collections:
            raise ValueError(
                f"No collection for region '{region}'. "
                f"Valid: {list(self._collections.keys())}"
            )
        return self._collections[region]

    def store_neuron(
        self,
        neuron_id: str,
        region: str,
        embedding: list[float],
        document: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Upsert a neuron embedding into the correct collection.

        Returns the chroma_id (same as neuron_id).
        """
        collection = self._get_collection(region)
        meta = metadata or {}
        meta["neuron_id"] = neuron_id
        meta["region"] = region

        collection.upsert(
            ids=[neuron_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[meta],
        )
        logger.info(f"Stored vector for {neuron_id} in {region}")
        return neuron_id

    def search(
        self,
        query_embedding: list[float],
        region: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Semantic search across one or all collections.

        Returns list of {id, document, metadata, distance}.
        """
        results = []

        if region:
            collections = [(region, self._get_collection(region))]
        else:
            collections = list(self._collections.items())

        for reg, coll in collections:
            if coll.count() == 0:
                continue

            res = coll.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, coll.count()),
                include=["documents", "metadatas", "distances"],
            )

            for i in range(len(res["ids"][0])):
                results.append({
                    "id": res["ids"][0][i],
                    "document": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "distance": res["distances"][0][i],
                    "region": reg,
                })

        # Sort by distance (lower = more similar) and return top_k
        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

    def get_similar(
        self,
        neuron_id: str,
        region: str,
        threshold: float = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Find neurons semantically similar to a given neuron.

        Used for synapse candidate detection.
        """
        threshold = threshold or Brain.SIMILARITY_THRESHOLD
        collection = self._get_collection(region)

        # Get the neuron's embedding
        try:
            result = collection.get(
                ids=[neuron_id],
                include=["embeddings"],
            )
            if not result["embeddings"]:
                return []
            embedding = result["embeddings"][0]
        except Exception:
            return []

        # Search for similar (exclude self)
        results = self.search(embedding, region=region, top_k=top_k + 1)
        similar = [
            r for r in results
            if r["id"] != neuron_id and (1 - r["distance"]) >= threshold
        ]
        return similar[:top_k]

    def delete_neuron(self, neuron_id: str, region: str):
        """Remove a neuron's embedding."""
        collection = self._get_collection(region)
        collection.delete(ids=[neuron_id])
        logger.info(f"Deleted vector for {neuron_id}")

    def count(self, region: Optional[str] = None) -> dict[str, int]:
        """Count vectors per collection."""
        if region:
            return {region: self._get_collection(region).count()}
        return {r: c.count() for r, c in self._collections.items()}
