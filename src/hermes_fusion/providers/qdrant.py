"""Qdrant vector database provider for semantic caching (P8)."""

import time
import uuid
from typing import Any

from hermes_fusion.config import QdrantConfig
from hermes_fusion.providers.base import Provider


class QdrantProvider(Provider):
    """Qdrant vector database provider for semantic caching."""
    name = "qdrant"

    def __init__(self, config: QdrantConfig, client: Any = None):
        self.config = config
        self.base_url = config.url.rstrip('/')
        self.collection = config.collection
        self._client = client  # Injected for testing; in prod use qdrant-client

    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client:
            return self._client
        # Lazy import for production
        try:
            from qdrant_client import AsyncQdrantClient
            self._client = AsyncQdrantClient(url=self.base_url)
            return self._client
        except ImportError:
            raise RuntimeError("qdrant-client not installed. Install with: pip install qdrant-client")

    async def _ensure_collection(self):
        """Ensure collection exists with HNSW index."""
        client = self._get_client()
        try:
            await client.get_collection(self.collection)
        except Exception:
            await client.create_collection(
                collection_name=self.collection,
                vectors_config={"size": self.config.vector_size, "distance": "Cosine"},
                hnsw_config={"m": 16, "ef_construct": 100},
            )

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding for text via LocalAI or fallback hash."""
        client = self._get_client()
        if hasattr(client, 'embed'):
            # Use qdrant-client's built-in embedding (requires model)
            result = await client.embed(model="", documents=[text])
            return result[0]
        # Fallback: deterministic hash-based pseudo-embedding
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Expand to vector_size dimensions
        vec = []
        for i in range(0, min(len(hash_bytes), self.config.vector_size), 4):
            chunk = int.from_bytes(hash_bytes[i:i+4], 'big')
            vec.append((chunk % 10000) / 10000.0 * 2 - 1)  # Normalize to [-1, 1]
        # Pad or truncate
        if len(vec) < self.config.vector_size:
            vec.extend([0.0] * (self.config.vector_size - len(vec)))
        return vec[:self.config.vector_size]

    async def chat(self, messages: list[dict[str, str]], model: str, **kwargs):
        raise NotImplementedError("QdrantProvider.chat - not a chat provider")

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise NotImplementedError("QdrantProvider.embed - use embedding models via LocalAI/OpenAI")

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            client = self._get_client()
            await client.get_collections()
            return True
        except Exception:
            return False

    async def get_similar(self, query: str, threshold: float = 0.92) -> dict[str, Any] | None:
        """Search for similar cached query above threshold."""
        try:
            await self._ensure_collection()
            vector = await self._embed(query)
            client = self._get_client()
            
            results = await client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=1,
                score_threshold=threshold,
                with_payload=True,
            )
            
            if results:
                hit = results[0]
                payload = hit.payload
                if "result" in payload:
                    cached = payload["result"].copy()
                    cached["_cache_score"] = hit.score
                    cached["_cache_query"] = payload.get("query", "")
                    return cached
            return None
        except Exception:
            return None

    async def store(self, query: str, result: dict[str, Any]) -> None:
        """Store fusion result in semantic cache."""
        try:
            await self._ensure_collection()
            vector = await self._embed(query)
            client = self._get_client()
            
            point_id = str(uuid.uuid4())
            payload = {
                "query": query,
                "result": result,
                "timestamp": time.time(),
            }
            
            await client.upsert(
                collection_name=self.collection,
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": payload,
                }]
            )
        except Exception:
            pass  # Fail silently on cache write

    async def cleanup_ttl(self, hours: int = 24) -> int:
        """Delete cache entries older than TTL hours."""
        try:
            await self._ensure_collection()
            client = self._get_client()
            cutoff = time.time() - (hours * 3600)
            
            # Scroll to find old points
            deleted_count = 0
            scroll_result = await client.scroll(
                collection_name=self.collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            
            old_ids = []
            points, _ = scroll_result
            for point in points:
                if point.payload and point.payload.get("timestamp", 0) < cutoff:
                    old_ids.append(point.id)
            
            if old_ids:
                await client.delete(
                    collection_name=self.collection,
                    points_selector=old_ids,
                )
                deleted_count = len(old_ids)
            
            return deleted_count
        except Exception:
            return 0