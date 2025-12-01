import json
import os
from typing import Dict, List, Any

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:  # pragma: no cover - 런타임 설치 안내용
    raise ImportError(
        "faiss 가 설치되지 않았습니다. 로컬 벡터 DB를 사용하려면 다음을 실행하세요:\n"
        "  pip install faiss-cpu"
    ) from e


class VectorDB:
    """
    단순 로컬 FAISS 기반 벡터 DB.

    - 텍스트 임베딩 벡터를 추가/검색
    - 메타데이터(JSON)와 인덱스(FAISS)를 로컬에 저장
    """

    def __init__(
        self,
        index_path: str = "vector_db/index.faiss",
        metadata_path: str = "vector_db/metadata.json",
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: int | None = None

        self._load()

    # ---------- 내부 유틸 ----------
    def _ensure_dir(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    def _load(self) -> None:
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.dimension = self.index.d
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def _save(self) -> None:
        self._ensure_dir()
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def reset(self) -> None:
        """기존 인덱스/메타데이터 초기화"""
        self.index = None
        self.dimension = None
        self.metadata = []
        self._ensure_dir()
        # 기존 파일 삭제(있으면)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.dimension = dim
            self.index = faiss.IndexFlatL2(dim)

    # ---------- 공개 API ----------
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        items: List[Dict[str, Any]],
    ) -> None:
        """
        이미 계산된 임베딩 + 해당 메타데이터를 추가.
        """
        if not embeddings:
            return
        if len(embeddings) != len(items):
            raise ValueError("embeddings 개수와 items 개수가 다릅니다.")

        vecs = np.asarray(embeddings, dtype="float32")
        self._ensure_index(vecs.shape[1])

        assert self.index is not None  # for type checker
        self.index.add(vecs)
        self.metadata.extend(items)

        self._save()

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        쿼리 임베딩으로 top_k 개의 유사 텍스트를 검색.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        q = np.asarray([query_embedding], dtype="float32")
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q, top_k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata[int(idx)]
            result = dict(meta)
            result["score"] = float(dist)
            results.append(result)
        return results


