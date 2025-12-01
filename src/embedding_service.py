import os
from typing import List

import requests

# .env 사용 시 자동 로드 (선택적)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class EmbeddingService:

    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = (server_url or os.getenv("Embedding_SERVER_URL") or "").strip()
        self.api_key = (api_key or os.getenv("ACCESS_TOKEN2") or "").strip()
        self.model = (model or os.getenv("EMBEDDING_MODEL_NAME") or "qwen3").strip()
        self.timeout = timeout

        if not self.base_url:
            raise ValueError("Embedding 서버 URL(Embedding_SERVER_URL)이 설정되지 않았습니다.")

        # 마지막 슬래시 정리
        self.base_url = self.base_url.rstrip("/")

    def _get_embeddings_endpoint(self) -> str:
        # 서버 주소에 맞게 그대로 사용
        return self.base_url  # 예: http://gpurent.kogrobo.com:11436/_inference/text_embedding/qwen3

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 리스트를 임베딩 벡터 리스트로 변환.
        """
        if not texts:
            return []

        url = self._get_embeddings_endpoint()

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["access_token"] = self.api_key

        # 서버 전용 payload 구조
        payload = {
            "input": texts if len(texts) > 1 else texts[0],
            "input_type": "string",
            "task_settings": {"additionalProp1": {}}
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding 서버 오류: {resp.status_code} - {resp.text[:500]}")

        data = resp.json()
        embeddings: List[List[float]] = []

        # 단일 텍스트/다중 텍스트 모두 처리
        results = data.get("inference_results")
        if not results:
            raise RuntimeError(f"Embedding 응답 형식 오류: {data}")

        if isinstance(payload["input"], list):
            # 다중 텍스트: 결과가 리스트로 반환될 것으로 가정
            for item in results:
                emb = item.get("text_embedding")
                if emb is None:
                    raise RuntimeError(f"Embedding 항목에 text_embedding 필드가 없습니다: {item}")
                embeddings.append(emb)
        else:
            # 단일 텍스트
            emb = results[0].get("text_embedding")
            if emb is None:
                raise RuntimeError(f"Embedding 항목에 text_embedding 필드가 없습니다: {results[0]}")
            embeddings.append(emb)

        return embeddings
