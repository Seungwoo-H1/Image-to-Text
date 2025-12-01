import os
from typing import List, Dict

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class LLMClient:

    def __init__(
        self,
        server_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = (server_url or os.getenv("LLM_SERVER_URL") or "").strip()
        self.api_key = (api_key or os.getenv("API_KEY") or "").strip()
        self.model = (model or os.getenv("LLM_MODEL_NAME") or "openai/gpt-oss-120b").strip()
        self.timeout = timeout

        if not self.base_url:
            raise ValueError("LLM 서버 URL(LLM_SERVER_URL)이 설정되지 않았습니다.")

        self.base_url = self.base_url.rstrip("/")

    def _get_chat_endpoint(self) -> str:
        # 사용자가 이미 /chat/completions 까지 적어놨으면 그대로 사용
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        # /v1 로 끝나면 /chat/completions 추가
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        # 그 외에는 /v1/chat/completions 가정
        return f"{self.base_url}/v1/chat/completions"

    def generate_answer(
        self,
        question: str,
        context_snippets: List[str],
        chat_history: List[Dict[str, str]] | None = None,
    ) -> str:
        """
        컨텍스트 + 기존 대화 기록을 바탕으로 답변 생성.
        """
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "너는 PDF 매뉴얼/데이터시트/스펙문서를 기반으로 인사이트를 주는 전문가 어시스턴트야. "
                    "주어진 컨텍스트 안에서만 근거를 찾고, 모르는 내용은 모른다고 말해. "
                    "최대한 한국어로, 구조적으로, 핵심을 요약해서 설명해줘."
                ),
            }
        ]

        # 이전 대화 히스토리(요약 수준)
        if chat_history:
            # 너무 길어지지 않도록 최근 몇 개만 사용
            for msg in chat_history[-6:]:
                messages.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    }
                )

        # 컨텍스트 주입
        if context_snippets:
            context_text = "\n".join(context_snippets)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "다음은 PDF에서 검색된 관련 컨텍스트야. 답변할 때 이 내용을 우선적으로 참고해:\n\n"
                        f"{context_text}"
                    ),
                }
            )

        # 최종 사용자 질문
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        url = self._get_chat_endpoint()

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM 서버 오류: {resp.status_code} - {resp.text[:500]}")

        data = resp.json()
        # OpenAI 호환 형식: {"choices":[{"message":{"content":...}}]}
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # pragma: no cover - 보호용
            raise RuntimeError(f"LLM 응답 파싱 실패: {data}") from e


