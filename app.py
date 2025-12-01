"""
PDF 텍스트 추출 및 다이어그램 Q&A GUI
- PDF 텍스트 추출 (OCR 없음)
- 다이어그램 감지 및 스냅샷
- Q&A 검색 (다이어그램 관련 질문 시 이미지 표시)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import base64
import io
import os
import threading
from typing import List, Dict, Optional

from src.pdf_text_extractor import PDFTextExtractor
from src.embedding_service import EmbeddingService
from src.vector_db import VectorDB
from src.llm_client import LLMClient


class DiagramQAGUI:
    """다이어그램 Q&A GUI 클래스"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Insight Chatbot (RAG)")
        self.root.geometry("1400x900")
        
        # PDF 텍스트 추출기
        self.extractor = PDFTextExtractor()

        # 임베딩 / 벡터 DB (RAG)
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDB()

        # LLM 클라이언트 (QA용)
        self.llm_client = LLMClient()

        # 대화 히스토리 (챗봇 느낌)
        self.chat_history: List[Dict[str, str]] = []
        
        # 현재 처리 결과 저장
        self.current_result = None
        self.text_items = []
        self.diagram_snapshots = []
        
        # UI 구성
        self.setup_ui()
    
    def setup_ui(self):
        """UI 구성 (챗봇 중심 레이아웃)"""
        # 전체를 위/아래로 나누는 메인 컨테이너
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 상단: PDF 업로드/상태
        top_frame = ttk.Frame(main_frame, padding="8")
        top_frame.pack(fill=tk.X)

        file_frame = ttk.LabelFrame(top_frame, text="PDF", padding="5")
        file_frame.pack(fill=tk.X)

        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=70).pack(
            side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True
        )
        ttk.Button(file_frame, text="PDF 선택", command=self.select_pdf).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(file_frame, text="텍스트 추출/인덱싱", command=self.extract_text_thread).pack(
            side=tk.LEFT, padx=4
        )

        # 중앙: Notebook 탭 (Chat, 원문/다이어그램)
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        notebook = ttk.Notebook(center_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # --- Chat 탭 ---
        chat_tab = ttk.Frame(notebook)
        notebook.add(chat_tab, text="Chat")

        # Chat 상단: 대화 내용
        chat_display_frame = ttk.Frame(chat_tab)
        chat_display_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 2))

        self.answer_text = scrolledtext.ScrolledText(
            chat_display_frame, wrap=tk.WORD, height=25
        )
        self.answer_text.pack(fill=tk.BOTH, expand=True)
        self.answer_text.insert(
            tk.END,
            "PDF를 선택하고 텍스트를 추출한 뒤, 궁금한 점을 질문해보세요.\n"
            "예) \"Figure 3.1의 리셋 시퀀스를 설명해줘\", \"SIC 관련 제한사항 정리해줘\" 등\n\n",
        )

        # Chat 하단: 입력 + 전송
        chat_input_frame = ttk.Frame(chat_tab)
        chat_input_frame.pack(fill=tk.X, pady=(2, 4))

        ttk.Label(chat_input_frame, text="질문:").pack(side=tk.LEFT, padx=5)
        self.question_entry = ttk.Entry(chat_input_frame)
        self.question_entry.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)
        self.question_entry.bind("<Return>", lambda e: self.search_question())
        ttk.Button(chat_input_frame, text="전송", command=self.search_question).pack(
            side=tk.LEFT, padx=5
        )

        # Chat 우측/하단: 다이어그램 이미지
        image_frame = ttk.LabelFrame(chat_tab, text="다이어그램 이미지", padding="5")
        image_frame.pack(fill=tk.X, pady=(0, 4))

        self.image_label = ttk.Label(image_frame, text="다이어그램 이미지가 표시됩니다")
        self.image_label.pack(expand=True)

        # --- Source 탭 (원문/다이어그램 목록) ---
        source_tab = ttk.Frame(notebook)
        notebook.add(source_tab, text="원문 / 다이어그램")

        # 원문 텍스트
        text_frame = ttk.LabelFrame(source_tab, text="추출된 텍스트", padding="5")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        self.text_result = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=15)
        self.text_result.pack(fill=tk.BOTH, expand=True)

        # 다이어그램 목록
        diagram_frame = ttk.LabelFrame(source_tab, text="감지된 다이어그램", padding="5")
        diagram_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 4))

        diagram_list_frame = ttk.Frame(diagram_frame)
        diagram_list_frame.pack(fill=tk.BOTH, expand=True)

        self.diagram_listbox = tk.Listbox(diagram_list_frame, height=5)
        self.diagram_listbox.pack(fill=tk.BOTH, expand=True)
        self.diagram_listbox.bind("<<ListboxSelect>>", self.on_diagram_select)

        # 하단: 상태바
        self.status_var = tk.StringVar(value="준비")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_pdf(self):
        """PDF 파일 선택"""
        file_path = filedialog.askopenfilename(
            title="PDF 파일 선택",
            filetypes=[("PDF 파일", "*.pdf"), ("모든 파일", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def extract_text_thread(self):
        """텍스트 추출 (별도 스레드에서 실행)"""
        if not self.file_path_var.get():
            messagebox.showwarning("경고", "PDF 파일을 선택해주세요.")
            return
        
        # UI 비활성화
        self.status_var.set("텍스트 추출 중...")
        self.text_result.delete(1.0, tk.END)
        self.answer_text.delete(1.0, tk.END)
        self.diagram_listbox.delete(0, tk.END)
        self.root.update()
        
        # 별도 스레드에서 처리
        thread = threading.Thread(target=self.extract_text_worker)
        thread.daemon = True
        thread.start()
    
    def extract_text_worker(self):
        """텍스트 추출 작업자 함수"""
        try:
            pdf_path = self.file_path_var.get()
            
            # PDF 처리 (텍스트 + 다이어그램 정보)
            result = self.extractor.process_pdf_with_diagrams(pdf_path)
            
            self.current_result = result
            self.text_items = result.get('text_items', [])
            self.diagram_snapshots = result.get('diagram_snapshots', [])

            # ---------- RAG 인덱스 생성 ----------
            # 기존 인덱스 초기화 후, 현재 PDF 텍스트로 다시 빌드
            try:
                self.vector_db.reset()
                texts = [item.get("text", "") for item in self.text_items if item.get("text")]
                if texts:
                    embeddings = self.embedding_service.embed(texts)
                    # 메타데이터와 임베딩 개수 맞추기
                    valid_items: List[Dict] = []
                    valid_embeddings: List[List[float]] = []
                    for emb, item in zip(embeddings, self.text_items):
                        if item.get("text"):
                            valid_items.append(item)
                            valid_embeddings.append(emb)
                    self.vector_db.add_embeddings(valid_embeddings, valid_items)
                    print(f"[INFO] 벡터 DB에 {len(valid_items)}개 텍스트 인덱싱 완료")
                else:
                    print("[WARN] 인덱싱할 텍스트가 없습니다.")
            except Exception as e:
                print(f"[WARN] 벡터 DB 인덱싱 중 오류 발생: {e}")
            
            # UI 업데이트 (메인 스레드에서)
            self.root.after(0, self.update_extraction_results, result)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("오류", f"텍스트 추출 실패: {e}"))
            self.root.after(0, lambda: self.status_var.set("오류 발생"))
            import traceback
            traceback.print_exc()
    
    def update_extraction_results(self, result: Dict):
        """추출 결과 업데이트"""
        # 텍스트 표시
        self.text_result.delete(1.0, tk.END)
        
        text_items = result.get('text_items', [])
        for item in text_items:
            page_num = item.get('page_num', 0)
            text = item.get('text', '')
            is_diagram = item.get('is_diagram', False)
            
            prefix = f"[Page {page_num}]"
            if is_diagram:
                prefix += " [다이어그램]"
            
            self.text_result.insert(tk.END, f"{prefix} {text}\n")
        
        # 다이어그램 목록 업데이트
        diagram_snapshots = result.get('diagram_snapshots', [])
        self.diagram_listbox.delete(0, tk.END)
        for snapshot in diagram_snapshots:
            title = snapshot.get('title', 'Unknown')
            page_num = snapshot.get('page_num', 0)
            self.diagram_listbox.insert(tk.END, f"Page {page_num}: {title[:50]}")
        
        # 상태 업데이트
        self.status_var.set(
            f"완료 - 텍스트: {len(text_items)}개, 다이어그램: {len(diagram_snapshots)}개"
        )
    
    def on_diagram_select(self, event):
        """다이어그램 선택 시 이미지 표시"""
        selection = self.diagram_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        if idx < len(self.diagram_snapshots):
            snapshot = self.diagram_snapshots[idx]
            self.display_diagram_image(snapshot.get('snapshot_base64', ''))
    
    def display_diagram_image(self, base64_str: str):
        """다이어그램 이미지 표시"""
        if not base64_str:
            return
        
        try:
            image_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(image_data))
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            print(f"이미지 표시 오류: {e}")
    
    def search_question(self):
        """질문 → 하이브리드 검색(exact + semantic) → LLM 답변"""
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("경고", "질문을 입력해주세요.")
            return

        if not self.text_items:
            messagebox.showwarning("경고", "먼저 PDF 텍스트를 추출해주세요.")
            return

        question_lower = question.lower()
        question_words = [w for w in question_lower.split() if len(w) > 2]

        # 사용자 메시지 UI에 추가
        self.answer_text.insert(tk.END, f"사용자: {question}\n")
        self.answer_text.see(tk.END)

        # 히스토리에 추가
        self.chat_history.append({"role": "user", "content": question})

        # ---------- 하이브리드 검색: semantic + exact ----------
        semantic_results: List[Dict] = []
        keyword_scores: Dict[int, int] = {}  # item_idx -> overlap count

        # 1) semantic (FAISS)
        if self.vector_db is None or self.vector_db.is_empty():
            self.answer_text.insert(
                tk.END,
                "[WARN] 벡터 DB가 비어 있습니다. 먼저 PDF를 처리하여 인덱스를 생성하세요.\n",
            )
        else:
            try:
                query_emb = self.embedding_service.embed([question])[0]
                semantic_results = self.vector_db.search(query_emb, top_k=20)
            except Exception as e:
                self.answer_text.insert(
                    tk.END,
                    f"[WARN] 벡터 검색 중 오류 발생: {e}\n",
                )

        # 2) exact (키워드 매칭)
        for item in self.text_items:
            text = item.get("text", "")
            if not text:
                continue
            text_lower = text.lower()
            overlap = sum(1 for w in question_words if w in text_lower)
            if overlap > 0:
                idx = item.get("item_idx", -1)
                if idx >= 0:
                    keyword_scores[idx] = max(keyword_scores.get(idx, 0), overlap)

        # 3) 결과 병합
        candidates: Dict[int, Dict] = {}

        # semantic 쪽: 거리(score)를 similarity로 변환 (작을수록 가깝기 때문에)
        def dist_to_sim(dist: float) -> float:
            return 1.0 / (1.0 + dist)

        for item in semantic_results:
            idx = item.get("item_idx")
            if idx is None:
                continue
            dist = float(item.get("score", 0.0))
            semantic_score = dist_to_sim(dist)
            base = dict(item)
            base["semantic_score"] = semantic_score
            base.setdefault("keyword_score", 0.0)
            candidates[idx] = base

        # exact 쪽
        max_kw = max(keyword_scores.values()) if keyword_scores else 0
        for item in self.text_items:
            idx = item.get("item_idx")
            if idx is None or idx not in keyword_scores:
                continue
            kw_raw = keyword_scores[idx]
            kw_norm = kw_raw / max_kw if max_kw > 0 else 0.0

            if idx in candidates:
                candidates[idx]["keyword_score"] = max(
                    candidates[idx].get("keyword_score", 0.0), kw_norm
                )
            else:
                base = dict(item)
                base["semantic_score"] = 0.0
                base["keyword_score"] = kw_norm
                candidates[idx] = base

        if not candidates:
            self.answer_text.insert(tk.END, "관련 텍스트를 찾을 수 없습니다.\n\n")
            self.status_var.set("검색 완료 - 결과 없음")
            return

        # 4) 하이브리드 스코어 계산 (가중합)
        alpha = 0.7  # semantic 비중
        beta = 0.3   # exact 비중
        ranked: List[Dict] = []
        for item in candidates.values():
            s = float(item.get("semantic_score", 0.0))
            k = float(item.get("keyword_score", 0.0))
            total = alpha * s + beta * k
            item["hybrid_score"] = total
            ranked.append(item)

        ranked.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

        # 상위 N개를 컨텍스트로 사용
        top_items = ranked[:10]

        # ---------- 다이어그램 관련 여부 확인 및 이미지 표시 ----------
        diagram_keywords = [
            "figure",
            "table",
            "diagram",
            "다이어그램",
            "블록",
            "플로우차트",
            "flowchart",
            "block diagram",
            "chart",
            "그림",
        ]
        is_diagram_question = any(k in question_lower for k in diagram_keywords)
        diagram_found = any(item.get("is_diagram") for item in top_items)

        if is_diagram_question or diagram_found:
            related_diagrams = []
            for item in top_items:
                if item.get("is_diagram"):
                    for snapshot in self.diagram_snapshots:
                        if item.get("text") in snapshot.get("related_texts", []):
                            if snapshot not in related_diagrams:
                                related_diagrams.append(snapshot)

            if not related_diagrams and self.diagram_snapshots:
                related_diagrams = [self.diagram_snapshots[0]]

            if related_diagrams:
                first_diagram = related_diagrams[0]
                self.display_diagram_image(first_diagram.get("snapshot_base64", ""))

        # ---------- LLM 호출을 위한 컨텍스트 구성 ----------
        context_snippets: List[str] = []
        for item in top_items:
            page = item.get("page_num", 0)
            text = item.get("text", "")
            context_snippets.append(f"[Page {page}] {text}")

        # ---------- LLM 호출 ----------
        try:
            self.status_var.set("응답 생성 중...")
            answer = self.llm_client.generate_answer(
                question=question,
                context_snippets=context_snippets,
                chat_history=self.chat_history,
            )
        except Exception as e:
            answer = f"(LLM 호출 실패) 관련 텍스트 기반으로 직접 해석해주세요.\n에러: {e}"

        # 어시스턴트 응답 표시 및 히스토리 업데이트
        self.answer_text.insert(tk.END, f"어시스턴트:\n{answer}\n\n")
        self.answer_text.insert(
            tk.END,
            "-" * 60 + "\n",
        )
        self.answer_text.see(tk.END)

        self.chat_history.append({"role": "assistant", "content": answer})
        self.status_var.set("응답 생성 완료")


def main():
    """메인 함수"""
    root = tk.Tk()
    app = DiagramQAGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

