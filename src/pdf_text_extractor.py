"""
PDF 텍스트 추출 모듈
- PDF에서 텍스트만 추출 (OCR 제외)
- 다이어그램/Figure 관련 텍스트 감지
- 다이어그램 영역 스냅샷 생성
"""

import PyPDF2
from pathlib import Path
from typing import List, Dict, Optional
import re
import base64
import cv2
import numpy as np
from PIL import Image
import io

# PyMuPDF 사용 
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    import sys
    print(f"[WARN] PyMuPDF를 import할 수 없습니다. 다이어그램 스냅샷 기능을 사용할 수 없습니다.")
    print(f"[INFO] 현재 Python: {sys.executable}")
    print(f"[INFO] 해결 방법:")
    print(f"      1. conda 환경 활성화: conda activate sk")
    print(f"      2. 또는 설치: pip install PyMuPDF")
    print(f"      3. 또는 sk 환경의 python 직접 사용: C:\\Users\\INSoft\\miniconda3\\envs\\sk\\python.exe diagram_gui.py")


class PDFTextExtractor:
    """PDF 텍스트 추출 클래스"""
    
    def __init__(self):
        """초기화"""
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        PDF에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
        
        Returns:
            List[Dict]: 각 라인의 텍스트 정보
            [
                {
                    "page_num": 1,
                    "text": "추출된 텍스트",
                    "text_type": "pdf_extracted",
                    "item_idx": 0
                },
                ...
            ]
        """
        text_items = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"[INFO] PDF 총 페이지 수: {total_pages}\n")
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            # 텍스트를 라인 단위로 분리하여 각각 저장
                            lines = text.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and len(line) > 1:  # 1자 이상만 저장
                                    text_items.append({
                                        "page_num": page_num,
                                        "text": line,
                                        "text_type": "pdf_extracted",
                                        "item_idx": len(text_items)
                                    })
                            print(f"[INFO] 페이지 {page_num}: {len(lines)}개 라인 추출")
                        else:
                            print(f"[WARN] 페이지 {page_num}: 텍스트 추출 실패 (스캔된 PDF일 수 있음)")
                    except Exception as e:
                        print(f"[WARN] 페이지 {page_num} 텍스트 추출 실패: {e}")
                        
        except Exception as e:
            print(f"[ERROR] PDF 텍스트 추출 실패: {e}")
            import traceback
            traceback.print_exc()
        
        if not text_items:
            print("[WARN] PDF에서 텍스트를 추출할 수 없습니다.")
            print("스캔된 PDF이거나 이미지 기반 PDF일 수 있습니다.")
        else:
            print(f"\n[INFO] 총 추출된 텍스트 아이템: {len(text_items)}개")
        
        return text_items
    
    def detect_diagram_texts(self, text_items: List[Dict]) -> List[Dict]:
        """
        다이어그램 관련 텍스트 감지
        
        Args:
            text_items: 텍스트 아이템 리스트
        
        Returns:
            List[Dict]: 다이어그램 관련 텍스트 아이템 리스트
        """
        diagram_keywords = [
            r'^Figure\s+\d+\.\d+',  # Figure 3.1
            r'^Table\s+\d+\.\d+',   # Table 3.1
            r'^Figure\s+\d+',        # Figure 3
            r'^Table\s+\d+',         # Table 3
            r'Block Diagram',
            r'Flowchart',
            r'Diagram',
        ]
        
        diagram_items = []
        
        for item in text_items:
            text = item.get('text', '')
            
            # 키워드 매칭
            is_diagram = False
            for pattern in diagram_keywords:
                if re.search(pattern, text, re.IGNORECASE):
                    is_diagram = True
                    break
            
            # 짧은 단어들만 있는 경우 (다이어그램 레이블)
            # 예: "CPU BUS", "Core Reset" 등
            words = text.split()
            if len(words) <= 3 and len(text) < 50:
                # 대문자로만 구성되거나, 짧은 단어들
                if text.isupper() or (len(words) <= 3 and all(len(w) <= 15 for w in words)):
                    # Figure나 Table 다음에 나오는 텍스트인지 확인
                    item_idx = item.get('item_idx', 0)
                    if item_idx > 0:
                        prev_text = text_items[item_idx - 1].get('text', '')
                        if any(re.search(p, prev_text, re.IGNORECASE) for p in diagram_keywords[:4]):
                            is_diagram = True
            
            if is_diagram:
                item['is_diagram'] = True
                diagram_items.append(item)
        
        print(f"[INFO] 다이어그램 관련 텍스트 감지: {len(diagram_items)}개")
        return diagram_items
    
    def group_diagram_texts(self, text_items: List[Dict], diagram_items: List[Dict]) -> List[Dict]:
        """
        다이어그램 텍스트를 그룹화 (Figure 제목 + 관련 레이블들)
        
        Args:
            text_items: 전체 텍스트 아이템
            diagram_items: 다이어그램 관련 텍스트 아이템
        
        Returns:
            List[Dict]: 그룹화된 다이어그램 정보
            [
                {
                    "diagram_id": "figure_3_1",
                    "title": "Figure 3.1 Block Diagram...",
                    "related_texts": [...],
                    "start_idx": 10,
                    "end_idx": 25
                },
                ...
            ]
        """
        if not diagram_items:
            return []
        
        diagram_groups = []
        current_group = None
        
        for item in text_items:
            text = item.get('text', '')
            item_idx = item.get('item_idx', 0)
            
            # Figure/Table 제목 감지
            if re.search(r'^(Figure|Table)\s+\d+', text, re.IGNORECASE):
                # 이전 그룹 저장
                if current_group:
                    diagram_groups.append(current_group)
                
                # 새 그룹 시작
                diagram_id = re.sub(r'[^\w]', '_', text.lower())[:50]
                current_group = {
                    "diagram_id": diagram_id,
                    "title": text,
                    "related_texts": [text],
                    "start_idx": item_idx,
                    "end_idx": item_idx,
                    "page_num": item.get('page_num')
                }
            elif current_group:
                # 현재 그룹에 추가할지 판단
                # 짧은 텍스트이거나, 같은 페이지의 연속된 텍스트
                if (item.get('page_num') == current_group['page_num'] and 
                    item_idx <= current_group['end_idx'] + 10):  # 최대 10개 라인까지
                    
                    words = text.split()
                    # 짧은 단어들만 있는 경우 (다이어그램 레이블)
                    if len(words) <= 5 and len(text) < 80:
                        current_group['related_texts'].append(text)
                        current_group['end_idx'] = item_idx
        
        # 마지막 그룹 저장
        if current_group:
            diagram_groups.append(current_group)
        
        print(f"[INFO] 다이어그램 그룹: {len(diagram_groups)}개")
        return diagram_groups
    
    def create_diagram_snapshots(
        self, 
        pdf_path: str, 
        diagram_groups: List[Dict],
        text_items: List[Dict],
        dpi: int = 300
    ) -> List[Dict]:
        """
        다이어그램 영역 스냅샷 생성
        
        Args:
            pdf_path: PDF 파일 경로
            diagram_groups: 다이어그램 그룹 리스트
            text_items: 전체 텍스트 아이템 (위치 추정용)
            dpi: 이미지 해상도
        
        Returns:
            List[Dict]: 다이어그램 스냅샷 리스트
            [
                {
                    "diagram_id": "figure_3_1",
                    "title": "Figure 3.1...",
                    "page_num": 1,
                    "snapshot_base64": "...",
                    "related_texts": [...]
                },
                ...
            ]
        """
        if not diagram_groups:
            return []
        
        snapshots = []
        
        if not HAS_FITZ:
            print("[WARN] PyMuPDF가 없어 다이어그램 스냅샷을 생성할 수 없습니다.")
            return snapshots
        
        try:
            # PyMuPDF로 PDF 열기
            pdf_doc = fitz.open(pdf_path)
            
            for group in diagram_groups:
                page_num = group.get('page_num', 1)
                if page_num < 1 or page_num > len(pdf_doc):
                    continue
                
                # 해당 페이지 렌더링
                page = pdf_doc[page_num - 1]
                zoom = dpi / 72.0  # DPI를 zoom으로 변환
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # PIL Image로 변환
                img_data = pix.tobytes("png")
                page_img = Image.open(io.BytesIO(img_data))
                page_array = np.array(page_img)
                
                # OpenCV 형식으로 변환 (RGB -> BGR)
                if len(page_array.shape) == 3:
                    page_cv = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)
                else:
                    page_cv = page_array
                
                height, width = page_cv.shape[:2]
                
                # 해당 페이지의 텍스트 아이템 찾기
                page_text_items = [item for item in text_items if item.get('page_num') == page_num]
                
                # Figure 제목의 인덱스 찾기
                start_idx = group.get('start_idx', 0)
                end_idx = group.get('end_idx', start_idx + 20)  # 최대 20개 라인
                
                # 다이어그램 영역 추정
                # Figure 제목 다음에 나오는 영역 (페이지 중앙~하단)
                # 관련 텍스트가 많을수록 다이어그램이 클 가능성
                num_related = len(group.get('related_texts', []))
                
                if num_related > 5:
                    # 큰 다이어그램 (많은 레이블)
                    diagram_y_start = int(height * 0.15)  # 상단 15% 제외
                    diagram_y_end = int(height * 0.95)     # 하단 5% 제외
                else:
                    # 작은 다이어그램
                    diagram_y_start = int(height * 0.25)  # 상단 25% 제외
                    diagram_y_end = int(height * 0.90)     # 하단 10% 제외
                
                diagram_x_start = int(width * 0.05)   # 좌측 5% 제외
                diagram_x_end = int(width * 0.95)      # 우측 5% 제외
                
                # 다이어그램 영역 crop
                diagram_region = page_cv[
                    diagram_y_start:diagram_y_end,
                    diagram_x_start:diagram_x_end
                ]
                
                # Base64 인코딩
                _, buffer = cv2.imencode('.png', diagram_region)
                snapshot_base64 = base64.b64encode(buffer).decode('utf-8')
                
                snapshots.append({
                    "diagram_id": group['diagram_id'],
                    "title": group['title'],
                    "page_num": page_num,
                    "snapshot_base64": snapshot_base64,
                    "related_texts": group['related_texts'],
                    "bbox": [diagram_x_start, diagram_y_start, 
                            diagram_x_end - diagram_x_start, 
                            diagram_y_end - diagram_y_start]
                })
                
                print(f"[INFO] 다이어그램 스냅샷 생성: {group['title']} (페이지 {page_num}, {num_related}개 레이블)")
            
            # PDF 문서 닫기
            pdf_doc.close()
        
        except Exception as e:
            print(f"[ERROR] 다이어그램 스냅샷 생성 실패: {e}")
            import traceback
            traceback.print_exc()
        
        return snapshots
    
    def process_pdf_with_diagrams(self, pdf_path: str) -> Dict:
        """
        PDF 처리 및 다이어그램 감지 전체 파이프라인
        
        Args:
            pdf_path: PDF 파일 경로
        
        Returns:
            Dict: 처리 결과
            {
                "text_items": [...],
                "diagram_items": [...],
                "diagram_groups": [...],
                "diagram_snapshots": [...]
            }
        """
        print(f"[INFO] PDF 처리 시작: {pdf_path}")
        
        # 1. 텍스트 추출
        text_items = self.extract_text_from_pdf(pdf_path)
        
        # 2. 다이어그램 감지
        diagram_items = self.detect_diagram_texts(text_items)
        
        # 3. 다이어그램 그룹화
        diagram_groups = self.group_diagram_texts(text_items, diagram_items)
        
        # 4. 다이어그램 스냅샷 생성
        diagram_snapshots = []
        if diagram_groups:
            diagram_snapshots = self.create_diagram_snapshots(pdf_path, diagram_groups, text_items)
        
        return {
            "text_items": text_items,
            "diagram_items": diagram_items,
            "diagram_groups": diagram_groups,
            "diagram_snapshots": diagram_snapshots
        }


if __name__ == "__main__":
    # 테스트
    extractor = PDFTextExtractor()
    
    pdf_path = "pdf/pdf_test_sample.pdf"
    text_items = extractor.extract_text_from_pdf(pdf_path)
    
    if text_items:
        print("\n[추출된 텍스트 샘플 - 처음 20개]:")
        for i, item in enumerate(text_items[:20]):
            preview = item['text'][:80] + '...' if len(item['text']) > 80 else item['text']
            print(f"  [{i+1}] Page {item['page_num']}: {preview}")
        
        # 다이어그램 감지
        diagram_items = extractor.detect_diagram_texts(text_items)
        print(f"\n[다이어그램 관련 텍스트]: {len(diagram_items)}개")
        for item in diagram_items[:10]:
            print(f"  - {item['text']}")
        
        # 다이어그램 그룹화
        diagram_groups = extractor.group_diagram_texts(text_items, diagram_items)
        print(f"\n[다이어그램 그룹]: {len(diagram_groups)}개")
        for group in diagram_groups:
            print(f"  - {group['title']}: {len(group['related_texts'])}개 관련 텍스트")

