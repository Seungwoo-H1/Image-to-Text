import cv2
import easyocr
import pandas as pd
import numpy as np
import requests
import os

from typing import List, Dict, Tuple

# EasyOCR Reader 초기화
_reader = None

def get_reader(langs: List[str] = ['ko', 'en']):
    """EasyOCR Reader 싱글톤"""
    global _reader
    if _reader is None:
        print("[INFO] EasyOCR Reader 초기화 중...")
        _reader = easyocr.Reader(langs, gpu=False)  # GPU 없으면 False
        print("[INFO] EasyOCR Reader 초기화 완료")
    return _reader


# Preprocessing (이미지 전처리)
def preprocess_image(image_path: str, enhance: bool = True) -> np.ndarray:

    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    
    if not enhance:
        return img
    
    # 해상도 증가 (작은 텍스트 인식 개선)
    # 원본 해상도가 낮으면 업스케일링
    height, width = img.shape[:2]
    # 최소 해상도 기준을 높여서 더 선명하게 (1200px로 증가)
    min_resolution = 1200
    if width < min_resolution or height < min_resolution:
        scale = max(min_resolution / width, min_resolution / height)
        # 최대 3배까지만 스케일링 (너무 크면 처리 속도 저하)
        scale = min(scale, 3.0)
        if scale > 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # 그레이스케일 변환 (대비 향상 전)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거 (Gaussian Blur)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 선명화
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # 3채널로 변환 (EasyOCR은 컬러/그레이 모두 처리 가능하지만, 일관성을 위해)
    img_processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return img_processed


# EasyOCR로 단어 단위 탐지
def ocr_with_structure(img: np.ndarray, level: str = 'word', langs: List[str] = ['ko', 'en']) -> List[Dict]:
    """
    EasyOCR을 사용하여 구조화된 OCR 수행
    
    level: 'word', 'line', 'paragraph' 중 선택
    langs: 인식할 언어 리스트 (예: ['ko', 'en'])
    """
    reader = get_reader(langs)
    
    # EasyOCR로 텍스트 추출
    # result: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence), ...]
    results = reader.readtext(img)
    
    words_data = []
    for idx, (bbox, text, conf) in enumerate(results):
        text = text.strip()
        if not text:
            continue
        
        # bbox: 4개 꼭짓점 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        # 직사각형 바운딩 박스로 변환
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)
        
        words_data.append({
            'text': text,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'conf': int(conf * 100),  # 0~100 스케일로 변환
            'idx': idx,
        })
    
    # Y좌표 기준 정렬
    words_data = sorted(words_data, key=lambda w: (w['y'], w['x']))
    
    # 라인 번호 할당 (Y좌표 기반 휴리스틱)
    if words_data:
        line_threshold = 15  # Y 차이가 이 이상이면 다른 라인
        current_line = 0
        last_y = -line_threshold * 2
        
        for w in words_data:
            if abs(w['y'] - last_y) > line_threshold:
                current_line += 1
            w['line_num'] = current_line
            last_y = w['y']
    
    return words_data


# 라인 단위로 텍스트 그룹화
def group_by_line(words_data: List[Dict], y_threshold: float = 5.0, x_gap_threshold: float = 15.0) -> List[Dict]:
    """
    단어들을 라인 단위로 그룹화 (좌표 기반)
    
    y_threshold: Y 좌표 차이 기준 (이 이상이면 다른 라인)
    x_gap_threshold: X 좌표 간격 기준 (이 이상이면 같은 라인이라도 분리)
    """
    if not words_data:
        return []
    
    # Y 좌표 기준으로 먼저 그룹화
    y_groups = []
    sorted_words = sorted(words_data, key=lambda w: (w['y'], w['x']))
    
    current_group = [sorted_words[0]]
    
    for w in sorted_words[1:]:
        # Y 좌표가 비슷하면 같은 그룹
        last_y = current_group[-1]['y']
        if abs(w['y'] - last_y) <= y_threshold:
            current_group.append(w)
        else:
            # 새로운 Y 그룹
            y_groups.append(current_group)
            current_group = [w]
    
    if current_group:
        y_groups.append(current_group)
    
    # 각 Y 그룹 내에서 X 간격이 큰 경우 별도 라인으로 분리
    lines = []
    for y_group in y_groups:
        if len(y_group) == 1:
            lines.append(y_group)
            continue
        
        # Y 그룹 내에서 X 좌표로 정렬
        y_group_sorted = sorted(y_group, key=lambda w: w['x'])
        
        current_line = [y_group_sorted[0]]
        for w in y_group_sorted[1:]:
            # 이전 단어의 오른쪽 끝 위치
            prev_x_end = current_line[-1]['x'] + current_line[-1]['w']
            # 현재 단어의 왼쪽 시작 위치
            curr_x_start = w['x']
            
            # X 간격이 threshold보다 크면 별도 라인
            x_gap = curr_x_start - prev_x_end
            if x_gap > x_gap_threshold:
                lines.append(current_line)
                current_line = [w]
            else:
                current_line.append(w)
        
        if current_line:
            lines.append(current_line)
    
    # 라인별로 텍스트와 바운딩 박스 계산
    result = []
    for line_idx, line_words in enumerate(lines):
        if not line_words:
            continue
            
        x_min = min([w['x'] for w in line_words])
        y_min = min([w['y'] for w in line_words])
        x_max = max([w['x'] + w['w'] for w in line_words])
        y_max = max([w['y'] + w['h'] for w in line_words])
        
        result.append({
            'text': ' '.join([w['text'] for w in line_words]),
            'x': x_min,
            'y': y_min,
            'w': x_max - x_min,
            'h': y_max - y_min,
            'line_num': line_idx
        })
    
    # Y 좌표로 최종 정렬
    result.sort(key=lambda l: (l['y'], l['x']))
    for idx, line in enumerate(result):
        line['line_num'] = idx
    
    return result


# 문단 단위로 텍스트 그룹화
def group_by_paragraph(words_data: List[Dict], para_threshold_ratio: float = 1.5) -> List[Dict]:
    """
    단어들을 문단 단위로 그룹화 (라인 간격 기반)
    
    para_threshold_ratio: 평균 라인 높이의 배수로 문단 구분 (1.5 = 평균 높이의 1.5배 이상이면 새 문단)
    """
    # 먼저 라인으로 그룹화
    lines = group_by_line(words_data)

    if not lines:
        return []
    
    # 평균 라인 높이 계산
    avg_line_height = sum([line['h'] for line in lines]) / len(lines) if lines else 20
    para_threshold = avg_line_height * para_threshold_ratio

    
    # 라인들을 문단으로 그룹화
    paragraphs = []
    current_para = [lines[0]]
    last_line_bottom = lines[0]['y'] + lines[0]['h']  # 이전 라인의 하단
    
    for line in lines[1:]:
        # 현재 라인의 상단
        curr_line_top = line['y']
        
        # 라인 간 간격 계산 (이전 라인 하단부터 현재 라인 상단까지)
        line_gap = curr_line_top - last_line_bottom

        
        # 간격이 threshold보다 크면 새 문단
        if line_gap > para_threshold:
            paragraphs.append(current_para)
            current_para = [line]
        else:
            current_para.append(line)
        
        last_line_bottom = line['y'] + line['h']
    
    if current_para:
        paragraphs.append(current_para)
    
    # 문단별로 텍스트와 바운딩 박스 계산
    result = []
    for para_idx, para_lines in enumerate(paragraphs):
        para_texts = [line['text'] for line in para_lines]
        x_min = min([line['x'] for line in para_lines])
        y_min = min([line['y'] for line in para_lines])
        x_max = max([line['x'] + line['w'] for line in para_lines])
        y_max = max([line['y'] + line['h'] for line in para_lines])
        
        result.append({
            'text': ' '.join(para_texts),
            'x': x_min,
            'y': y_min,
            'w': x_max - x_min,
            'h': y_max - y_min,
            'paragraph_num': para_idx
        })
    
    return result


# LLM 내용 파악 및 설명
from dotenv import load_dotenv

def generate_text_from_data(data: List[Dict]) -> str:
    """추출된 데이터를 텍스트로 변환"""
    return '\n'.join([d['text'] for d in data])

def call_llm_for_correction(text: str, server_url: str) -> str:
    """OCR 결과를 LLM으로 교정 (오타 수정)"""
    
    load_dotenv()
    
    # server_url이 None이거나 빈 문자열이면 환경변수에서 다시 읽기
    if not server_url:
        server_url = os.getenv("LLM_SERVER_URL")
    
    if not server_url:
        print("[WARN] LLM_SERVER_URL 환경변수가 설정되지 않음. 교정 생략")
        return text
    
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("[WARN] API_KEY 환경변수가 설정되지 않음. 교정 생략")
        return text
    
    system_prompt = """당신은 OCR 결과 교정 전문가입니다.
OCR로 추출된 텍스트에는 다음과 같은 오류가 있을 수 있습니다:
- 철자 오류 (예: "Recoghize" → "Recognize")
- 문자 오인식 (예: "Tent" → "Text", "UHLV" → "UNLV")
- 단어 분리 오류 (예: "Thisis" → "This is")

원본 텍스트의 의미와 맥락을 최대한 유지하면서, 명백한 OCR 오류만 수정해주세요.
원본 구조(줄바꿈, 공백 등)는 최대한 유지해주세요.
오류가 없다고 판단되면 원본 그대로 반환해주세요."""
    
    user_prompt = f"""다음 OCR 결과를 교정해주세요:

{text}"""
    
    payload = {
        "model": "openai/gpt-oss-120b", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,  # 교정은 낮은 temperature로 일관성 있게
        "max_tokens": 2000,
        "top_p": 1
    }
    
    try:
        url = f"{server_url}/chat/completions"
        res = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {api_key}"
        }, timeout=60)
        res.raise_for_status()
        
        result = res.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            corrected = str(content).strip()
            # 프롬프트로 인해 추가된 설명문 제거
            if "```" in corrected:
                lines = corrected.split("\n")
                corrected = "\n".join([l for l in lines if not l.strip().startswith("```")])
            return corrected
        return text
            
    except Exception as e:
        print(f"[WARN] LLM 교정 실패: {e}")
        return text

def call_llm_for_description(text: str, server_url: str) -> str:
    """전체 텍스트를 LLM에 보내서 자연스럽게 설명 받기"""
    
    load_dotenv()
    
    # server_url이 None이거나 빈 문자열이면 환경변수에서 다시 읽기
    if not server_url:
        server_url = os.getenv("LLM_SERVER_URL")
    
    if not server_url:
        print("[WARN] LLM_SERVER_URL 환경변수가 설정되지 않음. 설명 생략")
        return ""
    
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("[WARN] API_KEY 환경변수가 설정되지 않음. 설명 생략")
        return ""
    
    system_prompt = """당신은 이미지에서 추출된 텍스트를 분석하는 전문가입니다.
OCR로 추출된 텍스트를 보고 이미지의 내용을 자연스럽게 파악하여 설명해주세요.
텍스트에 오류가 있을 수 있으니, 맥락을 고려하여 의미를 파악한 후 설명해주세요.
모든 답변은 반드시 한국어로 작성해주세요."""
    
    user_prompt = f"""다음은 이미지에서 OCR로 추출된 텍스트입니다:

{text}

이 텍스트가 어떤 내용을 담고 있는지, 이미지에서 무엇을 보여주는지 자연스럽게 설명해주세요."""
    
    payload = {
        "model": "openai/gpt-oss-120b", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
        "top_p": 1
    }
    
    try:
        url = f"{server_url}/chat/completions"
        res = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {api_key}"
        }, timeout=60)
        res.raise_for_status()
        
        result = res.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            return str(content).strip()
        return ""
            
    except Exception as e:
        print(f"[WARN] LLM 설명 생성 실패: {e}")
        return ""


# Output
def save_output(data: List[Dict], txt_path: str, csv_path: str, description: str = ""):
    # TXT 저장
    with open(txt_path, 'w', encoding='utf-8') as f:
        # for d in data:
        #     f.write(d['text'] + "\n")
        
        if description:
            f.write("\n" + "="*60 + "\n")
            f.write("이미지 내용 설명:\n")
            f.write("="*60 + "\n")
            f.write(description + "\n")

    # CSV 저장
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding='utf-8')


# Pipeline
def process_image(image_path: str, txt_path: str, csv_path: str, llm_server_url: str, level: str, 
                  enable_correction: bool = True, enhance_preprocessing: bool = True):

    print(f"[INFO] 이미지 처리 시작: {image_path}")
    print(f"[INFO] 탐지 레벨: {level}")
    print(f"[INFO] 전처리 강화: {enhance_preprocessing}")
    print(f"[INFO] OCR 교정: {enable_correction}")
    
    # 전처리
    gray_img = preprocess_image(image_path, enhance=enhance_preprocessing)
    
    # OCR로 단어 단위 추출
    words_data = ocr_with_structure(gray_img, level='word')
    print(f"[INFO] 탐지된 단어 수: {len(words_data)}")
    
    # 레벨에 따라 그룹화
    if level == 'line':
        data = group_by_line(words_data)
        print(f"[INFO] 그룹화된 라인 수: {len(data)}")
    elif level == 'paragraph':
        data = group_by_paragraph(words_data)
        print(f"[INFO] 그룹화된 문단 수: {len(data)}")
    else:  # word
        data = words_data
    
    # OCR 교정 (Optional)
    if enable_correction:
        print(f"[INFO] OCR 교정 시작...")
        full_text = generate_text_from_data(data)
        corrected_text = call_llm_for_correction(full_text, llm_server_url)
        
        # 교정된 텍스트를 다시 데이터 구조로 분할
        if corrected_text != full_text:
            print(f"[INFO] OCR 교정 완료")
            # 교정된 텍스트를 라인별로 분할하여 업데이트
            corrected_lines = corrected_text.split('\n')
            for i, item in enumerate(data):
                if i < len(corrected_lines):
                    item['text'] = corrected_lines[i].strip()
                    item['corrected'] = True
                else:
                    item['corrected'] = False
        else:
            print(f"[INFO] OCR 교정: 변경사항 없음")
            for item in data:
                item['corrected'] = False
    else:
        for item in data:
            item['corrected'] = False
    
    # LLM 설명 생성
    full_text = generate_text_from_data(data)
    description = call_llm_for_description(full_text, llm_server_url)
    
    if description:
        print(f"[INFO] LLM 설명 생성 완료")
    
    # 저장
    save_output(data, txt_path, csv_path, description)
    
    print(f"[INFO] 완료! TXT: {txt_path}, CSV: {csv_path}")
    return data, description


# 실행
if __name__ == "__main__":
    
    load_dotenv()
    IMAGE_PATH = "img/img_to_text_sample_2.png"
    TXT_PATH = "output/output.txt"
    CSV_PATH = "output/output.csv"
    LLM_SERVER_URL = os.getenv("LLM_SERVER_URL")
    
    # level: 'word', 'line', 'paragraph' 중 선택
    process_image(IMAGE_PATH, TXT_PATH, CSV_PATH, LLM_SERVER_URL, level='line')
