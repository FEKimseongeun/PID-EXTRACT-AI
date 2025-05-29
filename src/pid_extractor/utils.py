import os
import re
import pandas as pd
import fitz  # PyMuPDF for PDF-to-image


def ensure_dir(path: str):
    """
    폴더가 없으면 생성합니다.
    """
    os.makedirs(path, exist_ok=True)


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200):
    """
    PDF를 이미지로 변환해 out_dir에 저장하고, 저장된 이미지 경로 리스트를 반환합니다.
    """
    #ensure_dir(out_dir)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    paths = []
    for i in range(doc.page_count):
        pix = doc[i].get_pixmap(matrix=mat)
        out_path = os.path.join(out_dir, f"page_{i}.png")
        pix.save(out_path)
        paths.append(out_path)

    return paths


def write_sheets(data_dict: dict, out_xlsx: str):
    """
    dict 형태({sheet_name: DataFrame})로 받은 데이터를 각 시트에 써서 엑셀로 저장합니다.
    """
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for sheet, df in data_dict.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet, index=False)

def clean_inst_tag_excel(input_excel: str,
                          output_excel: str,
                          sheet_name: str = "INST_TAG"):
    """
    INST_TAG 시트에서 OCR 결과를 정제해 새로운 엑셀로 저장합니다.

    - 알파벳+하이픈+(숫자+옵셔널알파벳 OR #)
    - 이후에 공백+여러 알파벳(옵션)
    - 괄호 제거
    """
    df = pd.read_excel(input_excel, sheet_name=sheet_name)
    pattern = re.compile(
        r'([A-Za-z]+-'               # 앞 알파벳+하이픈
        r'(?:[0-9]+[A-Za-z]?|#)'     # 숫자+옵셔널한 알파벳 하나 OR #
        r'(?: [A-Za-z]+)?)'          # 띄어쓰기+알파벳(1자 이상)
    )
    def clean_ocr(cell):
        if pd.isna(cell):
            return None
        s = str(cell).strip()
        s = s.replace('(', '').replace(')', '')
        m = pattern.search(s)
        return m.group(1) if m else None

    df['cleaned'] = df['Text'].apply(clean_ocr)
    df = df.dropna(subset=['cleaned'])
    df['Text'] = df['cleaned']
    df = df.drop(columns=['cleaned'])
    df.to_excel(output_excel, index=False)
    print(f"[DONE] cleaned INST_TAG 엑셀 저장: {output_excel}")


def pdf_to_text_lines(pdf_path: str):
    """
    PyMuPDF로 PDF의 각 페이지에서 텍스트 줄 단위 추출 (공백 후처리 없이 그대로).
    Yields (페이지 번호, 줄 텍스트)
    """
    import fitz
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        for line in text.splitlines():
            yield page_num + 1, line.strip()
