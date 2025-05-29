import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from configTest import TEXT_MODEL_DIR, TEXT
from utilsTest import write_sheets


# 1) 디바이스 설정, 모델·토크나이저 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    TEXT_MODEL_DIR,
    num_labels=len(TEXT["label_map"])
).to(device).eval()

def predict_from_txt_file(txt_file: str, output_excel: str):
    """
    단일 txt 파일(txt_file)을 읽어 LINE/EQT 예측 후,
    output_excel 경로에 결과를 저장합니다.
    """
    txt_path = Path(txt_file)
    lines = txt_path.read_text(encoding="utf-8").splitlines()

    eqt_records, line_records = [], []
    buf_texts = []

    def flush_buffer():
        if not buf_texts:
            return
        # 토크나이즈 & 예측
        enc = tokenizer(buf_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=TEXT["max_length"]).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        # 결과 레코드로 분류
        for txt, p in zip(buf_texts, preds):
            lbl = TEXT["label_map"][p]
            rec = {"Text": txt, "Page": 1}
            if lbl == "EQT":
                eqt_records.append(rec)
            elif lbl == "LINE":
                line_records.append(rec)
        buf_texts.clear()

    # 2) 줄 단위로 버퍼링 → 배치 예측
    for line in lines:
        line = line.strip()
        if not line:
            continue
        buf_texts.append(line)
        if len(buf_texts) >= TEXT["batch_size"]:
            flush_buffer()
    flush_buffer()

    # 3) DataFrame으로 묶어서 Excel로 저장
    dfs = {
        "EQT": pd.DataFrame(eqt_records).assign(Label="EQT"),
        "LINE": pd.DataFrame(line_records).assign(Label="LINE")
    }
    write_sheets(dfs, output_excel)
    print(f"[DONE] 단일 TXT 결과 저장: {output_excel}")


if __name__ == "__main__":
    # 여기에 분석할 .txt 파일 경로와 출력 Excel 경로를 지정하세요
    txt_file = "pdf_txt_file/1.txt"
    output_excel = "result_single_txt.xlsx"

    predict_from_txt_file(txt_file, output_excel)


