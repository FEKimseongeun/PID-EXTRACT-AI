import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from . import config, utils

def extract_text_labels(pdf_path: str, output_excel: str):
    # 1) 모델·토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.TEXT_MODEL_DIR,
        num_labels=len(config.TEXT["label_map"])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 2) PDF → 버퍼링 → 예측 → 리스트 수집
    eqt_records, line_records = [], []
    buf_texts, buf_pages = [], []

    def flush_buffer():
        if not buf_texts: return
        enc = tokenizer(buf_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=config.TEXT["max_length"]).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        for txt, pg, p in zip(buf_texts, buf_pages, preds):
            lbl = config.TEXT["label_map"][p]
            if lbl == "EQT":
                eqt_records.append({"Text": txt, "Page": pg})
            elif lbl == "LINE":
                line_records.append({"Text": txt, "Page": pg})
        buf_texts.clear(); buf_pages.clear()

    for pg, line in utils.pdf_to_text_lines(pdf_path):
        if line:
            buf_texts.append(line); buf_pages.append(pg)
            if len(buf_texts) >= config.TEXT["batch_size"]:
                flush_buffer()
    flush_buffer()

    # 3) 엑셀 쓰기
    dfs = {
        "EQT": pd.DataFrame(eqt_records).assign(Label="EQT"),
        "LINE": pd.DataFrame(line_records).assign(Label="LINE")
    }
    utils.write_sheets(dfs, output_excel)
    print(f"[DONE] TEXT 모드 결과: {output_excel}")
