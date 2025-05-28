import argparse
from .linelist_extraction import extract_text_labels
from .inst_extraction import extract_instrument_tags

def main():
    parser = argparse.ArgumentParser(
        description="P&ID 자동 추출기: 텍스트 vs. INST TAG 선택 실행"
    )
    parser.add_argument("-i", "--input",  required=True,
                        help="입력 PDF 파일 경로")
    parser.add_argument("-o", "--output", required=True,
                        help="출력 Excel 파일 경로")
    parser.add_argument("-m", "--mode",   required=True,
                        choices=["text", "symbol"],
                        help="text → LINE/EQT, symbol → INST TAG NO")
    args = parser.parse_args()

    if args.mode == "text":
        extract_text_labels(
            pdf_path=args.input,
            output_excel=args.output
        )
    else:
        extract_instrument_tags(
            pdf_path=args.input,
            output_excel=args.output
        )

if __name__ == "__main__":
    main()
