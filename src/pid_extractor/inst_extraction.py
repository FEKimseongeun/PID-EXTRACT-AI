import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.ops as ops  # NMS
import pytesseract
import re  # 정규식
from paddleocr import PaddleOCR
import pandas as pd  # 엑셀 출력을 위해 필요

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################################
# 1) 클래스별 NMS 후처리 함수   #
#################################
def apply_nms_per_class(boxes, labels, scores, iou_thresh=0.5):
    """
    boxes, labels, scores: np.array
    iou_thresh: 두 박스가 이 값 이상으로 겹치면 하나만 남기고 제거
    """
    final_boxes = []
    final_labels = []
    final_scores = []

    unique_labels = np.unique(labels)

    for cls in unique_labels:
        mask = (labels == cls)
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]

        if len(cls_boxes) == 0:
            continue

        cls_boxes_t = torch.from_numpy(cls_boxes)
        cls_scores_t = torch.from_numpy(cls_scores)
        keep_indices = ops.nms(cls_boxes_t, cls_scores_t, iou_thresh)
        keep_indices = keep_indices.numpy()

        final_boxes.append(cls_boxes[keep_indices])
        final_labels.append(np.full_like(keep_indices, cls))
        final_scores.append(cls_scores[keep_indices])

    if len(final_boxes) > 0:
        final_boxes = np.concatenate(final_boxes, axis=0)
        final_labels = np.concatenate(final_labels, axis=0)
        final_scores = np.concatenate(final_scores, axis=0)
    else:
        final_boxes = np.array([])
        final_labels = np.array([])
        final_scores = np.array([])

    return final_boxes, final_labels, final_scores


########################
# 2) 슬라이딩 윈도우   #
########################
def slice_image(img_pil, tile_size=640, overlap=100):
    """
    큰 이미지를 tile_size x tile_size로 오버랩하며 분할
    return: [((x1,y1,x2,y2), tile_img), ... ]
    """
    width, height = img_pil.size
    x_step = tile_size - overlap
    y_step = tile_size - overlap

    slices = []
    for y in range(0, height, y_step):
        for x in range(0, width, x_step):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            tile = img_pil.crop((x, y, x_end, y_end))
            slices.append(((x, y, x_end, y_end), tile))
    return slices


###################################################
# 3) OCR 전처리 함수 (이진화 + morphological 등)  #
###################################################
def preprocess_for_ocr(roi_pil):
    """
    OCR 인식률을 높이기 위한 전처리 예시.
    1) RGB -> GRAY
    2) adaptive threshold (이진화)
    3) morphology (열림/닫힘) 연산 등

    return: 전처리 후의 PIL 이미지
    """
    roi_np = np.array(roi_pil)  # PIL -> NumPy
    roi_gray = cv2.cvtColor(roi_np, cv2.COLOR_RGB2GRAY)

    # 1) 적당히 adaptiveThreshold
    roi_bin = cv2.adaptiveThreshold(
        roi_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or cv2.ADAPTIVE_THRESH_MEAN_C
        cv2.THRESH_BINARY,
        11,  # 블록 사이즈(이미지 상황에 따라 조절)
        2  # 상수(이미지 상황에 따라 조절)
    )

    # 2) Morphological 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # 열림(노이즈 제거)
    roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    # 닫힘(글자 사이 내부 구멍 메우기)
    roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 필요시 반전
    # roi_bin = cv2.bitwise_not(roi_bin)

    # 최종 결과를 다시 PIL로
    roi_pil_processed = Image.fromarray(roi_bin)
    return roi_pil_processed


#################################################
# 4) 슬라이딩 윈도우 추론 + 특정 라벨 OCR 파이프라인
#################################################
def infer_sliding_window_with_ocr(
        model,
        image_path,
        threshold,
        tile_size,
        overlap,
        iou_thresh,
        target_labels,
        save_path='TEST_PID_IMG/detected_img'
):
    """
    1) 슬라이싱 → 각 타일 추론
    2) threshold 후 결과 모으기
    3) 최종 NMS
    4) 라벨(target_labels)인 박스만 OCR (전처리 포함)
        - 첫 줄 = 알파벳만 추출
        - 두 번째 줄 = 숫자만 추출
        -> "알파벳-숫자" 형태로 병합
    5) 결과 시각화 + 저장
    """
    if target_labels is None:
        target_labels = [26, 27, 28, 29]  # 기본값

    # 원본 이미지 로드
    img_pil = Image.open(image_path).convert("RGB")
    width, height = img_pil.size

    # 1) 슬라이싱
    slices = slice_image(img_pil, tile_size=tile_size, overlap=overlap)

    all_boxes = []
    all_labels = []
    all_scores = []

    model.eval()
    for (coords, tile_img) in slices:
        tile_tensor = T.ToTensor()(tile_img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(tile_tensor)

        boxes = preds[0]['boxes'].cpu().numpy()
        labels = preds[0]['labels'].cpu().numpy()
        scores = preds[0]['scores'].cpu().numpy()

        # threshold 적용
        keep = scores >= threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        # 타일 좌표를 원본 이미지 좌표로 보정
        x_offset, y_offset = coords[0], coords[1]
        boxes[:, [0, 2]] += x_offset
        boxes[:, [1, 3]] += y_offset

        all_boxes.append(boxes)
        all_labels.append(labels)
        all_scores.append(scores)

    # 2) 전 타일 결과 합치기
    if len(all_boxes) > 0:
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
    else:
        all_boxes = np.array([])
        all_labels = np.array([])
        all_scores = np.array([])

    # 3) NMS
    all_boxes, all_labels, all_scores = apply_nms_per_class(
        all_boxes, all_labels, all_scores, iou_thresh
    )

    # 4) OCR 수행 및 바운딩 박스 좌표 출력
    ocr_results = []
    img_np = np.array(img_pil)

    for box, lbl, score in zip(all_boxes, all_labels, all_scores):
        if lbl not in target_labels:
            continue  # 타겟 라벨이 아닌 경우 건너뛰기

        x1, y1, x2, y2 = map(int, box)

        # (A) 박스 그리기 (blue)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img_np,
            f"{lbl}:{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )
        # 바운딩 박스 좌표, 라벨, 스코어 출력 및 결과 리스트에 저장
        print(f"Detected -> Label: {lbl}, Score: {score:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})")
        ocr_results.append((x1, y1, x2, y2))

    # 최종 시각화
    # plt.figure(figsize=(12, 12))
    # plt.imshow(img_np)
    # plt.axis("off")
    # plt.show()

    # 결과 이미지 저장 (save_path가 지정된 경우)
    if save_path is not None:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)

    return ocr_results


#################################
# 5) 메인 테스트 실행 (예시)    #
#################################
# 이미지 크롭
def crop_bounding_boxes_pil(image_path, bboxes):
    image = Image.open(image_path).convert("RGB")
    cropped_images = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cropped = image.crop((x1, y1, x2, y2))
        cropped_images.append(cropped)
        # print(f"Saved cropped image: cropped_pil_{i}.png | BBox: ({x1}, {y1}, {x2}, {y2})")
    return cropped_images


# PaddleOCR을 통한 OCR 수행 (예외처리 + 개선된 텍스트 전처리 추가)
def perform_ocr_on_images(cropped_images, use_preprocessing=False):
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        det_model_dir='paddle_model/en_PP-OCRv3_det_infer',
        rec_model_dir='paddle_model/en_PP-OCRv3_rec_infer',
        cls_model_dir='paddle_model/ch_ppocr_mobile_v2.0_cls_infer'
    )
    ocr_results = []

    for i, img in enumerate(cropped_images):
        img_np = np.array(img)
        result = ocr.ocr(img_np, cls=True)

        try:
            texts = [line[1][0] for line in result[0]]
            recognized_text = ' '.join(texts).strip()  # 맨 앞뒤 공백 제거

            # 맨 앞에 절대 하이픈이 안오도록, 텍스트 중간의 첫번째 공백만 하이픈으로 변경
            if ' ' in recognized_text:
                parts = recognized_text.split(' ', 1)  # 첫 번째 공백만 분리
                recognized_text = f"{parts[0]}-{parts[1]}"

        except TypeError:
            recognized_text = ""
            print(f"[Warning] OCR 결과 없음 (이미지 {i})")

        print(recognized_text)  # recognized_text만 출력
        ocr_results.append(recognized_text)

    return ocr_results


# 엑셀로 저장하는 함수 (모든 결과를 하나의 엑셀 파일로 저장)
def save_results_to_excel(ocr_results, excel_path):
    df = pd.DataFrame(ocr_results)
    df.to_excel(excel_path, index=False)
    print(f"[INFO] OCR 결과가 '{excel_path}' 파일로 저장되었습니다.")


####################################################################
# 6) 폴더 내 여러 이미지에 대해 추론+OCR 수행 후, 각 이미지별로 엑셀 파일 생성 #
####################################################################
if __name__ == "__main__":
    # (1) 모델 로드
    model_path = "../../models/faster_rcnn_pid_model_01.pth"
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    # (2) 처리할 이미지들이 들어 있는 폴더와 결과 저장 폴더 지정
    image_folder = "../temp_inst_imgs"  # 처리할 이미지 폴더
    output_folder = "../excel_result"
    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = [".png", ".jpg", ".jpeg"]
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in valid_extensions]

    # 각 이미지별로 OCR 결과를 개별 Excel 파일로 저장
    for img_file in image_files:
        image_path = os.path.join(image_folder, img_file)
        print(f"\n==== Processing Image: {img_file} ====")
        detected_img_path = os.path.join(output_folder, f"detected_{img_file}")

        # 기존 파라미터 사용 (필요시 조정)
        detected_bboxes = infer_sliding_window_with_ocr(
            model=model,
            image_path=image_path,
            threshold=0.55,
            tile_size=1280,
            overlap=200,
            iou_thresh=0.3,
            target_labels=[26, 27, 28, 29],
            save_path=detected_img_path
        )

        if len(detected_bboxes) == 0:
            print(f"[INFO] {img_file} 이미지에서 타겟 객체를 검출하지 못했습니다.")
            continue

        # 검출된 바운딩 박스 영역 크롭 및 OCR 수행
        cropped_imgs = crop_bounding_boxes_pil(detected_img_path, detected_bboxes)
        ocr_texts = perform_ocr_on_images(cropped_imgs, use_preprocessing=False)

        # 이미지별 OCR 결과를 리스트에 기록 (한 이미지에 여러 박스 있을 경우)
        per_image_ocr_results = []
        for i, text in enumerate(ocr_texts):
            bbox = detected_bboxes[i]  # (x1, y1, x2, y2)
            per_image_ocr_results.append({
                "Image": img_file,
                "BBox": f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})",
                "OCR_Result": text
            })

        # 이미지 파일명 기반의 엑셀 파일명 생성 (예: ocr_results_sample.xlsx)
        base_name = os.path.splitext(img_file)[0]
        excel_output_path = os.path.join(output_folder, f"ocr_results_{base_name}.xlsx")
        save_results_to_excel(per_image_ocr_results, excel_output_path)

        print(f"[INFO] {img_file} 이미지에 대한 OCR 결과가 '{excel_output_path}'에 저장되었습니다.")