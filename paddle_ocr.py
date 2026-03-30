#!/usr/bin/env python3
"""PaddleOCR 기반 한국어 OCR. 보정된 이미지에서 텍스트 추출."""
import os
import sys
import cv2
import tempfile

def run_ocr(image_path, output_txt=None):
    """이미지에서 PaddleOCR로 텍스트 추출."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="korean")

    # tif → png 변환 (PaddleOCR은 tif 미지원)
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지 읽기 실패: {image_path}")
        return

    tmp = tempfile.mktemp(suffix=".png")
    cv2.imwrite(tmp, img)

    result = ocr.predict(tmp)
    os.unlink(tmp)

    lines = []
    for item in result:
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])
        for text, score in zip(texts, scores):
            lines.append((score, text))

    if output_txt is None:
        output_txt = os.path.splitext(image_path)[0] + ".txt"

    with open(output_txt, "w", encoding="utf-8") as f:
        for score, text in lines:
            f.write(text + "\n")

    print(f"  {len(lines)}줄 -> {os.path.basename(output_txt)}")
    return lines


def batch_ocr(result_dir):
    """결과 디렉토리의 모든 보정 이미지에 OCR 적용."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="korean")

    files = sorted([f for f in os.listdir(result_dir) if f.endswith(".tif")])
    print(f"총 {len(files)}개 파일 처리")

    for fname in files:
        img_path = os.path.join(result_dir, fname)
        txt_path = os.path.splitext(img_path)[0] + ".txt"

        img = cv2.imread(img_path)
        if img is None:
            print(f"  {fname}: 읽기 실패")
            continue

        tmp = tempfile.mktemp(suffix=".png")
        cv2.imwrite(tmp, img)

        result = ocr.predict(tmp)
        os.unlink(tmp)

        lines = []
        for item in result:
            texts = item.get("rec_texts", [])
            scores = item.get("rec_scores", [])
            for text, score in zip(texts, scores):
                lines.append(text)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  {len(lines):3d}줄  {fname}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 paddle_ocr.py <이미지/디렉토리>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        batch_ocr(target)
    else:
        run_ocr(target)
