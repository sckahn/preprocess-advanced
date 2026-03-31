#!/usr/bin/env python3
"""
진료비영수증 이미지 보정 프로그램 (모듈식)

모듈:
  perspective - 사진에서 용지 영역 추출 + 원근 보정 (스캔 효과)
  rotate      - 90/180/270도 회전 보정 (OCR confidence)
  deskew      - 미세 기울기 보정 (jdeskew/deskew)

사용 예시:
  python line_recover.py image.tif --steps perspective,rotate,deskew
  python line_recover.py image.tif --steps rotate,deskew
  python line_recover.py image.tif --steps perspective
  python line_recover.py /폴더/ --batch --steps rotate,deskew
"""

import cv2
import numpy as np
from PIL import Image
import subprocess
import tempfile
import argparse
import os
import glob


TESSERACT_CMD = "/opt/homebrew/bin/tesseract"


# ============================================================
# 모듈 1: perspective — 용지 영역 추출 + 원근 보정
# ============================================================

def find_document_contour(img_cv):
    """이미지에서 용지 사각형을 찾기.

    스캐너 앱 방식:
    1) 밝기 유사도 기반 floodFill로 배경 분리 (가장 안정적)
    2) 큰 블러 + Canny + 컨투어 fallback
    """
    h, w = img_cv.shape[:2]
    img_area = h * w

    # 속도를 위해 축소
    scale = min(1.0, 800.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img_cv, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img_cv.copy()
    sh, sw = small.shape[:2]
    small_area = sh * sw

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    best_contour = None
    best_score = 0

    # === 방법 1: 밝기 유사도 floodFill (4코너에서 배경 채우기) ===
    for lo_diff in [6, 10, 15, 20]:
        for up_diff in [6, 10, 15, 20]:
            bg_mask = np.zeros((sh, sw), dtype=np.uint8)
            corners = [(0, 0), (sw - 1, 0), (0, sh - 1), (sw - 1, sh - 1)]

            for cx, cy in corners:
                flood_img = blur.copy()
                mask = np.zeros((sh + 2, sw + 2), dtype=np.uint8)
                cv2.floodFill(flood_img, mask, (cx, cy), 128,
                              loDiff=lo_diff, upDiff=up_diff,
                              flags=cv2.FLOODFILL_FIXED_RANGE)
                bg_mask |= (flood_img == 128).astype(np.uint8) * 255

            doc_mask = cv2.bitwise_not(bg_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            doc_mask = cv2.morphologyEx(doc_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            doc_mask = cv2.morphologyEx(doc_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contour = _find_best_quad_hull(doc_mask, small_area)
            if contour is not None:
                area = cv2.contourArea(contour)
                ratio = area / small_area
                if 0.15 < ratio < 0.95:
                    pts = contour.reshape(4, 2)
                    on_edge = sum(1 for p in pts
                                  if p[0] <= 3 or p[0] >= sw - 4
                                  or p[1] <= 3 or p[1] >= sh - 4)
                    score = area * (1.0 + 0.1 * (4 - on_edge))
                    if score > best_score:
                        best_score = score
                        best_contour = contour

    # === 방법 2: 큰 블러 + Canny fallback ===
    if best_contour is None:
        blur_large = cv2.GaussianBlur(gray, (31, 31), 0)
        for lo, hi in [(20, 60), (30, 100), (50, 150)]:
            edges = cv2.Canny(blur_large, lo, hi)
            edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=2)

            contour = _find_best_quad_hull(edges, small_area)
            if contour is not None:
                area = cv2.contourArea(contour)
                if 0.15 * small_area < area < 0.95 * small_area and area > best_score:
                    best_score = area
                    best_contour = contour

    if best_contour is None:
        return None

    # 원본 좌표로 복원
    best_contour = (best_contour.astype(np.float32) / scale).astype(np.int32)
    return best_contour


def _find_best_quad_hull(binary_or_edge, img_area):
    """이진 이미지에서 가장 큰 볼록 사각형을 찾기 (convexHull 기반)."""
    contours, _ = cv2.findContours(binary_or_edge, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.1:
            break

        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps in [0.02, 0.03, 0.05, 0.08]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
                best_area = area
                best = approx
                break

    return best


def order_points(pts):
    """4개 꼭짓점을 [좌상, 우상, 우하, 좌하] 순서로 정렬."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 좌상
    rect[2] = pts[np.argmax(s)]   # 우하
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # 우상
    rect[3] = pts[np.argmax(d)]   # 좌하
    return rect


def perspective_transform(img_cv, contour):
    """4점 원근 변환으로 용지를 직사각형으로 펴기."""
    pts = contour.reshape(4, 2).astype(np.float32)
    rect = order_points(pts)

    tl, tr, br, bl = rect
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    new_w = int(max(w1, w2))

    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    new_h = int(max(h1, h2))

    dst = np.array([
        [0, 0], [new_w - 1, 0],
        [new_w - 1, new_h - 1], [0, new_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_cv, M, (new_w, new_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    return warped


def scan_effect(img_cv):
    """흰 배경 강화 + 글자 선명하게 (스캔 느낌)."""
    # LAB 색공간에서 밝기 채널 보정
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE로 대비 강화
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 배경 흰색 강화: 밝은 영역을 더 밝게
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # 배경 추정 (큰 블러)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    # 배경 대비 보정
    normalized = cv2.divide(gray, bg, scale=255)

    # 다시 컬러로
    ratio = normalized.astype(np.float32) / (gray.astype(np.float32) + 1)
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c].astype(np.float32) * ratio, 0, 255).astype(np.uint8)

    return result


def _needs_perspective(img_cv):
    """카메라 촬영 이미지인지 판단.

    카메라 사진: 가장자리(책상/배경)가 중앙(종이)보다 어두움
    스캔 이미지: 가장자리(여백)가 중앙(내용)보다 밝음
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    margin = 20
    border = np.concatenate([
        blur[:margin, :].flatten(), blur[-margin:, :].flatten(),
        blur[:, :margin].flatten(), blur[:, -margin:].flatten()
    ])
    center = blur[h // 4:3 * h // 4, w // 4:3 * w // 4].flatten()
    border_mean = float(border.mean())
    center_mean = float(center.mean())
    contrast = abs(center_mean - border_mean)
    # 카메라 사진: 가장자리가 어둡고 중앙이 밝음 (center > border)
    is_camera = center_mean > border_mean and contrast > 12
    return contrast, is_camera


def mod_perspective(img_cv, **kwargs):
    """[perspective 모듈] 용지 검출 → 원근 보정 → 스캔 효과.

    카메라로 촬영한 이미지(배경 대비 > 12)만 perspective 보정 적용.
    이미 스캔된 이미지는 보정 불필요.
    """
    contrast, needs = _needs_perspective(img_cv)
    if not needs:
        print("    스캔 이미지 감지 (대비={:.0f}), perspective 스킵".format(contrast))
        return img_cv

    print("    카메라 촬영 감지 (대비={:.0f}), 용지 검출 중...".format(contrast))

    contour = find_document_contour(img_cv)
    if contour is None:
        print("    용지 윤곽 감지 실패, 스캔 효과만 적용")
        return scan_effect(img_cv)

    h, w = img_cv.shape[:2]
    area_ratio = cv2.contourArea(contour) / (h * w)
    print("    용지 감지: 면적비 {:.1%}".format(area_ratio))

    if area_ratio < 0.2:
        print("    용지 너무 작음, 스캔 효과만 적용")
        return scan_effect(img_cv)

    warped = perspective_transform(img_cv, contour)
    result = scan_effect(warped)
    h2, w2 = result.shape[:2]
    print("    원근 보정 완료: {}x{}".format(w2, h2))
    return result


# ============================================================
# 모듈 2: rotate — 90도 단위 회전 보정
# ============================================================

def ocr_confidence(img_cv):
    """tesseract OCR confidence 평균. 속도를 위해 긴 변 1200px로 축소."""
    h, w = img_cv.shape[:2]
    MAX_OCR_SIZE = 1200
    if max(h, w) > MAX_OCR_SIZE:
        scale = MAX_OCR_SIZE / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)
    tmp = tempfile.mktemp(suffix=".png")
    cv2.imwrite(tmp, img_cv)
    try:
        result = subprocess.run(
            [TESSERACT_CMD, tmp, "stdout", "-l", "kor", "--psm", "4", "tsv"],
            capture_output=True, text=True, timeout=60
        )
    finally:
        os.unlink(tmp)

    confs = []
    for line in result.stdout.split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 12:
            try:
                c = float(parts[10])
                text = parts[11].strip()
                if c > 0 and len(text) > 0:
                    confs.append(c)
            except (ValueError, IndexError):
                pass

    return (float(np.mean(confs)) if confs else 0.0, len(confs))


def mod_rotate(img_cv, **kwargs):
    """[rotate 모듈] 4방향 OCR confidence 비교로 최적 회전 찾기.

    단순 confidence 평균이 아니라 conf × words (인식 총점)를 기준으로 판단.
    진료비영수증처럼 표 헤더가 세로로 적힌 문서에서 90° 오감지를 방지하기 위해,
    0°(원본)를 기본으로 두고 다른 방향이 확실히 우세할 때만 회전 적용.
    """
    rotations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    scores = {}
    for deg, rot_code in rotations:
        rotated = cv2.rotate(img_cv, rot_code) if rot_code else img_cv
        conf, n_words = ocr_confidence(rotated)
        # 인식 총점 = 평균 confidence × 인식된 단어 수
        score = conf * n_words
        scores[deg] = (conf, n_words, score)

    # 최고 점수 방향 찾기
    best_rot = max(scores, key=lambda d: scores[d][2])
    best_score = scores[best_rot][2]
    orig_score = scores[0][2]

    for deg in [0, 90, 180, 270]:
        conf, n_words, score = scores[deg]
        mark = " <-" if deg == best_rot else ""
        print("    {}도: conf={:.1f}, words={}, score={:.0f}{}".format(
            deg, conf, n_words, score, mark))

    # 0°(원본) 대비 20% 이상 우세해야 회전 적용 (오감지 방지)
    ROTATE_THRESHOLD = 1.2
    if best_rot != 0 and best_score > orig_score * ROTATE_THRESHOLD:
        rot_codes = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        img_cv = cv2.rotate(img_cv, rot_codes[best_rot])
        print("    -> {}도 회전 적용 (점수 {:.0f} vs 원본 {:.0f})".format(
            best_rot, best_score, orig_score))
    else:
        if best_rot != 0:
            print("    -> 회전 불필요 (차이 미미: {:.0f} vs {:.0f})".format(
                best_score, orig_score))
        else:
            print("    -> 회전 불필요")

    return img_cv


# ============================================================
# 모듈 3: deskew — 미세 기울기 보정
# ============================================================

def detect_skew_angle(img_cv):
    """jdeskew -> deskew -> hough 순서로 각도 감지."""
    try:
        from jdeskew.estimator import get_angle
        angle = get_angle(img_cv)
        if angle is not None and abs(angle) < 45:
            return angle, "jdeskew"
    except Exception:
        pass

    try:
        from deskew import determine_skew
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(gray)
        if angle is not None and abs(angle) < 45:
            return angle, "deskew"
    except Exception:
        pass

    # hough fallback
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    h, w = gray.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                            minLineLength=w // 8, maxLineGap=w // 20)
    if lines is None or len(lines) < 5:
        return 0.0, "hough"
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 15:
            angles.append(a)
    return (float(np.median(angles)) if angles else 0.0, "hough")


def rotate_image(img, angle):
    """각도만큼 회전. 잘리지 않게 캔버스 확장."""
    if abs(angle) < 0.1:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def mod_deskew(img_cv, **kwargs):
    """[deskew 모듈] 미세 기울기 감지 및 보정."""
    angle, method = detect_skew_angle(img_cv)
    if abs(angle) >= 0.3:
        img_cv = rotate_image(img_cv, angle)
        print("    {:.2f}도 보정 ({})".format(angle, method))
    else:
        print("    보정 불필요 ({:.2f}도)".format(angle))
    return img_cv


# ============================================================
# 모듈 레지스트리
# ============================================================

# ============================================================
# 모듈 4: enhance — SwinIR 기반 화질 개선
# ============================================================

_swinir_model = None


def _load_swinir():
    """SwinIR 모델 로드 (최초 1회)."""
    global _swinir_model
    if _swinir_model is not None:
        return _swinir_model

    import sys
    swinir_path = os.path.join(os.path.dirname(__file__), "swinir_repo", "models")
    if swinir_path not in sys.path:
        sys.path.insert(0, swinir_path)

    from network_swinir import SwinIR
    import torch

    model = SwinIR(
        upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv'
    )

    weights_path = os.path.join(os.path.dirname(__file__), "swinir_weights.pth")
    weights = torch.load(weights_path, map_location='cpu')
    param_key = 'params_ema' if 'params_ema' in weights else 'params'
    model.load_state_dict(weights[param_key], strict=True)
    model.eval()

    _swinir_model = model
    return model


def swinir_upscale(img_cv, scale=4, tile_size=256, tile_overlap=32):
    """SwinIR로 이미지 업스케일 (타일 분할 처리로 메모리 절약)."""
    import torch

    model = _load_swinir()
    h, w = img_cv.shape[:2]
    window_size = 8

    # 작은 이미지는 통째로 처리
    if h <= tile_size and w <= tile_size:
        img_t = img_cv.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).unsqueeze(0)
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.no_grad():
            output = model(img_t)
        output = output[:, :, :h * scale, :w * scale]
        output = output.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        return (output * 255).astype(np.uint8)

    # 타일 분할 처리
    output = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    weight = np.zeros((h * scale, w * scale, 1), dtype=np.float32)

    step = tile_size - tile_overlap
    n_tiles = 0

    for y in range(0, h, step):
        for x in range(0, w, step):
            # 타일 좌표 (입력)
            y1 = min(y, h - tile_size) if y + tile_size > h else y
            x1 = min(x, w - tile_size) if x + tile_size > w else x
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)

            tile = img_cv[y1:y2, x1:x2]
            th, tw = tile.shape[:2]

            tile_t = tile.astype(np.float32) / 255.0
            tile_t = torch.from_numpy(tile_t.transpose(2, 0, 1)).unsqueeze(0)
            pad_h = (window_size - th % window_size) % window_size
            pad_w = (window_size - tw % window_size) % window_size
            tile_t = torch.nn.functional.pad(tile_t, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                out_tile = model(tile_t)

            out_tile = out_tile[:, :, :th * scale, :tw * scale]
            out_tile = out_tile.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

            # 출력 좌표
            oy1, ox1 = y1 * scale, x1 * scale
            oy2, ox2 = oy1 + th * scale, ox1 + tw * scale

            output[oy1:oy2, ox1:ox2] += out_tile
            weight[oy1:oy2, ox1:ox2] += 1.0
            n_tiles += 1

    print("      타일 {}개 처리".format(n_tiles))
    output /= np.maximum(weight, 1.0)
    return (output * 255).astype(np.uint8)


def _estimate_text_height(gray):
    """글자의 대략적인 높이(px)를 추정. adaptive threshold 후 컴포넌트 분석.

    표 선분(가로/세로로 긴 얇은 컨투어)과 노이즈를 필터링하여
    실제 글자 컴포넌트만의 높이를 추정한다.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    # 작은 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    heights = []
    h_img, w_img = gray.shape
    for cnt in contours:
        _, _, cw, ch = cv2.boundingRect(cnt)
        # 너무 작거나(노이즈) 너무 큰(테두리/표) 것은 제외
        if ch < 8 or ch > h_img * 0.15:
            continue
        if cw < 3 or cw > w_img * 0.5:
            continue
        # 표 선분 필터링: 가로로 매우 긴 얇은 것 (aspect ratio > 15)
        aspect = cw / max(ch, 1)
        if aspect > 15:
            continue
        # 세로로 매우 긴 얇은 것 (세로선)
        aspect_v = ch / max(cw, 1)
        if aspect_v > 15:
            continue
        # 글자는 보통 정사각형에 가까운 비율 (aspect 0.2~8 정도)
        heights.append(ch)

    if len(heights) < 10:
        return None
    return float(np.median(heights))


def mod_shadow_remove(img_cv, **kwargs):
    """[shadow_remove 모듈] 그림자/조명 불균일 제거.

    Division-based normalization: 원본을 배경 조명 추정치로 나누어
    균일한 조명을 만든 뒤, 대비를 복원한다.
    조명 불균일도가 높을 때만 적용 (std of local brightness > 임계값).
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 조명 불균일도 측정: 블록별 평균 밝기의 표준편차
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    block_h, block_w = h // 4, w // 4
    block_means = []
    for i in range(4):
        for j in range(4):
            block = blur[i * block_h:(i + 1) * block_h,
                         j * block_w:(j + 1) * block_w]
            block_means.append(block.mean())
    illumination_std = float(np.std(block_means))

    if illumination_std < 8:
        print("    조명 균일 (std={:.1f}), 그림자 제거 불필요".format(illumination_std))
        return img_cv

    print("    조명 불균일 감지 (std={:.1f}), 그림자 제거 적용".format(illumination_std))

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. 배경 조명 추정 (morphological closing + blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (101, 101), 0)

    # 2. Division normalization
    normalized = (gray / (bg + 1.0)) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # 3. 대비 스트레칭 (1-99 percentile)
    p1, p99 = np.percentile(normalized, [1, 99])
    stretched = np.clip(
        (normalized.astype(np.float32) - p1) / (p99 - p1 + 1) * 255,
        0, 255
    ).astype(np.uint8)

    # 4. CLAHE (적응형 히스토그램 균등화)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(stretched)

    # 5. 원본 색조 유지하면서 밝기만 교체
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = enhanced
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    print("    그림자 제거 완료")
    return result


def mod_enhance(img_cv, **kwargs):
    """[enhance 모듈] 저화질 문서 이미지 → OCR 최적화.

    전략:
      - 글자 높이 < 20px (저화질) → SwinIR 4x 업스케일 후 OCR 최적 크기로 다운스케일
      - 글자 높이 20~60px → INTER_CUBIC으로 미세 조정만
      - 글자 높이 > 60px → 다운스케일
    어떤 경우든 최종 글자 높이를 ~40px로 맞추고, 문서용 후처리 적용.
    """
    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 1. 글자 높이 추정
    TARGET_TEXT_HEIGHT = 40.0
    text_h = _estimate_text_height(gray)

    if text_h and text_h > 0:
        print("    글자 높이 추정: {:.1f}px".format(text_h))
    else:
        # 추정 실패 시 긴 변 기준으로 추정 (A4 300DPI 기준 글자 ~30px)
        text_h = 30.0 * max(h, w) / 3500.0
        print("    글자 높이 추정 실패, 해상도 기반 추정: {:.1f}px".format(text_h))

    # 2. 업스케일 전략 결정
    # 이미 고해상도(긴 변 > 2000px)인데 글자 높이가 작게 추정된 경우는
    # 추정 오류일 가능성이 높으므로 SwinIR 스킵 + 리사이즈도 제한
    long_side = max(h, w)
    use_swinir = text_h < 20 and long_side < 1200  # 진짜 저화질(소형 이미지)만 SwinIR
    if text_h < 20 and long_side >= 1200:
        print("    고해상도 이미지({}px) → SwinIR 스킵, 후처리만 적용".format(long_side))
        # 글자 높이 추정이 부정확할 가능성 → 리사이즈 억제
        text_h = TARGET_TEXT_HEIGHT  # 스케일 1.0으로 고정
    final_scale = TARGET_TEXT_HEIGHT / text_h
    final_scale = max(0.5, min(4.0, final_scale))

    if use_swinir:
        # 저화질: SwinIR로 디테일 복원
        # 전략: 입력을 축소 → SwinIR 4x → 최종 목표 크기 도달
        # 예) 글자 10px, 이미지 1728x2333
        #     목표 글자 40px → 최종 스케일 4x
        #     SwinIR 입력 크기 = 최종 / 4 = 원본과 동일 (축소 불필요)
        #     SwinIR 입력 크기가 max 1024px 이하가 되도록 축소
        print("    저화질 감지 → SwinIR 업스케일 + 다운스케일 전략")

        # SwinIR 입력 크기 결정:
        # 목표: text_h * pre_scale * 4 = TARGET_TEXT_HEIGHT
        # → pre_scale = TARGET / (text_h * 4)
        # 추가 제약: SwinIR 입력 긴 변 max 1024px (메모리)
        SWINIR_MAX_INPUT = 1024
        ideal_pre_scale = TARGET_TEXT_HEIGHT / (text_h * 4)
        pre_scale = min(ideal_pre_scale, 1.0, SWINIR_MAX_INPUT / max(h, w))
        if pre_scale < 0.95:
            sr_input = cv2.resize(img_cv, (int(w * pre_scale), int(h * pre_scale)),
                                  interpolation=cv2.INTER_AREA)
            print("    SwinIR 입력 축소: {}x{} -> {}x{}".format(
                w, h, sr_input.shape[1], sr_input.shape[0]))
        else:
            sr_input = img_cv

        # SwinIR 전 경량 노이즈 제거 (SR 품질 향상)
        sr_input = cv2.fastNlMeansDenoisingColored(sr_input, None, 6, 6, 7, 21)

        print("    SwinIR x4 업스케일 중...")
        upscaled = swinir_upscale(sr_input, scale=4)
        h_up, w_up = upscaled.shape[:2]
        print("    {}x{} -> {}x{}".format(sr_input.shape[1], sr_input.shape[0], w_up, h_up))

        # 최종 목표: 원본 대비 final_scale 배율, 단 긴 변 max 3500px
        MAX_OUTPUT = 3500
        target_long = int(max(h, w) * final_scale)
        if target_long > MAX_OUTPUT:
            final_scale = MAX_OUTPUT / max(h, w)

        target_w = int(w * final_scale)
        target_h = int(h * final_scale)
        out_scale = target_w / w_up  # SwinIR 출력 → 최종 목표

        img_cv = cv2.resize(upscaled, (target_w, target_h),
                            interpolation=cv2.INTER_AREA if out_scale < 1.0 else cv2.INTER_CUBIC)
        print("    최종 리사이즈: {}x{} -> {}x{}".format(w_up, h_up, target_w, target_h))

        h, w = img_cv.shape[:2]
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    elif abs(final_scale - 1.0) > 0.1:
        # 중간/고해상도: INTER_CUBIC으로 크기만 조정
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        interp = cv2.INTER_CUBIC if final_scale > 1.0 else cv2.INTER_AREA
        img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=interp)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        print("    리사이즈: {}x{} -> {}x{} ({})"
              .format(w, h, new_w, new_h, "확대" if final_scale > 1 else "축소"))
        h, w = new_h, new_w
    else:
        print("    리사이즈 불필요 (글자 크기 적정)")

    # 3. 문서 후처리 (공통)
    # 경량 노이즈 제거
    denoised_gray = cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)

    # 배경 정규화 (조명 불균일 보정)
    bg = cv2.GaussianBlur(denoised_gray, (51, 51), 0)
    normalized = cv2.divide(denoised_gray, bg, scale=255)

    # 대비 강화 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # 컬러 복원
    ratio = enhanced.astype(np.float32) / (gray.astype(np.float32) + 1)
    result = img_cv.copy()
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c].astype(np.float32) * ratio, 0, 255).astype(np.uint8)

    print("    enhance 완료: {}x{}".format(w, h))
    return result


# ============================================================
# 모듈 5: crop — 외부 경계(검은/회색 테두리) 자동 제거
# ============================================================

def mod_crop(img_cv, **kwargs):
    """[crop 모듈] 이미지 외곽의 불필요한 테두리를 잘라냄."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 밝기 기반: 내용이 있는 영역 찾기
    # 배경(테두리)은 매우 어둡거나(<30) 매우 밝음(>240)
    # 내용 영역은 그 사이
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 행/열별 흰 픽셀 비율로 내용 영역 판별
    row_white = np.mean(thresh > 0, axis=1)
    col_white = np.mean(thresh > 0, axis=1)

    # 수평 방향: 각 열의 내용 비율
    col_content = np.mean(thresh > 0, axis=0)
    # 수직 방향: 각 행의 내용 비율
    row_content = np.mean(thresh > 0, axis=1)

    # 내용이 5% 이상인 영역 찾기
    content_threshold = 0.05
    rows_with_content = np.where(row_content > content_threshold)[0]
    cols_with_content = np.where(col_content > content_threshold)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        print("    크롭 불필요 (내용 영역 감지 실패)")
        return img_cv

    y1 = max(0, rows_with_content[0] - 5)
    y2 = min(h, rows_with_content[-1] + 5)
    x1 = max(0, cols_with_content[0] - 5)
    x2 = min(w, cols_with_content[-1] + 5)

    # 크롭 비율이 너무 작으면 (10% 이상 잘리면) 적용
    crop_ratio = (y2 - y1) * (x2 - x1) / (h * w)
    if crop_ratio > 0.95:
        print("    크롭 불필요 (테두리 없음)")
        return img_cv

    cropped = img_cv[y1:y2, x1:x2]
    print("    크롭: {}x{} -> {}x{} ({:.1%} 유지)".format(w, h, x2 - x1, y2 - y1, crop_ratio))
    return cropped



def mod_dewarp(img_cv, **kwargs):
    """[dewarp 모듈] 구겨진/왜곡된 문서를 펴기 (UVDoc ML 모델).

    venv_paddle의 PaddleOCR UVDoc을 서브프로세스로 호출.
    카메라 촬영 이미지에만 적용.
    """
    contrast, is_camera = _needs_perspective(img_cv)
    if not is_camera:
        print("    스캔 이미지 → dewarp 불필요")
        return img_cv

    script_dir = os.path.dirname(os.path.abspath(__file__))
    paddle_python = os.path.join(script_dir, "venv_paddle", "bin", "python3")
    dewarp_script = os.path.join(script_dir, "uvdoc_dewarp.py")

    if not os.path.exists(paddle_python):
        print("    venv_paddle 없음, dewarp 스킵")
        return img_cv

    # 임시 파일로 입출력
    tmp_in = tempfile.mktemp(suffix=".png")
    tmp_out = tempfile.mktemp(suffix=".png")
    cv2.imwrite(tmp_in, img_cv)

    try:
        result = subprocess.run(
            [paddle_python, dewarp_script, tmp_in, tmp_out],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(tmp_out):
            dewarped = cv2.imread(tmp_out)
            if dewarped is not None:
                h, w = dewarped.shape[:2]
                print("    UVDoc dewarp 완료: {}x{}".format(w, h))
                return dewarped
        print("    dewarp 실패: {}".format(result.stderr.strip()[:100]))
    except subprocess.TimeoutExpired:
        print("    dewarp 타임아웃")
    finally:
        for f in [tmp_in, tmp_out]:
            if os.path.exists(f):
                os.unlink(f)

    return img_cv


def mod_restore(img_cv, **kwargs):
    """[restore 모듈] TAIR 기반 텍스트 인식 글씨 복원 (Diffusion).

    venv_tair의 TAIR(TeReDiff)를 서브프로세스로 호출.
    저화질/흐릿한 문서의 글씨를 선명하게 복원.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tair_python = os.path.join(script_dir, "venv_tair", "bin", "python3")
    tair_script = os.path.join(script_dir, "tair_restore.py")

    if not os.path.exists(tair_python):
        print("    venv_tair 없음, restore 스킵")
        return img_cv

    tmp_in = tempfile.mktemp(suffix=".png")
    tmp_out = tempfile.mktemp(suffix=".png")
    cv2.imwrite(tmp_in, img_cv)

    try:
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        result = subprocess.run(
            [tair_python, tair_script, tmp_in, tmp_out, "--steps", "15"],
            capture_output=True, text=True, timeout=300, env=env
        )
        if result.returncode == 0 and os.path.exists(tmp_out):
            restored = cv2.imread(tmp_out)
            if restored is not None:
                h, w = restored.shape[:2]
                print("    TAIR 글씨복원 완료: {}x{}".format(w, h))
                return restored
        print("    restore 실패: {}".format(result.stderr.strip()[-200:]))
    except subprocess.TimeoutExpired:
        print("    restore 타임아웃 (300초 초과)")
    finally:
        for f in [tmp_in, tmp_out]:
            if os.path.exists(f):
                os.unlink(f)

    return img_cv


MODULES = {
    "perspective": mod_perspective,
    "rotate": mod_rotate,
    "deskew": mod_deskew,
    "dewarp": mod_dewarp,
    "shadow_remove": mod_shadow_remove,
    "restore": mod_restore,
    "enhance": mod_enhance,
    "crop": mod_crop,
}


# ============================================================
# 이미지 로드/저장
# ============================================================

def load_image(path):
    pil_img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def save_image(img, path):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)


# ============================================================
# 메인 파이프라인
# ============================================================

def process_image(input_path, output_path=None, steps=None):
    if steps is None:
        steps = ["rotate", "deskew"]

    print("=" * 50)
    print("처리: " + os.path.basename(input_path))
    print("단계: " + " -> ".join(steps))

    img = load_image(input_path)
    h, w = img.shape[:2]
    print("  크기: {}x{}".format(w, h))

    for i, step_name in enumerate(steps, 1):
        if step_name not in MODULES:
            print("  [{}] {} — 알 수 없는 모듈, 건너뜀".format(i, step_name))
            continue
        print("  [{}] {}".format(i, step_name))
        img = MODULES[step_name](img)
        h, w = img.shape[:2]
        print("    결과: {}x{}".format(w, h))

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_보정" + ext

    save_image(img, output_path)
    print("  저장: " + output_path)

    # OCR 결과 txt 저장
    txt_path = os.path.splitext(output_path)[0] + ".txt"
    try:
        tmp = tempfile.mktemp(suffix=".png")
        cv2.imwrite(tmp, img)
        result = subprocess.run(
            [TESSERACT_CMD, tmp, "stdout", "-l", "kor", "--psm", "6"],
            capture_output=True, text=True, timeout=120
        )
        os.unlink(tmp)
        ocr_text = result.stdout.strip()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        line_count = len([l for l in ocr_text.split("\n") if l.strip()])
        print("  OCR: {}줄 -> {}".format(line_count, os.path.basename(txt_path)))
    except Exception as e:
        print("  OCR 실패: {}".format(e))

    return img


def main():
    all_steps = ", ".join(MODULES.keys())
    parser = argparse.ArgumentParser(
        description="진료비영수증 이미지 보정 (모듈식)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 가능한 모듈: {}

예시:
  python line_recover.py image.tif --steps perspective,rotate,deskew
  python line_recover.py image.tif --steps rotate,deskew
  python line_recover.py image.tif --steps perspective
  python line_recover.py /폴더/ --batch --steps rotate,deskew
        """.format(all_steps)
    )
    parser.add_argument("input", help="입력 이미지 또는 폴더")
    parser.add_argument("-o", "--output", help="출력 경로")
    parser.add_argument("--batch", action="store_true", help="폴더 일괄 처리")
    parser.add_argument("--steps", default="rotate,deskew",
                        help="처리 단계 (쉼표 구분, 순서대로 실행). 기본: rotate,deskew")

    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",")]

    if args.batch or os.path.isdir(args.input):
        files = []
        for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
            files.extend(glob.glob(os.path.join(args.input, ext)))
        if not files:
            print("파일 없음: " + args.input)
            return
        out_dir = os.path.join(args.input, "결과")
        os.makedirs(out_dir, exist_ok=True)
        print("총 {}개 -> {}/\n".format(len(files), out_dir))
        for f in sorted(files):
            try:
                bn = os.path.basename(f)
                name, fext = os.path.splitext(bn)
                out_path = os.path.join(out_dir, name + "_보정" + fext)
                process_image(f, output_path=out_path, steps=steps)
            except Exception as e:
                print("  오류: " + str(e))
            print()
    else:
        process_image(args.input, output_path=args.output, steps=steps)


if __name__ == "__main__":
    main()
