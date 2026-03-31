#!/usr/bin/env python3
"""한글 진료비영수증 합성 데이터셋 생성기 (TAIR 호환 포맷).

TAIR 학습용 두 가지 데이터셋 생성:
  1. sa_text 포맷 (학습용): HQ 512x512 이미지 + restoration_dataset.json
     - LQ는 TAIR의 RealESRGAN 파이프라인이 학습 중 on-the-fly 생성
  2. real_text 포맷 (검증용): HQ/LQ 쌍 + real_benchmark_dataset.json

사용법:
  python3 generate_dataset.py --count 5000 --output dataset/
  python3 generate_dataset.py --count 100 --output dataset/ --mode val
"""
import os
import json
import random
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 한글 텍스트 소스 (실제 진료비영수증 데이터)
# ============================================================

# 실제 한국인 이름 (세대별, 통계청 기준 빈도 높은 성씨+이름)
KOREAN_NAMES = [
    # 노년층 (1940~1960년대)
    "김영수", "이순자", "박영자", "최정호", "정옥순", "장복남",
    "윤덕배", "강금순", "조점례", "한영식", "임순례", "오말순",
    "김순옥", "이영희", "박정수", "최영남", "정순덕", "송명자",
    # 중장년층 (1970~1980년대)
    "김지훈", "이은주", "박민지", "최성호", "정혜진", "장수진",
    "윤동현", "강지영", "조미선", "한정훈", "임유진", "오은정",
    "서성민", "류미영", "황지현", "신동준", "고은혜", "문소연",
    "권혁준", "안소영", "송진우", "전미숙",
    # 청년층 (1990~2000년대)
    "김민준", "이서연", "박지민", "최준혁", "정하은", "장예나",
    "윤도현", "강다은", "조유나", "한건우", "임채원", "오예린",
    "서시우", "류나은", "황윤서", "신수빈", "안지호", "전예진",
    # 소아/청소년 (2010년대~)
    "김이준", "이서아", "박하윤", "최도윤", "정지안", "장아린",
    "윤하준", "강서윤", "조지유", "한은우", "임하린", "오이서",
    "신선우", "고채아", "문서하",
]

# 실제 병원명
HOSPITAL_NAMES = [
    "서울대학교병원", "세브란스병원", "강남세브란스병원", "서울아산병원",
    "삼성서울병원", "서울성모병원", "고려대학교병원", "고려대학교구로병원",
    "건국대학교병원", "경희대학교병원", "한양대학교병원", "중앙대학교병원",
    "이화여자대학교목동병원", "강북삼성병원", "분당서울대학교병원",
    "아주대학교병원", "인하대학교병원", "가톨릭대학교인천성모병원",
    "순천향대학교부천병원", "부산대학교병원", "동아대학교병원",
    "경북대학교병원", "영남대학교병원", "전남대학교병원", "조선대학교병원",
    "충남대학교병원", "강원대학교병원", "원주세브란스기독병원",
    "울산대학교병원",
    "연세내과의원", "서울정형외과의원", "한마음소아청소년과의원",
    "미래피부과의원", "튼튼이비인후과의원", "밝은안과의원",
    "사랑가정의학과의원", "건강신경외과의원", "오늘산부인과의원",
    "참좋은내과의원", "우리들정형외과의원", "맑은피부과의원",
]

# 진료과목 (의료법 시행규칙 기준)
DEPARTMENTS = [
    "내과", "외과", "정형외과", "신경외과", "신경과", "피부과",
    "이비인후과", "안과", "산부인과", "소아청소년과", "비뇨의학과",
    "정신건강의학과", "재활의학과", "가정의학과", "응급의학과",
    "심장혈관흉부외과", "성형외과", "마취통증의학과", "영상의학과",
    "진단검사의학과", "방사선종양학과", "직업환경의학과",
]

# KCD 질병분류코드
KCD_CODES = [
    ("I10", "본태성 고혈압"), ("I20", "협심증"), ("I25", "만성 허혈성 심장질환"),
    ("I63", "뇌경색"), ("E11", "제2형 당뇨병"), ("E78.0", "순수 고콜레스테롤혈증"),
    ("J00", "급성 비인두염"), ("J06.9", "급성 상기도 감염"),
    ("J18", "폐렴"), ("J44", "만성 폐쇄성 폐질환"), ("J45", "천식"),
    ("K21", "위식도역류질환"), ("K25", "위궤양"), ("K29", "위염 및 십이지장염"),
    ("K52", "비감염성 위장염 및 대장염"),
    ("M17", "무릎 관절증"), ("M51", "요추 추간판 탈출증"), ("M54.5", "요통"),
    ("M75", "어깨 병변"), ("N39.0", "요로감염"), ("N40", "전립선 비대증"),
    ("F32", "우울증"), ("F41", "불안장애"), ("G20", "파킨슨병"),
    ("L20", "아토피 피부염"), ("L50", "두드러기"),
    ("R10", "복통"), ("R51", "두통"),
    ("C50", "유방의 악성 신생물"), ("C34", "기관지 및 폐의 악성 신생물"),
    ("Z00", "일반 건강검진"), ("Z23", "예방접종"),
]

# 항목별 현실적 금액 범위 (원 단위)
AMOUNT_RANGES = {
    "진찰료":       (13000, 35000),
    "입원료":       (25000, 200000),
    "식대":         (8000, 36000),
    "투약및조제료":  (1000, 10000),
    "주사료":       (3000, 150000),
    "마취료":       (10000, 150000),
    "처치및수술료":  (5000, 1000000),
    "검사료":       (3000, 80000),
    "영상진단료":    (10000, 500000),
    "방사선치료료":  (50000, 300000),
    "치료재료대":    (5000, 200000),
    "혈액대":       (30000, 150000),
    "전산정보관리료": (1000, 5000),
    "제증명료":     (1000, 20000),
}


def random_date():
    y = random.randint(2020, 2026)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return "{}.{:02d}.{:02d}".format(y, m, d)


def random_phone():
    return "0{}-{:04d}-{:04d}".format(
        random.choice([2, 31, 32, 33, 41, 42, 51, 52, 53, 54, 55, 62, 63, 64]),
        random.randint(1000, 9999), random.randint(1000, 9999))


def random_receipt_no():
    fmt = random.choice(["date", "year", "simple"])
    if fmt == "date":
        return "{}{:02d}{:02d}-{:05d}".format(
            random.randint(2020, 2026), random.randint(1, 12),
            random.randint(1, 28), random.randint(1, 99999))
    elif fmt == "year":
        return "{}-{:06d}".format(random.randint(2020, 2026), random.randint(1, 999999))
    else:
        return "{:07d}".format(random.randint(1, 9999999))


def random_patient_no():
    fmt = random.choice(["8digit", "7digit", "year"])
    if fmt == "8digit":
        return "{:08d}".format(random.randint(10000000, 99999999))
    elif fmt == "7digit":
        return "{:07d}".format(random.randint(1000000, 9999999))
    else:
        return "{:02d}-{:06d}".format(random.randint(20, 26), random.randint(1, 999999))


# ============================================================
# 폰트
# ============================================================

FONTS = ["/System/Library/Fonts/AppleSDGothicNeo.ttc"]

def get_font(size, bold=False):
    idx = 5 if bold else 0
    try:
        return ImageFont.truetype(FONTS[0], size, index=idx)
    except Exception:
        return ImageFont.truetype(FONTS[0], size, index=0)


# ============================================================
# 텍스트 bbox 측정 유틸리티
# ============================================================

def get_text_bbox(draw, pos, text, font):
    """텍스트의 bounding box를 (x1, y1, x2, y2) 형태로 반환."""
    bbox = draw.textbbox(pos, text, font=font)
    return [bbox[0], bbox[1], bbox[2], bbox[3]]


def bbox_to_polygon(bbox, n_points=16):
    """bbox [x1,y1,x2,y2]를 n_points개의 polygon 좌표로 변환."""
    x1, y1, x2, y2 = bbox
    # 상단 좌→우, 우측 상→하, 하단 우→좌, 좌측 하→상 순서로 균등 배치
    pts_per_side = n_points // 4
    poly = []
    # 상단
    for i in range(pts_per_side):
        t = i / pts_per_side
        poly.append([x1 + t * (x2 - x1), y1])
    # 우측
    for i in range(pts_per_side):
        t = i / pts_per_side
        poly.append([x2, y1 + t * (y2 - y1)])
    # 하단 (우→좌)
    for i in range(pts_per_side):
        t = i / pts_per_side
        poly.append([x2 - t * (x2 - x1), y2])
    # 좌측 (하→상)
    for i in range(pts_per_side):
        t = i / pts_per_side
        poly.append([x1, y2 - t * (y2 - y1)])
    return poly


# ============================================================
# HQ 문서 이미지 생성 (512x512) + 텍스트 어노테이션 수집
# ============================================================

def draw_text_tracked(draw, pos, text, font, fill, instances):
    """텍스트를 그리면서 bbox/polygon 어노테이션을 수집."""
    draw.text(pos, text, fill=fill, font=font)
    bbox = get_text_bbox(draw, pos, text, font)
    # 유효한 크기의 텍스트만 기록
    if bbox[2] - bbox[0] > 2 and bbox[3] - bbox[1] > 2:
        poly = bbox_to_polygon(bbox, n_points=16)
        instances.append({
            "text": text,
            "bbox": bbox,
            "polygon": poly,
        })


def generate_receipt_hq():
    """진료비영수증 HQ 이미지(512x512) + 텍스트 어노테이션 반환."""
    img = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(img)
    text_instances = []

    y = 8
    # 제목
    font_title = get_font(16, bold=True)
    title = random.choice(["진료비 계산서 · 영수증", "진료비영수증", "외래 진료비 영수증"])
    tx = 512 // 2 - len(title) * 8
    draw_text_tracked(draw, (tx, y), title, font_title, "black", text_instances)
    y += 28

    draw.line([(10, y), (502, y)], fill="black", width=1)
    y += 8

    # 환자 정보
    font_sm = get_font(10)
    font_md = get_font(12)
    dept = random.choice(DEPARTMENTS)
    kcd = random.choice(KCD_CODES)
    patient_name = random.choice(KOREAN_NAMES)
    hospital_name = random.choice(HOSPITAL_NAMES)

    info_lines = [
        ("환자등록번호", random_patient_no()),
        ("환자성명", patient_name),
        ("진료기간", "{} ~ {}".format(random_date(), random_date())),
        ("진료과목", dept),
        ("질병분류코드", "{} ({})".format(kcd[0], kcd[1])),
        ("병실", random.choice(["", "일반병실(6인실)", "4인실", "2인실", "1인실", "중환자실(ICU)"])),
        ("환자구분", random.choice(["국민건강보험", "의료급여", "산재보험", "자동차보험", "일반"])),
        ("영수증번호", random_receipt_no()),
    ]
    for label, value in info_lines:
        line_text = "{}: {}".format(label, value)
        draw_text_tracked(draw, (15, y), line_text, font_sm, "black", text_instances)
        y += 16

    y += 5
    draw.line([(10, y), (502, y)], fill="black", width=2)
    y += 8

    # 표 헤더
    headers = ["항 목", "급여", "비급여", "금액산정내역"]
    col_x = [15, 150, 250, 350]
    for hdr, x in zip(headers, col_x):
        draw_text_tracked(draw, (x, y), hdr, font_md, "black", text_instances)
    y += 18
    draw.line([(10, y), (502, y)], fill="black", width=1)
    y += 4

    # 표 항목
    items = [
        "진찰료", "입원료", "식대", "투약및조제료", "주사료",
        "마취료", "처치및수술료", "검사료", "영상진단료",
        "치료재료대", "혈액대", "전산정보관리료", "제증명료",
    ]
    random.shuffle(items)
    n_items = random.randint(5, min(10, len(items)))
    total_sum = 0
    copay_sum = 0

    for item in items[:n_items]:
        amt_val = random.randint(*AMOUNT_RANGES.get(item, (1000, 100000)))
        amt_val = (amt_val // 100) * 100
        total_sum += amt_val
        amt1 = "{:,}".format(amt_val)

        non_covered = ""
        if random.random() > 0.7:
            nc = random.randint(1000, amt_val // 2 + 1) if amt_val > 2000 else 0
            nc = (nc // 100) * 100
            non_covered = "{:,}".format(nc) if nc > 0 else ""

        copay_rate = random.choice([0.2, 0.3, 0.4, 0.5, 0.6])
        copay = int(amt_val * copay_rate)
        copay = (copay // 100) * 100
        copay_sum += copay
        amt3 = "{:,}".format(copay)

        draw_text_tracked(draw, (15, y), item, font_sm, "black", text_instances)
        draw_text_tracked(draw, (150, y), amt1, font_sm, "black", text_instances)
        if non_covered:
            draw_text_tracked(draw, (250, y), non_covered, font_sm, "black", text_instances)
        draw_text_tracked(draw, (350, y), amt3, font_sm, "black", text_instances)
        y += 15
        draw.line([(10, y), (502, y)], fill=(200, 200, 200), width=1)
        y += 2

    y += 5
    draw.line([(10, y), (502, y)], fill="black", width=2)
    y += 8

    # 합계
    total_str = "{:,}".format(total_sum)
    copay_str = "{:,}".format(copay_sum)
    already_paid = random.choice([0, copay_sum])
    remaining = copay_sum - already_paid
    remain_str = "{:,}".format(remaining)

    draw_text_tracked(draw, (15, y), "진료비총액", font_md, "black", text_instances)
    draw_text_tracked(draw, (350, y), total_str, font_md, "black", text_instances)
    y += 20
    draw_text_tracked(draw, (15, y), "환자부담총액", font_md, "black", text_instances)
    draw_text_tracked(draw, (350, y), copay_str, font_md, "black", text_instances)
    y += 20
    draw_text_tracked(draw, (15, y), "납부할금액", font_md, "black", text_instances)
    draw_text_tracked(draw, (350, y), remain_str, font_md, "black", text_instances)
    y += 25

    draw.line([(10, y), (502, y)], fill="black", width=1)
    y += 10

    # 하단 정보
    font_xs = get_font(9)
    footer1 = "발행일: {}".format(random_date())
    footer2 = "{}  전화: {}".format(hospital_name, random_phone())
    footer3 = "※ 이 영수증은 소득공제용으로 사용할 수 있습니다."
    draw_text_tracked(draw, (15, y), footer1, font_xs, "gray", text_instances)
    y += 14
    draw_text_tracked(draw, (15, y), footer2, font_xs, "gray", text_instances)
    y += 14
    draw_text_tracked(draw, (15, y), footer3, font_xs, "gray", text_instances)

    # 프롬프트 생성 (TAIR 형식)
    prompt = "한국어 진료비영수증 문서. 환자명 {}, {}, {} 진료.".format(
        patient_name, dept, hospital_name)

    return img, text_instances, prompt


# ============================================================
# LQ 열화 (검증용 real_text 데이터셋)
# ============================================================

def degrade_image(hq_img, severity="random"):
    """HQ→LQ 열화. 텍스트가 육안으로 식별 가능한 수준."""
    img = np.array(hq_img).astype(np.float32)

    if severity == "random":
        severity = random.choices(
            ["mild", "moderate", "severe"],
            weights=[0.5, 0.35, 0.15], k=1)[0]

    # 1. 약한 기울기 (30% 확률, ±1.5도)
    if random.random() > 0.7:
        angle = random.uniform(-1.5, 1.5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    # 2. 조명 불균일 (25% 확률)
    if random.random() > 0.75:
        h, w = img.shape[:2]
        direction = random.choice(["h", "v"])
        if direction == "h":
            gradient = np.linspace(0.88, 1.0, w).reshape(1, -1).repeat(h, axis=0)
        else:
            gradient = np.linspace(0.88, 1.0, h).reshape(-1, 1).repeat(w, axis=1)
        img = img * gradient[:, :, np.newaxis]

    # 3. 가벼운 블러
    k = {"mild": 3, "moderate": 3, "severe": 5}[severity]
    img = cv2.GaussianBlur(img, (k, k), 0)

    # 4. 미세 노이즈
    noise_std = {"mild": 2, "moderate": 3, "severe": 5}[severity]
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    # 5. JPEG 압축
    quality = {"mild": 88, "moderate": 72, "severe": 55}[severity]
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)

    # 6. 축소 (512 유지 — TAIR는 512x512 입력)
    # real_text LQ도 512x512로 유지
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# ============================================================
# TAIR 어노테이션 포맷 변환
# ============================================================

def build_annotation(text_instances, img_size=512):
    """텍스트 인스턴스를 TAIR restoration_dataset.json 포맷으로 변환."""
    ann_instances = []
    for inst in text_instances:
        ann_instances.append({
            "text": inst["text"],
            "bbox": inst["bbox"],  # [x1, y1, x2, y2]
            "polygon": inst["polygon"],  # 16x2
        })
    return {"0": {"text_instances": ann_instances}}


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="한글 진료비영수증 TAIR 호환 데이터셋 생성")
    parser.add_argument("--count", type=int, default=5000, help="생성 개수 (기본: 5000)")
    parser.add_argument("--output", default="dataset", help="출력 디렉토리")
    parser.add_argument("--mode", default="train", choices=["train", "val"],
                        help="train: sa_text 포맷 (HQ만), val: real_text 포맷 (HQ+LQ)")
    args = parser.parse_args()

    annotations = {}

    if args.mode == "train":
        # sa_text 포맷: images/ + restoration_dataset.json
        img_dir = os.path.join(args.output, "images")
        os.makedirs(img_dir, exist_ok=True)
        print("TAIR sa_text 포맷 생성 (학습용)")
        print("이미지: {}/".format(img_dir))
        print("LQ는 TAIR 학습 파이프라인이 on-the-fly 생성합니다.")
    else:
        # real_text 포맷: HQ/ + LQ/ + real_benchmark_dataset.json
        hq_dir = os.path.join(args.output, "HQ")
        lq_dir = os.path.join(args.output, "LQ")
        os.makedirs(hq_dir, exist_ok=True)
        os.makedirs(lq_dir, exist_ok=True)
        print("TAIR real_text 포맷 생성 (검증용)")
        print("HQ: {}/".format(hq_dir))
        print("LQ: {}/".format(lq_dir))

    print("생성 시작: {}개".format(args.count))

    for i in range(args.count):
        img_id = "kr_{:06d}".format(i)

        # HQ 생성 + 어노테이션 수집
        hq, text_instances, prompt = generate_receipt_hq()

        if args.mode == "train":
            hq.save(os.path.join(img_dir, img_id + ".jpg"), quality=95)
        else:
            hq.save(os.path.join(hq_dir, img_id + ".jpg"), quality=95)
            lq = degrade_image(hq)
            lq.save(os.path.join(lq_dir, img_id + ".jpg"), quality=90)

        # 어노테이션 저장
        ann = build_annotation(text_instances)
        ann["0"]["prompt"] = prompt
        annotations[img_id] = ann

        if (i + 1) % 100 == 0:
            print("  {}/{}".format(i + 1, args.count))

    # JSON 어노테이션 저장
    if args.mode == "train":
        json_path = os.path.join(args.output, "restoration_dataset.json")
    else:
        json_path = os.path.join(args.output, "real_benchmark_dataset.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print("완료: {} 이미지 + {} 생성".format(args.count, os.path.basename(json_path)))
    print("어노테이션: {}".format(json_path))


if __name__ == "__main__":
    main()
