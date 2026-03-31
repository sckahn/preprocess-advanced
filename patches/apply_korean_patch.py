#!/usr/bin/env python3
"""TAIR 한글 지원 패치 적용 스크립트.

tair_repo/terediff/dataset/utils.py를 수정하여:
  1. CTLABELS에 한글 완성형 음절 (가~힣) 추가
  2. encode()/decode()를 한글 호환으로 교체
  3. load_file_list()에 korean_receipt 데이터셋 핸들러 추가

사용법:
  python3 patches/apply_korean_patch.py
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
UTILS_PATH = os.path.join(PROJECT_DIR, "tair_repo", "terediff", "dataset", "utils.py")

# 백업
if not os.path.exists(UTILS_PATH):
    print("Error: {} not found".format(UTILS_PATH))
    print("먼저 tair_repo를 클론하세요:")
    print("  git clone https://github.com/cvlab-kaist/TAIR.git tair_repo")
    sys.exit(1)

backup_path = UTILS_PATH + ".bak"
if not os.path.exists(backup_path):
    import shutil
    shutil.copy2(UTILS_PATH, backup_path)
    print("백업 생성: {}".format(backup_path))

# 패치된 utils.py 내용
PATCHED_CONTENT = r'''import random
import math
import os
import json
import numpy as np
from PIL import Image
import cv2
import torch
from .diffjpeg import DiffJPEG
from torch.nn import functional as F


# ============================================================
# 한글 + ASCII 통합 문자 인코딩
# ============================================================

# 원본 ASCII 라벨 (95자)
_ASCII_LABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

# 한글 완성형 음절 (가~힣, 11172자)
_HANGUL_SYLLABLES = [chr(c) for c in range(0xAC00, 0xD7A4)]
_KOREAN_EXTRA = ['\u00b7', '~', '\u203b', '\u2460', '\u2461', '\u2462', '\u2463', '\u2464', '\u20a9']

# 통합 라벨 테이블
CTLABELS = _ASCII_LABELS + _HANGUL_SYLLABLES + _KOREAN_EXTRA

# char -> index 빠른 룩업
_CHAR2IDX = {ch: i for i, ch in enumerate(CTLABELS)}

# EOS 토큰 인덱스
_EOS_IDX = len(CTLABELS)

_MAX_WORD_LEN = 25


def decode(idxs):
    s = ''
    for idx in idxs:
        if idx < len(CTLABELS):
            s += CTLABELS[idx]
        else:
            return s
    return s


def encode(word):
    s = []
    for i in range(_MAX_WORD_LEN):
        if i < len(word):
            char = word[i]
            idx = _CHAR2IDX.get(char)
            if idx is not None:
                s.append(idx)
            else:
                s.append(_EOS_IDX)
        else:
            s.append(_EOS_IDX)
    return s


def load_file_list(file_list_path: str, data_args=None):

    mode = data_args['mode']
    datasets = data_args['datasets']
    ann_path = data_args['ann_path']
    model_H, model_W = data_args['model_img_size']

    files = []
    for dataset in datasets:

        if dataset == 'korean_receipt':
            # 한글 진료비영수증 데이터셋 로더
            json_path = ann_path
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                json_data = sorted(json_data.items())

            split_index = int(len(json_data) * 10 / 11)
            if mode == 'TRAIN':
                json_data = dict(json_data[:split_index])
            elif mode == 'VAL':
                json_data = dict(json_data[split_index:])

            imgs_path = f'{file_list_path}/images'
            imgs = sorted(os.listdir(imgs_path))

            for img in imgs:
                gt_path = f'{imgs_path}/{img}'
                img_id = img.split('.')[0]
                if img_id not in json_data:
                    continue

                img_ann = json_data[img_id]['0']['text_instances']
                prompt = json_data[img_id]['0'].get('prompt', '')

                boxes = []
                texts = []
                text_encs = []
                polys = []

                for ann in img_ann:
                    text = ann['text']
                    if len(text) == 0 or len(text) > _MAX_WORD_LEN:
                        continue
                    if all(ch in _CHAR2IDX for ch in text):
                        texts.append(text)
                        text_encs.append(encode(text))
                    else:
                        filtered = ''.join(ch for ch in text if ch in _CHAR2IDX)
                        if len(filtered) > 0:
                            texts.append(filtered)
                            text_encs.append(encode(filtered))
                        else:
                            continue

                    box_xyxy = ann['bbox']
                    x1, y1, x2, y2 = box_xyxy
                    box_xyxy_scaled = [x1/model_W, y1/model_H, x2/model_W, y2/model_H]
                    sx1, sy1, sx2, sy2 = box_xyxy_scaled
                    box_cxcywh = [(sx1+sx2)/2, (sy1+sy2)/2, sx2-sx1, sy2-sy1]
                    processed_box = [round(v, 4) for v in box_cxcywh]
                    boxes.append(processed_box)

                    poly = np.array(ann['polygon']).astype(np.float64)
                    poly_scaled = poly / np.array([model_W, model_H])
                    polys.append(poly_scaled)

                assert len(boxes) == len(texts) == len(text_encs) == len(polys), "Check loader!"

                if len(boxes) == 0 or len(polys) == 0:
                    continue

                if not prompt:
                    caption = [f'"{txt}"' for txt in texts[:5]]
                    prompt = f"Korean medical receipt document with texts: {', '.join(caption)}"

                files.append({
                    "image_path": gt_path,
                    "prompt": prompt,
                    "text": texts,
                    "bbox": boxes,
                    "poly": polys,
                    "text_enc": text_encs,
                    "img_name": img_id,
                })

        elif dataset == 'sam_cleaned_100k':
            # 원본 SA-Text 데이터셋 로더 (영어)
            json_path = ann_path
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                json_data = sorted(json_data.items())

            split_index = int(len(json_data) * 10 / 11)
            if mode == 'TRAIN':
                json_data = dict(json_data[:split_index])
            elif mode == 'VAL':
                json_data = dict(json_data[split_index:])

            imgs_path = f'{file_list_path}/images'
            imgs = sorted(os.listdir(imgs_path))

            for img in imgs:
                gt_path = f'{imgs_path}/{img}'
                img_id = img.split('.')[0]
                if img_id not in json_data:
                    continue
                img_ann = json_data[img_id]['0']['text_instances']

                boxes = []
                texts = []
                text_encs = []
                polys = []

                for ann in img_ann:
                    text = ann['text']
                    count = sum(1 for ch in text if 32 <= ord(ch) < 127)
                    if count == len(text) and count < 26:
                        texts.append(text)
                        text_encs.append(encode(text))
                    else:
                        continue

                    box_xyxy = ann['bbox']
                    x1, y1, x2, y2 = box_xyxy
                    box_xyxy_scaled = [v/model_H for v in box_xyxy]
                    sx1, sy1, sx2, sy2 = box_xyxy_scaled
                    box_cxcywh = [(sx1+sx2)/2, (sy1+sy2)/2, sx2-sx1, sy2-sy1]
                    processed_box = [round(v, 4) for v in box_cxcywh]
                    boxes.append(processed_box)

                    poly = np.array(ann['polygon']).astype(np.int32)
                    poly_scaled = poly / np.array([model_W, model_H])
                    polys.append(poly_scaled)

                assert len(boxes) == len(texts) == len(text_encs) == len(polys), "Check loader!"
                if len(boxes) == 0 or len(polys) == 0:
                    continue

                caption = [f'"{txt}"' for txt in texts]
                prompt = f"A realistic scene where the texts {', '.join(caption)} appear clearly."

                files.append({
                    "image_path": gt_path,
                    "prompt": prompt,
                    "text": texts,
                    "bbox": boxes,
                    "poly": polys,
                    "text_enc": text_encs,
                    "img_name": img_id,
                })

    if mode == 'VAL':
        n_val = min(2, len(files))
        files = random.sample(files, n_val)

    return files


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:
            cv2.flip(img, 1, img)
        if vflip:
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def filter2D(img, kernel):
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size")
    ph, pw = img.size()[-2:]
    if kernel.size(0) == 1:
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class USMSharp(torch.nn.Module):
    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer("kernel", kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur
        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img
'''

with open(UTILS_PATH, 'w', encoding='utf-8') as f:
    f.write(PATCHED_CONTENT)

print("한글 패치 적용 완료: {}".format(UTILS_PATH))
print("")
print("변경 사항:")
print("  - CTLABELS: ASCII 95자 → ASCII + 한글 11276자")
print("  - encode()/decode(): 한글 문자 지원")
print("  - load_file_list(): korean_receipt 데이터셋 핸들러 추가")
print("")
print("복원하려면:")
print("  cp {} {}".format(backup_path, UTILS_PATH))
