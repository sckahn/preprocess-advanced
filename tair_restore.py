#!/usr/bin/env python3
"""TAIR 타일 기반 글씨복원. venv_tair에서 실행.

이미지를 128x128 타일로 분할 → 각각 TAIR 복원(512x512, 4x) → 합성.
TAIR 공식 로직: LQ 128 → resize 512 → SwinIR → ControlLDM Diffusion → 512 HQ

사용법: python3 tair_restore.py input.png output.png [--steps 50]
"""
import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIR_DIR = os.path.join(SCRIPT_DIR, "tair_repo")
sys.path.insert(0, TAIR_DIR)

import torch
import numpy as np
import cv2
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from terediff.model import ControlLDM, Diffusion
from terediff.model.swinir import SwinIR
from terediff.sampler import SpacedSampler
from terediff.utils.common import instantiate_from_config
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def load_models(device, weights_dir=None):
    if weights_dir is None:
        weights_dir = os.path.join(TAIR_DIR, "weights")

    cfg_path = os.path.join(TAIR_DIR, "configs", "val", "val_terediff.yaml")
    cfg = OmegaConf.load(cfg_path)

    # SwinIR
    swinir = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(os.path.join(weights_dir, "realesrgan_s4_swinir_100k.pth"), map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    swinir.load_state_dict(sd, strict=True)
    swinir.requires_grad_(False)

    # ControlLDM
    cldm = instantiate_from_config(cfg.model.cldm)
    sd_weights = torch.load(os.path.join(weights_dir, "sd2.1-base-zsnr-laionaes5.ckpt"), map_location="cpu")["state_dict"]
    cldm.load_pretrained_sd(sd_weights)
    cldm.load_controlnet_from_ckpt(torch.load(os.path.join(weights_dir, "DiffBIR_v2.1.pt"), map_location="cpu"))

    # TeReDiff stage3
    stage3_path = os.path.join(weights_dir, "TAIR", "terediff_stage3.pt")
    if os.path.exists(stage3_path):
        ckpt = torch.load(stage3_path, map_location="cpu")
        if "cldm" in ckpt:
            cldm.load_state_dict(ckpt["cldm"], strict=False)
        if "swinir" in ckpt:
            swinir.load_state_dict(ckpt["swinir"], strict=False)
        print("TeReDiff stage3 loaded")

    swinir.eval().to(device)
    cldm.eval().requires_grad_(False).to(device)

    diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)

    return swinir, cldm, sampler


# LQ 전처리: 128x128 → 512x512, [0,1]
PREPROCESS_LQ = T.Compose([
    T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor()
])


@torch.no_grad()
def restore_tile(tile_pil, swinir, cldm, sampler, device, steps):
    """128x128 PIL 타일 → 512x512 numpy HQ."""
    val_lq = PREPROCESS_LQ(tile_pil).unsqueeze(0).to(device)
    val_clean = swinir(val_lq)
    val_cond = cldm.prepare_condition(val_clean, [""])

    pure_noise = torch.randn((1, 4, 64, 64), device=device)
    sampler.make_schedule(steps)
    sampler.to(device)

    x = pure_noise
    timesteps = np.flip(sampler.timesteps)
    total_steps = len(sampler.timesteps)

    for i, current_timestep in enumerate(timesteps):
        model_t = torch.full((1,), current_timestep, device=device, dtype=torch.long)
        t = torch.full((1,), total_steps - i - 1, device=device, dtype=torch.long)
        x, _ = sampler.p_sample(cldm, x, model_t, t, val_cond, None,
                                sampler.get_cfg_scale(1.0, current_timestep))

    restored = torch.clamp((cldm.vae_decode(x) + 1) / 2, 0, 1)
    return restored[0].permute(1, 2, 0).cpu().numpy()  # 512x512x3, [0,1]


def restore_image(input_path, output_path, swinir, cldm, sampler, device, steps):
    """이미지를 128x128 타일로 분할 → TAIR 복원(4x) → 합성."""
    img = cv2.imread(input_path)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    LQ = 128
    HQ = 512
    SCALE = HQ // LQ  # 4
    OVERLAP = 16  # LQ 기준

    h_out, w_out = h * SCALE, w * SCALE
    step = LQ - OVERLAP

    output = np.zeros((h_out, w_out, 3), dtype=np.float64)
    weight = np.zeros((h_out, w_out, 1), dtype=np.float64)

    # 블렌딩 가중치
    blend = np.ones((HQ, HQ, 1), dtype=np.float64)
    ramp_len = OVERLAP * SCALE
    ramp = np.linspace(0, 1, ramp_len)
    for i in range(ramp_len):
        blend[i, :, 0] *= ramp[i]
        blend[-(i + 1), :, 0] *= ramp[i]
        blend[:, i, 0] *= ramp[i]
        blend[:, -(i + 1), 0] *= ramp[i]

    # 타일 좌표 계산
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = min(y, max(0, h - LQ))
            x1 = min(x, max(0, w - LQ))
            tiles.append((y1, x1))
    # 중복 제거
    tiles = list(dict.fromkeys(tiles))

    print("  tiles: {} ({}x{} -> {}x{})".format(len(tiles), w, h, w_out, h_out))

    for idx, (y1, x1) in enumerate(tqdm(tiles, desc="  TAIR")):
        y2 = min(y1 + LQ, h)
        x2 = min(x1 + LQ, w)
        th, tw = y2 - y1, x2 - x1

        # 128x128 패딩 (빈 부분은 흰색)
        tile = np.full((LQ, LQ, 3), 255, dtype=np.uint8)
        tile[:th, :tw] = img_rgb[y1:y2, x1:x2]
        tile_pil = Image.fromarray(tile)

        # TAIR 복원 → 512x512
        result = restore_tile(tile_pil, swinir, cldm, sampler, device, steps)

        # 출력 좌표
        oy1, ox1 = y1 * SCALE, x1 * SCALE
        oth, otw = th * SCALE, tw * SCALE

        # 블렌딩
        w_tile = blend[:oth, :otw]
        output[oy1:oy1 + oth, ox1:ox1 + otw] += result[:oth, :otw] * w_tile
        weight[oy1:oy1 + oth, ox1:ox1 + otw] += w_tile

    output /= np.maximum(weight, 1e-8)
    output = (output * 255).clip(0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, output)
    print("  OK {}x{} -> {}x{}".format(w, h, w_out, h_out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    swinir, cldm, sampler = load_models(device)
    restore_image(args.input, args.output, swinir, cldm, sampler, device, args.steps)


if __name__ == "__main__":
    main()
