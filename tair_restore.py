#!/usr/bin/env python3
"""TAIR 기반 텍스트 인식 이미지 복원. venv_tair에서 실행.

SwinIR(전처리) + ControlLDM(Stable Diffusion 2.1 기반 복원)으로
저화질/흐릿한 문서 이미지의 글씨를 복원한다.
testr(텍스트 스팟팅) 없이 이미지 복원만 수행하는 경량 버전.

사용법: python3 tair_restore.py input.png output.png [--steps 25]
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

from terediff.model import ControlLDM, Diffusion
from terediff.model.swinir import SwinIR
from terediff.sampler import SpacedSampler
from terediff.utils.common import instantiate_from_config
import torchvision.transforms as T


def load_models(device, weights_dir=None):
    """SwinIR + ControlLDM 모델 로드."""
    if weights_dir is None:
        weights_dir = os.path.join(TAIR_DIR, "weights")

    cfg_path = os.path.join(TAIR_DIR, "configs", "val", "val_terediff.yaml")
    cfg = OmegaConf.load(cfg_path)

    # SwinIR 로드
    swinir = instantiate_from_config(cfg.model.swinir)
    swinir_path = os.path.join(weights_dir, "realesrgan_s4_swinir_100k.pth")
    sd = torch.load(swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    swinir.load_state_dict(sd, strict=True)
    swinir.requires_grad_(False)
    swinir.to(device)
    print("SwinIR loaded")

    # ControlLDM 로드
    cldm = instantiate_from_config(cfg.model.cldm)
    sd_path = os.path.join(weights_dir, "sd2.1-base-zsnr-laionaes5.ckpt")
    sd_weights = torch.load(sd_path, map_location="cpu")["state_dict"]
    cldm.load_pretrained_sd(sd_weights)

    resume_path = os.path.join(weights_dir, "DiffBIR_v2.1.pt")
    if os.path.exists(resume_path):
        cldm.load_controlnet_from_ckpt(torch.load(resume_path, map_location="cpu"))
        print("ControlNet loaded")
    else:
        cldm.load_controlnet_from_unet()
        print("ControlNet: using SD default weights")

    # TeReDiff stage3 checkpoint
    stage3_path = os.path.join(weights_dir, "TAIR", "terediff_stage3.pt")
    if os.path.exists(stage3_path):
        ckpt = torch.load(stage3_path, map_location="cpu")
        if "cldm" in ckpt:
            cldm.load_state_dict(ckpt["cldm"], strict=False)
            print("TeReDiff stage3 cldm loaded")
        if "swinir" in ckpt:
            swinir.load_state_dict(ckpt["swinir"], strict=False)
            print("TeReDiff stage3 swinir loaded")

    cldm.requires_grad_(False)
    cldm.to(device)

    # Diffusion
    diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)

    return swinir, cldm, sampler


@torch.no_grad()
def restore_image(img_cv, swinir, cldm, sampler, device, steps=25):
    """단일 이미지 복원."""
    h_orig, w_orig = img_cv.shape[:2]

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    preprocess = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    # SwinIR: 512x512 input (unshuffle 8x internally)
    clean = swinir(img_tensor)
    clean = clean.clamp(0, 1)

    # ControlLDM
    clean_norm = clean * 2 - 1
    cond = cldm.prepare_condition(clean_norm, [""])

    pure_noise = torch.randn((1, 4, 64, 64), device=device)

    # Diffusion sampling (without testr, direct loop)
    sampler.make_schedule(steps)
    sampler.to(device)

    x = pure_noise
    timesteps = np.flip(sampler.timesteps)
    total_steps = len(sampler.timesteps)

    for i, current_timestep in enumerate(timesteps):
        model_t = torch.full((1,), current_timestep, device=device, dtype=torch.long)
        t = torch.full((1,), total_steps - i - 1, device=device, dtype=torch.long)
        cur_cfg_scale = sampler.get_cfg_scale(1.0, current_timestep)
        x, _ = sampler.p_sample(cldm, x, model_t, t, cond, None, cur_cfg_scale)

    z = x
    restored = cldm.vae_decode(z)
    restored = torch.clamp((restored + 1) / 2, 0, 1)

    restored_np = (restored[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

    if (h_orig, w_orig) != (512, 512):
        restored_bgr = cv2.resize(restored_bgr, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    return restored_bgr


def main():
    parser = argparse.ArgumentParser(description="TAIR text-aware image restoration")
    parser.add_argument("input", help="input image path")
    parser.add_argument("output", help="output image path")
    parser.add_argument("--steps", type=int, default=25, help="diffusion sampling steps (default: 25)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    swinir, cldm, sampler = load_models(device)

    img = cv2.imread(args.input)
    if img is None:
        print("Failed to read:", args.input)
        sys.exit(1)

    print("Input: {}x{}".format(img.shape[1], img.shape[0]))
    restored = restore_image(img, swinir, cldm, sampler, device, steps=args.steps)
    cv2.imwrite(args.output, restored)
    print("Output: {}x{} -> {}".format(restored.shape[1], restored.shape[0], args.output))


if __name__ == "__main__":
    main()
