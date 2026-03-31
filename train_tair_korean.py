#!/usr/bin/env python3
"""TAIR Korean fine-tuning (Stage1: ControlNet + U-Net attention).

24GB GPU. Data: dataset/hq/ (512x512) + dataset/lq/ (128x128).

Usage:
  python3 train_tair_korean.py --data_dir ./dataset --steps 20000 --batch_size 4

Files needed on GPU server:
  - tair_repo/              (TAIR code)
  - tair_repo/weights/      (checkpoints)
  - dataset/hq/, dataset/lq/ (training data)
  - train_tair_korean.py    (this script)
"""
import os
import sys
import argparse
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIR_DIR = os.path.join(SCRIPT_DIR, "tair_repo")
sys.path.insert(0, TAIR_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from terediff.model import ControlLDM, Diffusion
from terediff.model.swinir import SwinIR
from terediff.sampler import SpacedSampler
from terediff.utils.common import instantiate_from_config


class KoreanDocDataset(Dataset):
    def __init__(self, data_dir, split="train", val_ratio=0.05):
        hq_dir = os.path.join(data_dir, "hq")
        lq_dir = os.path.join(data_dir, "lq")
        hq_files = sorted(glob(os.path.join(hq_dir, "*.jpg")))
        lq_files = sorted(glob(os.path.join(lq_dir, "*.jpg")))
        assert len(hq_files) == len(lq_files)

        n_val = max(1, int(len(hq_files) * val_ratio))
        if split == "train":
            self.hq_files = hq_files[n_val:]
            self.lq_files = lq_files[n_val:]
        else:
            self.hq_files = hq_files[:n_val]
            self.lq_files = lq_files[:n_val]

        self.hq_transform = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.lq_transform = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.hq_files)

    def __getitem__(self, idx):
        hq = Image.open(self.hq_files[idx]).convert("RGB")
        lq = Image.open(self.lq_files[idx]).convert("RGB")
        return {"hq": self.hq_transform(hq), "lq": self.lq_transform(lq)}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    if torch.cuda.is_available():
        print("GPU: {}".format(torch.cuda.get_device_name(0)))

    cfg_path = os.path.join(TAIR_DIR, "configs", "val", "val_terediff.yaml")
    cfg = OmegaConf.load(cfg_path)
    weights_dir = os.path.join(TAIR_DIR, "weights")

    # Load SwinIR (frozen)
    print("Loading SwinIR...")
    swinir = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(os.path.join(weights_dir, "realesrgan_s4_swinir_100k.pth"), map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    swinir.load_state_dict(sd, strict=True)
    swinir.requires_grad_(False)
    swinir.to(device)

    # Load ControlLDM
    print("Loading ControlLDM...")
    cldm = instantiate_from_config(cfg.model.cldm)
    sd_weights = torch.load(os.path.join(weights_dir, "sd2.1-base-zsnr-laionaes5.ckpt"), map_location="cpu")["state_dict"]
    cldm.load_pretrained_sd(sd_weights)
    cldm.load_controlnet_from_ckpt(torch.load(os.path.join(weights_dir, "DiffBIR_v2.1.pt"), map_location="cpu"))

    stage3_path = os.path.join(weights_dir, "TAIR", "terediff_stage3.pt")
    if os.path.exists(stage3_path):
        ckpt = torch.load(stage3_path, map_location="cpu")
        if "cldm" in ckpt:
            cldm.load_state_dict(ckpt["cldm"], strict=False)
        if "swinir" in ckpt:
            swinir.load_state_dict(ckpt["swinir"], strict=False)
        print("Loaded stage3 checkpoint as starting point")
    cldm.to(device)

    # Diffusion
    diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)

    # Freeze: only train controlnet + unet attention
    train_params = []
    for name, param in cldm.named_parameters():
        if "controlnet" in name or ("unet" in name and "attn" in name):
            param.requires_grad = True
            train_params.append(param)
        else:
            param.requires_grad = False

    n_train = sum(p.numel() for p in train_params)
    n_total = sum(p.numel() for p in cldm.parameters())
    print("Trainable: {:.1f}M / {:.1f}M ({:.1f}%)".format(n_train/1e6, n_total/1e6, 100*n_train/n_total))

    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)

    # Dataset
    train_ds = KoreanDocDataset(args.data_dir, split="train")
    val_ds = KoreanDocDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    print("Train: {}, Val: {}, Batch: {}".format(len(train_ds), len(val_ds), args.batch_size))

    os.makedirs(args.ckpt_dir, exist_ok=True)
    cldm.train()
    global_step = 0
    epoch = 0
    running_loss = 0.0

    print("\n" + "=" * 60)
    print("Training {} steps, lr={}".format(args.steps, args.lr))
    print("Checkpoints: {}".format(args.ckpt_dir))
    print("=" * 60 + "\n")

    while global_step < args.steps:
        epoch += 1
        pbar = tqdm(train_loader, desc="Epoch {}".format(epoch))

        for batch in pbar:
            if global_step >= args.steps:
                break

            hq = batch["hq"].to(device)
            lq = batch["lq"].to(device)
            bs = hq.shape[0]

            with torch.no_grad():
                clean = swinir(lq)

            t = torch.randint(0, diffusion.num_timesteps, (bs,), device=device)

            with torch.no_grad():
                z = cldm.vae_encode(hq)

            noise = torch.randn_like(z)
            z_noisy = diffusion.q_sample(z, t, noise)
            cond = cldm.prepare_condition(clean, [""] * bs)
            model_output = cldm(z_noisy, t, cond)

            if diffusion.parameterization == "v":
                target = diffusion.get_v(z, noise, t)
            else:
                target = noise

            loss = F.mse_loss(model_output, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                pbar.set_postfix(step=global_step, loss="{:.4f}".format(avg_loss))
                running_loss = 0.0

            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, "korean_step{}.pt".format(global_step))
                torch.save({
                    "cldm": cldm.state_dict(),
                    "swinir": swinir.state_dict(),
                    "step": global_step,
                    "optimizer": optimizer.state_dict(),
                }, ckpt_path)
                print("\n  Saved: {}".format(ckpt_path))

            if global_step % args.sample_every == 0 and len(val_ds) > 0:
                cldm_was_training = cldm.training
                cldm.eval()
                with torch.no_grad():
                    sample = val_ds[0]
                    val_lq = sample["lq"].unsqueeze(0).to(device)
                    val_hq = sample["hq"].unsqueeze(0).to(device)
                    val_clean = swinir(val_lq)
                    val_cond = cldm.prepare_condition(val_clean, [""])

                    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
                    sampler.make_schedule(25)
                    sampler.to(device)
                    x = torch.randn((1, 4, 64, 64), device=device)
                    timesteps = np.flip(sampler.timesteps)
                    for i, ts in enumerate(timesteps):
                        mt = torch.full((1,), ts, device=device, dtype=torch.long)
                        tt = torch.full((1,), len(timesteps)-i-1, device=device, dtype=torch.long)
                        x, _ = sampler.p_sample(cldm, x, mt, tt, val_cond, None,
                                                sampler.get_cfg_scale(1.0, ts))
                    restored = torch.clamp((cldm.vae_decode(x) + 1) / 2, 0, 1)
                    lq_up = F.interpolate(val_lq, size=(512, 512), mode="bicubic", align_corners=False)
                    gt_vis = torch.clamp((val_hq + 1) / 2, 0, 1)
                    import torchvision.utils as vutils
                    grid = torch.cat([lq_up, restored, gt_vis], dim=0)
                    vutils.save_image(grid, os.path.join(args.ckpt_dir, "sample_step{}.png".format(global_step)),
                                      nrow=3, padding=4)
                    print("  Sample: sample_step{}.png".format(global_step))
                if cldm_was_training:
                    cldm.train()

    final_path = os.path.join(args.ckpt_dir, "korean_final.pt")
    torch.save({"cldm": cldm.state_dict(), "swinir": swinir.state_dict(), "step": global_step}, final_path)
    print("\nDone! Final: {}".format(final_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ckpt_dir", default="./checkpoints/korean_stage1")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--sample_every", type=int, default=1000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
