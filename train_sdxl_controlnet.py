"""
SDXL + ControlNet (canny) training using Hugging Face Diffusers + Accelerate.

Dataset layout:
  baseline/sdxl_controlnet_canny/data/<City>/<Community_ID>/
    control.png        (roads only) OR control_canny.png if using canny
    target.png         (roads + buildings)
    meta.json

Train full finetune of UNet + ControlNet at 1024. Text encoders and VAE are frozen.
Prompts: empty string (unconditional) for simplicity.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from accelerate import Accelerator
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from huggingface_hub import login as hf_login


class CommunityImagePairDataset(Dataset):
    def __init__(self, root_dir: str, use_canny: bool = True, image_size: int = 1024):
        self.root = Path(root_dir)
        self.use_canny = use_canny
        self.image_size = image_size
        self.samples = []
        for city_dir in sorted(self.root.glob("*")):
            if not city_dir.is_dir():
                continue
            for comm_dir in sorted(city_dir.glob("*")):
                if not comm_dir.is_dir():
                    continue
                control = comm_dir / ("control_canny.png" if use_canny else "control.png")
                target = comm_dir / "target.png"
                if control.exists() and target.exists():
                    self.samples.append((control, target))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # CHW
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        control_path, target_path = self.samples[idx]
        control = self._load_image(control_path)
        target = self._load_image(target_path)
        return {
            "pixel_values": target,
            "conditioning": control,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default="baseline/sdxl_controlnet_canny/data")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default="diffusers/controlnet-canny-sdxl-1.0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--use_canny", action="store_true", default=True)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no","fp16","bf16"])  # use "no" to disable
    parser.add_argument("--output_dir", type=str, default="baseline/sdxl_controlnet_canny/ckpts")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--hf_token", type=str, default="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")
    args = parser.parse_args()

    # Accelerator (handles gradient accumulation + mixed precision)
    mp = None if args.mixed_precision == "no" else args.mixed_precision
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mp,
    )

    # Optional HF login (for gated models)
    if args.hf_token:
        try:
            hf_login(token=args.hf_token)
            accelerator.print("Logged into Hugging Face Hub.")
        except Exception:
            accelerator.print("HF login failed; proceeding without.")

    ds = CommunityImagePairDataset(args.train_data_dir, use_canny=args.use_canny, image_size=args.resolution)
    accelerator.print(f"Loaded {len(ds)} training pairs from {args.train_data_dir}")
    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Build pipeline and components
    accelerator.print("Loading SDXL base and ControlNet...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=torch.bfloat16 if mp=="bf16" else None)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16 if mp=="bf16" else None,
    )

    # Freeze VAE and text encoders
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    # Train UNet + ControlNet
    pipe.unet.requires_grad_(True)
    pipe.controlnet.requires_grad_(True)

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Optimizer
    params = list(pipe.unet.parameters()) + list(pipe.controlnet.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    # Prepare with accelerator
    pipe.unet, pipe.controlnet, optimizer, dl = accelerator.prepare(
        pipe.unet, pipe.controlnet, optimizer, dl
    )
    vae = pipe.vae.to(accelerator.device)
    text_encoder = pipe.text_encoder.to(accelerator.device)
    text_encoder_2 = pipe.text_encoder_2.to(accelerator.device)

    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.print(
        f"Starting training | steps={args.max_train_steps}, accum={args.gradient_accumulation_steps}, mp={args.mixed_precision}"
    )

    # Helper to build SDXL added_cond (time ids)
    def build_time_ids(bsz: int, height: int, width: int) -> torch.Tensor:
        # original, crop, target (all 1024 for training)
        add_time_ids = pipe._get_add_time_ids(
            (height, width), (0, 0), (height, width), dtype=torch.float32
        )  # (6,)
        add_time_ids = add_time_ids.unsqueeze(0).repeat(bsz, 1)
        return add_time_ids

    global_step = 0
    pipe.unet.train(); pipe.controlnet.train()

    while global_step < args.max_train_steps:
        for batch in dl:
            with accelerator.accumulate(pipe.unet):
                target = batch["pixel_values"].to(accelerator.device)  # [0,1]
                control = batch["conditioning"].to(accelerator.device)  # [0,1]

                # Encode target image to latents
                target = (target * 2.0 - 1.0).clamp(-1, 1)  # [-1,1]
                latents = vae.encode(target).latent_dist.sample()
                latents = latents * 0.18215

                # Sample random noise & timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text encodings (empty prompt)
                prompt = [""] * bsz
                prompt_embeds, pooled_embeds = pipe.encode_prompt(prompt, device=latents.device)
                time_ids = build_time_ids(bsz, args.resolution, args.resolution).to(latents.device)
                added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": time_ids}

                # ControlNet forward to get residuals
                down_samples, mid_sample = pipe.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=control,
                    conditioning_scale=1.0,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                # UNet noise prediction with residuals
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                loss = torch.mean((model_pred - noise) ** 2)
                accelerator.backward(loss)
                optimizer.step(); optimizer.zero_grad()

            if accelerator.is_main_process and (global_step % 50 == 0):
                accelerator.print(f"step={global_step} loss={loss.detach().float().item():.6f}")

            global_step += 1
            if accelerator.is_main_process and args.save_every and (global_step % args.save_every == 0):
                # Save UNet + ControlNet weights
                unet_to_save = accelerator.unwrap_model(pipe.unet)
                ctrl_to_save = accelerator.unwrap_model(pipe.controlnet)
                unet_to_save.save_pretrained(os.path.join(args.output_dir, f"unet_step_{global_step}"))
                ctrl_to_save.save_pretrained(os.path.join(args.output_dir, f"controlnet_step_{global_step}"))

            if global_step >= args.max_train_steps:
                break
        # end for
    # end while

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_to_save = accelerator.unwrap_model(pipe.unet)
        ctrl_to_save = accelerator.unwrap_model(pipe.controlnet)
        unet_to_save.save_pretrained(os.path.join(args.output_dir, "unet_final"))
        ctrl_to_save.save_pretrained(os.path.join(args.output_dir, "controlnet_final"))
    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()

