import argparse
import os

import torch
import torch.nn as nn
import numpy as np
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.dataset import ChannelDataset

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # 处理图像尺寸
    if args.image_height is not None and args.image_width is not None:
        image_height = int(args.image_height)
        image_width = int(args.image_width)
    else:
        image_size = int(args.image_size) if args.image_size is not None else 64
        image_height = image_width = image_size

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_samples = []
    while len(all_samples) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, args.in_channels, image_height, image_width),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples.extend([sample.cpu().numpy() for sample in sample])

    arr = np.concatenate(all_samples, axis=0)
    arr = arr[: args.num_samples]
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=160,
        batch_size=32,
        use_fp16=False,
        model_path="",
        num_workers=4,
        image_height=None,
        image_width=None,
        in_channels=2,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main() 