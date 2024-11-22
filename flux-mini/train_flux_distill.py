import argparse
import logging
import math
import os
import re
import random
import shutil
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file
from PIL import Image 

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import omegaconf
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import load_file as load_sft
from einops import rearrange
from src.flux.sampling import  get_noise, prepare, denoise_simple, unpack, get_schedule
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_flow_model_stu, load_t5, print_load_warning)
from image_datasets.dataset import loader

if is_wandb_available():
    import wandb
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")


def get_models(name: str, device, offload: bool, is_schnell: bool, args):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model_t = load_flow_model2(name, args, device="cpu")
    model_s = load_flow_model_stu(name, args, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model_t, model_s, vae, t5, clip


def distill_loss(args, intermediate_double_s, intermediate_double_t, intermediate_single_s, intermediate_single_t):

    intermediate_double_t_index = args.distill_slim_double
    intermediate_double_t = [x for i, x in enumerate(intermediate_double_t) if i in intermediate_double_t_index]
        
    intermediate_single_t_index = args.distill_target_single
    intermediate_single_t = [x for i, x in enumerate(intermediate_single_t) if i in intermediate_single_t_index]
        
    fn = F.mse_loss     
    loss_kd_double = {}
    for i, (feat_t, feat_s) in enumerate(zip(intermediate_double_t, intermediate_double_s)):
        img_t, txt_t = feat_t
        img_s, txt_s = feat_s

        loss_img = fn(img_s, img_t)
        loss_txt = fn(txt_s, txt_t)

        loss_kd_double[f'double_{str(i)}_img'] = loss_img
        loss_kd_double[f'double_{str(i)}_txt'] = loss_txt

    loss_kd_single = {}
    for i, (feat_t, feat_s) in enumerate(zip(intermediate_single_t, intermediate_single_s)):
        img_t, txt_t = feat_t
        img_s, txt_s = feat_s

        loss_img = fn(img_s, img_t)
        loss_txt = fn(txt_s, txt_t)

        loss_kd_single[f'single_{str(i)}_img'] = loss_img
        loss_kd_single[f'single_{str(i)}_txt'] = loss_txt

    return loss_kd_double, loss_kd_single


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config


@torch.no_grad()
def inference_prompts(prompts, dit_model, t5, clip, vae, is_schnell,
        height=512, width=512, device='cuda:0', dtype=torch.bfloat16, seed=123):

    images = []
    for i, prompt in enumerate(prompts):
        noise = get_noise(1, height, width, device, dtype, seed)
        input = prepare(t5, clip, noise, prompt)

        num_steps = 4 if is_schnell else 50
        timesteps = get_schedule(num_steps, input['img'].shape[1], shift=(not is_schnell))
        denoised = denoise_simple(dit_model, **input, 
                                  timesteps=timesteps, guidance=3.5)

        denoised = unpack(denoised.float(), height, width)
        image = vae.decode(denoised)

        image = image.clamp(-1, 1)
        image = rearrange(image[0], 'c h w -> h w c')
        
        images.append(image)
    return images


class DiTDistiller(torch.nn.Module):
    def __init__(self, t, s, args):
        super().__init__()
        self.t = t
        self.s = s


def main():
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    deepspeed_plugin = accelerate.DeepSpeedPlugin(args.deepspeed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin
    )

    train_dataloader = loader(**args.data_config)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    dit_t, dit_s, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, 
                                             is_schnell=is_schnell, args=args)
    dit = DiTDistiller(dit_t, dit_s, args)        

    teacher_size = sum([p.numel() for p in dit_t.parameters()]) / 1000000000
    student_size = sum([p.numel() for p in dit_s.parameters()]) / 1000000000
    logger.info(f"Teacher sizel {teacher_size:.6f} B, Student size:{student_size:.6f} B")

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit.t.requires_grad_(False)
    dit.s.requires_grad_(True)
    dit = dit.to(torch.float32)
    dit.t.train(); dit.s.train()

    optimizer_cls = torch.optim.AdamW
    param_groups = [p for p in dit.s.parameters()]
    optimizer = optimizer_cls(
        param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_warmup_steps = args.lr_warmup_steps * accelerator.num_processes
    num_training_steps = args.max_train_steps * accelerator.num_processes
    step_rules = getattr(args, 'step_rules', None)
    
    logger.info(f"warmup steps: {num_warmup_steps}")
    logger.info(f"training steps: {num_training_steps}")
    logger.info(f"stepping rules: {step_rules}")
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        step_rules=step_rules,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    global_step = 0
    first_epoch = 0

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.data_config.train_batch_size
    dit, optimizer, lr_scheduler = accelerator.prepare(
        dit, optimizer, lr_scheduler
    )


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))

            if '/' in path:
                path = path.split('/')[-1]
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )


    img_dir = os.path.join(args.output_dir, 'image')
    os.makedirs(img_dir, exist_ok=True)
    with accelerator.autocast():
        for epoch in range(first_epoch, args.num_train_epochs):
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                dit.s.train()
                dit.t.eval()
                with accelerator.accumulate(dit):
                    img, prompts = batch
                    with torch.no_grad():
                        x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                        inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                        x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                    bs = img.shape[0]                
                    t = torch.sigmoid(torch.randn((bs,), device=accelerator.device))
                    
                    t_ = t.reshape(-1, 1, 1)
                    x_0 = torch.randn_like(x_1).to(accelerator.device)
                    x_t = (1 - t_) * x_1 + t_ * x_0
                    guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                    with torch.no_grad():
                        model_pred_t, intermediate_double_t, intermediate_single_t = dit.t(img=x_t.to(weight_dtype),
                                                        img_ids=inp['img_ids'].to(weight_dtype),
                                                        txt=inp['txt'].to(weight_dtype),
                                                        txt_ids=inp['txt_ids'].to(weight_dtype),
                                                        y=inp['vec'].to(weight_dtype),
                                                        timesteps=t.to(weight_dtype),
                                                        guidance=guidance_vec.to(weight_dtype),
                                                        return_intermediate=True)
                        

                    model_pred_s, intermediate_double_s, intermediate_single_s = dit.s(img=x_t.to(weight_dtype),
                                                                                img_ids=inp['img_ids'].to(weight_dtype),
                                                                                txt=inp['txt'].to(weight_dtype),
                                                                                txt_ids=inp['txt_ids'].to(weight_dtype),
                                                                                y=inp['vec'].to(weight_dtype),
                                                                                timesteps=t.to(weight_dtype),
                                                                                guidance=guidance_vec.to(weight_dtype),
                                                                                return_intermediate=True)

                    loss_denoise = F.mse_loss(model_pred_s.float(), (x_0 - x_1).float(), reduction="mean")
                    loss_output  = F.mse_loss(model_pred_s.float(), model_pred_t.float(), reduction="mean")


                    loss_double, loss_single = distill_loss(args, intermediate_double_s, intermediate_double_t, intermediate_single_s, intermediate_single_t)
                    
                    # nomralize the weights according to their scale
                    loss_double_last = [v for k, v in loss_double.items() if '4' in k]
                    loss_kd =  (sum(loss_double.values()) - sum(loss_double_last) * 0.9 ) / len(loss_double) + \
                            sum(loss_single.values()) / len(loss_single) * 0.05
                            
                    loss = loss_denoise + args.kd_weight * loss_kd + args.output_weight * loss_output

                    
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    avg_loss_denoise = accelerator.gather(loss_denoise.repeat(args.train_batch_size)).mean()
                    avg_loss_output  = accelerator.gather(loss_output.repeat(args.train_batch_size)).mean()

                    avg_loss_double = {
                        k: accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                        for k, loss in loss_double.items()
                    }
                    avg_loss_double['full_double'] = sum(avg_loss_double.values()) / len(avg_loss_double.values())

                    avg_loss_single = {
                        k: accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                        for k, loss in loss_single.items()
                    }
                    avg_loss_single['full_single'] = sum(avg_loss_single.values()) / len(avg_loss_single.values())

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(dit.s.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()


                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    accelerator.log(avg_loss_double, step=global_step)
                    accelerator.log(avg_loss_single, step=global_step)
                    accelerator.log({"output_loss": avg_loss_output}, step=global_step)
                    accelerator.log({"denoise": avg_loss_denoise}, step=global_step)

                    lr = optimizer.param_groups[0]['lr']
                    accelerator.log({"lr": lr}, step=global_step)

                    train_loss = 0.0
                    if global_step % args.every_log_image == 0:
                        if accelerator.is_main_process:

                            dit.s.eval()
                            logger.info("logging images for S validation...")
                            images = inference_prompts(prompts, dit.s, t5, clip, vae, is_schnell,
                                height=args.data_config.img_size, width=args.data_config.img_size, dtype=weight_dtype)
                            for i, image in enumerate(images):
                                image = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
                                save_path = os.path.join(img_dir, f'S-{global_step}-{i}.jpg')
                                logger.info(f"saving image to: {save_path}")
                                image.save(save_path, quality=95, subsampling=0)

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        if getattr(args, 'save_states', None):
                            accelerator.save_state(save_path)
                        unwrapped_model_state = accelerator.unwrap_model(dit.s).state_dict()

                        save_file(
                            unwrapped_model_state,
                            os.path.join(save_path, "flux-mini.safetensors")
                        )
                        logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
