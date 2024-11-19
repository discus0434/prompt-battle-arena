import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler
from PIL import Image

from .sdxl_pipeline import StableDiffusionXLPipeline


def measure_time(func):
    def wrapper(*args, **kwargs):
        begin = time.time()
        result = func(*args, **kwargs)
        print(f"Time: {time.time() - begin:.3f}s")
        return result

    return wrapper


def _cleanup_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    while True:
        if len(prompt) > 0 and prompt[-1] == ",":
            prompt = prompt[:-1].strip()
        else:
            break
    return prompt


@dataclass
class PipelineWrapperParams:
    width: int = 1152
    height: int = 768
    guidance_scale: int = 6
    timesteps: Iterable[float] = (
        999,
        913,
        758,
        650,
        548,
        451,
        337,
        251,
        189,
        143,
        92,
        30,
    )
    num_images_per_prompt: int = 1
    prompt: str = (
        "1girl, general, solo, masterpiece, best quality, very aesthetic, newest"
    )
    negative_prompt: str = "low quality, worst quality, bad quality, normal quality, displeasing, very displeasing"
    latents: torch.Tensor | None = None
    generator: torch.Generator | None = None

    def register_prompt(self, prompt: str) -> None:
        self.prompt = _cleanup_prompt(prompt)

    def register_negative_prompt(self, negative_prompt: str) -> None:
        self.negative_prompt = _cleanup_prompt(negative_prompt)

    def register_latents(self, latents: torch.Tensor) -> None:
        self.latents = latents


class PipelineWrapper:
    def __init__(
        self,
        pretrained_model_link_or_path: str | os.PathLike,
        load_lora_model_link_or_path: str | os.PathLike | None = None,
        height: int = 480,
        width: int = 800,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.params = PipelineWrapperParams(width=width, height=height)
        self._device = device
        self._dtype = dtype
        self._generator = self._seed_everything(42)
        self._noise = self._generate_initial_noise(
            self.params.height, self.params.width
        )

        self.model = self._load_model(
            pretrained_model_link_or_path, load_lora_model_link_or_path
        )

        self._last_image: Image.Image | None = None

    @measure_time
    @torch.inference_mode()
    def __call__(self, prompt: str, negative_prompt: str) -> Image.Image:
        if (
            self._last_image is not None
            and _cleanup_prompt(prompt) == self.params.prompt
            and _cleanup_prompt(negative_prompt) == self.params.negative_prompt
        ):
            return self._last_image
        else:
            self.params.register_prompt(prompt)
            self.params.register_negative_prompt(negative_prompt)
            self._last_image = self.model(**self.params.__dict__)
            return self._last_image

    def _load_model(
        self,
        pretrained_model_link_or_path: str | os.PathLike,
        load_lora_model_link_or_path: str | os.PathLike | None,
    ) -> StableDiffusionXLPipeline:
        model: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_link_or_path,
            torch_dtype=self._dtype,
        )
        if load_lora_model_link_or_path:
            model.load_lora_weights(load_lora_model_link_or_path)
            model.fuse_lora(lora_scale=0.8)
        model.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", torch_dtype=self._dtype
        )
        model.scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            rescale_betas_zero_snr=False,
            steps_offset=1,
            timestep_spacing="leading",
        )
        model.fuse_qkv_projections(vae=False)
        model.to(self._device, self._dtype)
        model.safety_checker = None

        try:
            from sfast.compilers.diffusion_pipeline_compiler import (
                CompilationConfig,
                compile,
            )

            model = compile(
                model,
                CompilationConfig.Default(
                    memory_format=torch.channels_last,
                    enable_xformers=True,
                    enable_triton=True,
                    enable_cuda_graph=True,
                ),
            )
            self._warmup(model)
        except RuntimeError as e:
            model.enable_xformers_memory_efficient_attention()
        except ImportError:
            model.enable_xformers_memory_efficient_attention()
        return model

    def _seed_everything(self, seed: int) -> torch.Generator:
        random.seed(seed)
        np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)

        return generator

    def _generate_initial_noise(self, height: int, width: int) -> torch.Tensor:
        return torch.randn(
            1, 4, height // 8, width // 8, device="cuda", dtype=self._dtype
        )

    @torch.inference_mode()
    def _warmup(self, model: StableDiffusionXLPipeline | None = None) -> None:
        for _ in range(3):
            if model is None:
                self.model(**self.params.__dict__)
            else:
                model(**self.params.__dict__)
