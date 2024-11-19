# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_invisible_watermark_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_invisible_watermark_available():
    pass


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs:
    text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \\( - literal character '('
      \\[ - literal character '['
      \\) - literal character ')'
      \\] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\\(literal\\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """
    import re

    re_attention = re.compile(
        r"""
            \\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|
            \)|]|[^\\()\[\]:]+|:
        """,
        re.X,
    )

    re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_tokens_with_weights(clip_tokenizer: CLIPTokenizer, prompt: str):
    """
    Get prompt token ids and weights, this function works for both
    prompt and negative prompt

    Args:
        pipe (CLIPTokenizer)
            A CLIPTokenizer
        prompt (str)
            A prompt string with weights

    Returns:
        text_tokens (list)
            A list contains token ids
        text_weight (list)
            A list contains the correspondent weight of token ids

    Example:
        import torch
        from transformers import CLIPTokenizer

        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , subfolder = "tokenizer"
            , dtype = torch.float16
        )

        token_id_list, token_weight_list = get_prompts_tokens_with_weights(
            clip_tokenizer = clip_tokenizer
            ,prompt = "a (red:1.5) cat"*70
        )
    """
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens, text_weights = [], []
    for word, weight in texts_and_weights:
        # tokenize and discard the starting and the ending token
        token = clip_tokenizer(word, truncation=False).input_ids[
            1:-1
        ]  # so that tokenize whatever length prompt
        # the returned token is a 1d list: [320, 1125, 539, 320]

        # merge the new tokens to the all tokens holder: text_tokens
        text_tokens = [*text_tokens, *token]

        # each token chunk will come with one weight, like ['red cat', 2.0]
        # need to expand weight for each token.
        chunk_weights = [weight] * len(token)

        # append the weight back to the weight holder: text_weights
        text_weights = [*text_weights, *chunk_weights]
    return text_tokens, text_weights


def group_tokens_and_weights(token_ids: list, weights: list, pad_last_block=False):
    """
    Produce tokens and weights in groups and pad the missing tokens

    Args:
        token_ids (list)
            The token ids from tokenizer
        weights (list)
            The weights list from function get_prompts_tokens_with_weights
        pad_last_block (bool)
            Control if fill the last token list to 75 tokens with eos
    Returns:
        new_token_ids (2d list)
        new_weights (2d list)

    Example:
        token_groups,weight_groups = group_tokens_and_weights(
            token_ids = token_id_list
            , weights = token_weight_list
        )
    """
    bos, eos = 49406, 49407

    # this will be a 2d list
    new_token_ids = []
    new_weights = []
    while len(token_ids) >= 75:
        # get the first 75 tokens
        head_75_tokens = [token_ids.pop(0) for _ in range(75)]
        head_75_weights = [weights.pop(0) for _ in range(75)]

        # extract token ids and weights
        temp_77_token_ids = [bos] + head_75_tokens + [eos]
        temp_77_weights = [1.0] + head_75_weights + [1.0]

        # add 77 token and weights chunk to the holder list
        new_token_ids.append(temp_77_token_ids)
        new_weights.append(temp_77_weights)

    # padding the left
    if len(token_ids) > 0:
        padding_len = 75 - len(token_ids) if pad_last_block else 0

        temp_77_token_ids = [bos] + token_ids + [eos] * padding_len + [eos]
        new_token_ids.append(temp_77_token_ids)

        temp_77_weights = [1.0] + weights + [1.0] * padding_len + [1.0]
        new_weights.append(temp_77_weights)

    return new_token_ids, new_weights


def get_weighted_text_embeddings_sdxl(
    pipe: StableDiffusionXLPipeline,
    prompt: str = "",
    prompt_2: str = None,
    negative_prompt: str = "",
    negative_prompt_2: str = None,
    num_images_per_prompt: int = 1,
    device: torch.device | None = None,
    clip_skip: int | None = None,
    lora_scale: int | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    pooled_prompt_embeds: torch.Tensor | None = None,
    negative_pooled_prompt_embeds: torch.Tensor | None = None,
):
    """
    This function can process long prompt with weights, no length limitation
    for Stable Diffusion XL

    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        prompt_2 (str)
        negative_prompt (str)
        negative_prompt_2 (str)
        num_images_per_prompt (int)
        device (torch.device)
        clip_skip (int)
    Returns:
        prompt_embeds (torch.Tensor)
        negative_prompt_embeds (torch.Tensor)
    """
    if prompt_embeds is not None and negative_prompt_embeds is not None:
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    device = device or pipe._execution_device

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(pipe, StableDiffusionXLLoraLoaderMixin):
        pipe._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if pipe.text_encoder is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(pipe.text_encoder, lora_scale)
            else:
                scale_lora_layers(pipe.text_encoder, lora_scale)

        if pipe.text_encoder_2 is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(pipe.text_encoder_2, lora_scale)
            else:
                scale_lora_layers(pipe.text_encoder_2, lora_scale)

    if prompt_2:
        prompt = f"{prompt} {prompt_2}"

    if negative_prompt_2:
        negative_prompt = f"{negative_prompt} {negative_prompt_2}"

    prompt_t1 = prompt_t2 = prompt
    negative_prompt_t1 = negative_prompt_t2 = negative_prompt

    if isinstance(pipe, TextualInversionLoaderMixin):
        prompt_t1 = pipe.maybe_convert_prompt(prompt_t1, pipe.tokenizer)
        negative_prompt_t1 = pipe.maybe_convert_prompt(
            negative_prompt_t1, pipe.tokenizer
        )
        prompt_t2 = pipe.maybe_convert_prompt(prompt_t2, pipe.tokenizer_2)
        negative_prompt_t2 = pipe.maybe_convert_prompt(
            negative_prompt_t2, pipe.tokenizer_2
        )

    eos = pipe.tokenizer.eos_token_id

    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt_t1
    )
    negative_prompt_tokens, negative_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, negative_prompt_t1
    )

    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt_t2
    )
    negative_prompt_tokens_2, negative_prompt_weights_2 = (
        get_prompts_tokens_with_weights(pipe.tokenizer_2, negative_prompt_t2)
    )

    # padding the shorter one for prompt set 1
    prompt_token_len = len(prompt_tokens)
    negative_prompt_token_len = len(negative_prompt_tokens)

    if prompt_token_len > negative_prompt_token_len:
        # padding the negative_prompt with eos token
        negative_prompt_tokens = negative_prompt_tokens + [eos] * abs(
            prompt_token_len - negative_prompt_token_len
        )
        negative_prompt_weights = negative_prompt_weights + [1.0] * abs(
            prompt_token_len - negative_prompt_token_len
        )
    else:
        # padding the prompt
        prompt_tokens = prompt_tokens + [eos] * abs(
            prompt_token_len - negative_prompt_token_len
        )
        prompt_weights = prompt_weights + [1.0] * abs(
            prompt_token_len - negative_prompt_token_len
        )

    # padding the shorter one for token set 2
    prompt_token_len_2 = len(prompt_tokens_2)
    negative_prompt_token_len_2 = len(negative_prompt_tokens_2)

    if prompt_token_len_2 > negative_prompt_token_len_2:
        # padding the negative_prompt with eos token
        negative_prompt_tokens_2 = negative_prompt_tokens_2 + [eos] * abs(
            prompt_token_len_2 - negative_prompt_token_len_2
        )
        negative_prompt_weights_2 = negative_prompt_weights_2 + [1.0] * abs(
            prompt_token_len_2 - negative_prompt_token_len_2
        )
    else:
        # padding the prompt
        prompt_tokens_2 = prompt_tokens_2 + [eos] * abs(
            prompt_token_len_2 - negative_prompt_token_len_2
        )
        prompt_weights_2 = prompt_weights + [1.0] * abs(
            prompt_token_len_2 - negative_prompt_token_len_2
        )

    embeds = []
    negative_embeds = []

    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy(), prompt_weights.copy()
    )

    negative_prompt_token_groups, negative_prompt_weight_groups = (
        group_tokens_and_weights(
            negative_prompt_tokens.copy(), negative_prompt_weights.copy()
        )
    )

    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy(), prompt_weights_2.copy()
    )

    negative_prompt_token_groups_2, negative_prompt_weight_groups_2 = (
        group_tokens_and_weights(
            negative_prompt_tokens_2.copy(), negative_prompt_weights_2.copy()
        )
    )

    # get prompt embeddings one by one is not working.
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]], dtype=torch.long, device=device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i], dtype=torch.float16, device=device
        )

        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]], dtype=torch.long, device=device
        )

        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(device), output_hidden_states=True
        )

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(device), output_hidden_states=True
        )
        pooled_prompt_embeds = prompt_embeds_2[0]

        if clip_skip is None:
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]
            prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[
                -(clip_skip + 2)
            ]
            prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[
                -(clip_skip + 2)
            ]

        prompt_embeds_list = [
            prompt_embeds_1_hidden_states,
            prompt_embeds_2_hidden_states,
        ]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)

        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                token_embedding[j] = (
                    token_embedding[-1]
                    + (token_embedding[j] - token_embedding[-1]) * weight_tensor[j]
                )

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)

        # get negative prompt embeddings with weights
        negative_token_tensor = torch.tensor(
            [negative_prompt_token_groups[i]], dtype=torch.long, device=device
        )
        negative_token_tensor_2 = torch.tensor(
            [negative_prompt_token_groups_2[i]], dtype=torch.long, device=device
        )
        negative_weight_tensor = torch.tensor(
            negative_prompt_weight_groups[i], dtype=torch.float16, device=device
        )

        # use first text encoder
        negative_prompt_embeds_1 = pipe.text_encoder(
            negative_token_tensor.to(device), output_hidden_states=True
        )
        negative_prompt_embeds_1_hidden_states = negative_prompt_embeds_1.hidden_states[
            -2
        ]

        # use second text encoder
        negative_prompt_embeds_2 = pipe.text_encoder_2(
            negative_token_tensor_2.to(device), output_hidden_states=True
        )
        negative_prompt_embeds_2_hidden_states = negative_prompt_embeds_2.hidden_states[
            -2
        ]
        negative_pooled_prompt_embeds = negative_prompt_embeds_2[0]

        negative_prompt_embeds_list = [
            negative_prompt_embeds_1_hidden_states,
            negative_prompt_embeds_2_hidden_states,
        ]
        negative_token_embedding = torch.concat(
            negative_prompt_embeds_list, dim=-1
        ).squeeze(0)

        for z in range(len(negative_weight_tensor)):
            if negative_weight_tensor[z] != 1.0:
                negative_token_embedding[z] = (
                    negative_token_embedding[-1]
                    + (negative_token_embedding[z] - negative_token_embedding[-1])
                    * negative_weight_tensor[z]
                )

        negative_token_embedding = negative_token_embedding.unsqueeze(0)
        negative_embeds.append(negative_token_embedding)

    prompt_embeds = torch.cat(embeds, dim=1)
    negative_prompt_embeds = torch.cat(negative_embeds, dim=1)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt,
    # using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(
        bs_embed * num_images_per_prompt, seq_len, -1
    )

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(
        1, num_images_per_prompt, 1
    ).view(bs_embed * num_images_per_prompt, -1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
        1, num_images_per_prompt, 1
    ).view(bs_embed * num_images_per_prompt, -1)

    if (
        pipe.text_encoder is not None
        and isinstance(pipe, StableDiffusionXLLoraLoaderMixin)
        and USE_PEFT_BACKEND
    ):
        # Retrieve the original scale by scaling back the LoRA layers
        unscale_lora_layers(pipe.text_encoder, lora_scale)

    if (
        pipe.text_encoder_2 is not None
        and isinstance(pipe, StableDiffusionXLLoraLoaderMixin)
        and USE_PEFT_BACKEND
    ):
        # Retrieve the original scale by scaling back the LoRA layers
        unscale_lora_layers(pipe.text_encoder_2, lora_scale)

    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


class StableDiffusionXLPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.default_sample_size = self.unet.config.sample_size

        self.text_encoder.text_model.config.bos_token_id = 49406
        self.text_encoder_2.text_model.config.bos_token_id = 49406
        self.tokenizer.bos_token_id = 49406
        self.tokenizer_2.bos_token_id = 49406
        self.text_encoder.text_model.config.eos_token_id = 49407
        self.text_encoder_2.text_model.config.eos_token_id = 49407
        self.tokenizer.eos_token_id = 49407
        self.tokenizer_2.eos_token_id = 49407
        self.text_encoder.text_model.config.pad_token_id = 1
        self.text_encoder_2.text_model.config.pad_token_id = 1
        self.tokenizer.pad_token_id = 1
        self.tokenizer_2.pad_token_id = 1

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is None:
            latents = randn_tensor(
                (
                    batch_size,
                    num_channels_latents,
                    int(height) // self.vae_scale_factor,
                    int(width) // self.vae_scale_factor,
                ),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        # scale the initial noise by the standard deviation required by the scheduler
        return latents * self.scheduler.init_noise_sigma

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @torch.inference_mode()
    def denoising_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        time_embeds: torch.Tensor,
        timestep: int,
    ):
        # predict the noise residual
        noise_pred_uncond, noise_pred_text = self.unet(
            self.scheduler.scale_model_input(torch.cat([latents] * 2), timestep),
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": time_embeds,
            },
            return_dict=False,
        )[0].chunk(2)

        # perform guidance and compute the previous noisy sample x_t -> x_t-1
        return self.scheduler.step(
            noise_pred_uncond
            + self.guidance_scale * (noise_pred_text - noise_pred_uncond),
            timestep,
            latents,
            return_dict=False,
        )[0]

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str] = None,
        prompt_2: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] = None,
        guidance_scale: float = 5.0,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        negative_pooled_prompt_embeds: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> Image.Image:
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        batch_size = 1

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
        )

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        elif sigmas is not None:
            self.scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = self.scheduler.timesteps
        else:
            raise ValueError("Either timesteps or sigmas must be provided.")

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare added time ids & embeddings
        time_embeds = torch.tensor(
            [[height, width, 0, 0, height, width] * 2],
            dtype=torch.long,
            device=device,
        ).repeat(batch_size * num_images_per_prompt, 1)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        for t in timesteps:
            latents = self.denoising_loop(
                latents,
                prompt_embeds,
                pooled_prompt_embeds,
                time_embeds,
                t,
            )
        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        if (
            hasattr(self.vae.config, "latents_mean")
            and self.vae.config.latents_mean is not None
        ):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, 4, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, 4, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = (
                latents * latents_std / self.vae.config.scaling_factor + latents_mean
            )
        else:
            latents /= self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]
        return Image.fromarray(
            (
                (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
                * 255
            )
            .round()
            .astype("uint8")[0]
        )

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: str | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | None = None,
        negative_prompt_2: str | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        negative_pooled_prompt_embeds: torch.Tensor | None = None,
        lora_scale: float | None = None,
        clip_skip: int | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(
            self, StableDiffusionXLLoraLoaderMixin
        ):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = (
            [self.tokenizer, self.tokenizer_2]
            if self.tokenizer is not None
            else [self.tokenizer_2]
        )
        text_encoders = (
            [self.text_encoder, self.text_encoder_2]
            if self.text_encoder is not None
            else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(
                prompts, tokenizers, text_encoders, strict=False
            ):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids

                prompt_embeds = text_encoder(
                    text_input_ids.to(device), output_hidden_states=True
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = (
            negative_prompt is None and self.config.force_zeros_for_empty_prompt
        )
        if (
            do_classifier_free_guidance
            and negative_prompt_embeds is None
            and zero_out_negative_prompt
        ):
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_prompt_2 = (
                batch_size * [negative_prompt_2]
                if isinstance(negative_prompt_2, str)
                else negative_prompt_2
            )

            uncond_tokens = [negative_prompt, negative_prompt_2]
            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(
                uncond_tokens, tokenizers, text_encoders, strict=False
            ):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(
                        negative_prompt, tokenizer
                    )

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the
                # final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(
                dtype=self.text_encoder_2.dtype, device=device
            )
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt,
        # using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt,
            # using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=self.text_encoder_2.dtype, device=device
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=self.unet.dtype, device=device
                )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            1, num_images_per_prompt
        ).view(bs_embed * num_images_per_prompt, -1)
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                1, num_images_per_prompt
            ).view(bs_embed * num_images_per_prompt, -1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
