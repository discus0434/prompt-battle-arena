import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image


class AestheticPredictor:
    def __init__(self, height: int = 480, width: int = 800) -> None:
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._height = height
        self._width = width
        self.model = self.model.to(torch.bfloat16).cuda()
        self.model = torch.compile(self.model)

    def __call__(self, images: list[Image.Image]) -> list[float]:
        pixel_values = (
            self.preprocessor(images=images, return_tensors="pt")
            .pixel_values.to(torch.bfloat16)
            .cuda()
        )
        with torch.inference_mode():
            scores = self.model(pixel_values).logits.squeeze().float().cpu().numpy()

        scores = scores.astype(float).tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores

    def _warmup(self) -> None:
        pixel_values = (
            self.preprocessor(
                images=[Image.new("RGB", (self._width, self._height))],
                return_tensors="pt",
            )
            .pixel_values.to(torch.bfloat16)
            .cuda()
        )
        for _ in range(10):
            self.model(pixel_values)
