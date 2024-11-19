import argparse
import asyncio
import base64
import gc
import logging
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from nudenet import NudeDetector
from PIL import Image
from pydantic import BaseModel

from battle_arena import AI, AestheticPredictor, PipelineWrapper

load_dotenv()
ROOT_DIR = Path(__file__).parent

logger = logging.getLogger("uvicorn")


class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: str
    source: Literal["user", "ai"]


class RegisterRequest(BaseModel):
    name: str


class Response(BaseModel):
    imageBase64: str
    aesthetic_score: float


class LLMResponse(BaseModel):
    prompt: str
    negative_prompt: str
    comment: str
    imageBase64: str
    aesthetic_score: float


class BestResponse(BaseModel):
    userImageBase64: str
    user_aesthetic_score: float
    aiImageBase64: str
    ai_aesthetic_score: float


class EndResponse(BaseModel):
    rankings: list[dict[str, str | float | bool]]
    yourRank: int


class BattleArenaAPI:
    def __init__(
        self,
        pretrained_model_link_or_path: str | os.PathLike,
        lora_model_link_or_path: str | os.PathLike | None = None,
        height: int = 768,
        width: int = 1152,
    ) -> None:
        self.pipeline = PipelineWrapper(
            pretrained_model_link_or_path,
            lora_model_link_or_path,
            height=height,
            width=width,
        )
        self.aesthetic_predictor = AestheticPredictor(height=height, width=width)
        self.nude_detector = NudeDetector()
        self.ai = AI()
        self.app = FastAPI()

        self.app.add_api_route(
            "/api/txt2img", self.txt2img, methods=["POST"], response_model=Response
        )
        self.app.add_api_route(
            "/api/generate", self.generate, methods=["GET"], response_model=LLMResponse
        )
        self.app.add_api_route(
            "/api/best", self._best, methods=["GET"], response_model=BestResponse
        )
        self.app.add_api_route(
            "/api/end", self._end, methods=["GET"], response_model=EndResponse
        )
        self.app.add_api_route("/api/register", self._register, methods=["POST"])
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.mount(
            "/txt2img",
            StaticFiles(directory=ROOT_DIR / "view" / "txt2img" / "build", html=True),
            name="txt2img",
        )

        # cache
        self.user_prompts = []
        self.user_negative_prompts = []
        self.user_aesthetic_scores = []
        self.user_images_base64 = []

        self.ai_prompts = []
        self.ai_negative_prompts = []
        self.ai_aesthetic_scores = []
        self.ai_images_base64 = []

        # never ganna give you up
        self._never_gonna_let_you_down = Image.open(
            ROOT_DIR / "assets" / "never_gonna_let_you_down.webp"
        )
        # convert to base64
        buffered = BytesIO()
        self._never_gonna_let_you_down.save(buffered, format="WEBP")
        self._never_gonna_let_you_down_base64 = base64.b64encode(
            buffered.getvalue()
        ).decode("utf-8")

        self._txt2img_lock = asyncio.Lock()

        # leaderboard
        self.leaderboard: pd.DataFrame = pd.DataFrame(
            columns=[
                "name",
                "aesthetic_score",
                "imageBase64",
                "prompt",
                "negative_prompt",
            ]
        )
        self.leaderboard_path = ROOT_DIR / "data" / "leaderboard.json"
        if self.leaderboard_path.exists():
            self.leaderboard = pd.read_json(self.leaderboard_path).sort_values(
                by="aesthetic_score", ascending=False
            )

    async def txt2img(self, request: Txt2ImgRequest) -> Response:
        image_base64, aesthetic_score = await self._txt2img(
            request.prompt, request.negative_prompt
        )
        if aesthetic_score != 0:
            self.user_prompts.append(request.prompt)
            self.user_negative_prompts.append(request.negative_prompt)
            self.user_aesthetic_scores.append(aesthetic_score)
            self.user_images_base64.append(image_base64)
        return Response(imageBase64=image_base64, aesthetic_score=aesthetic_score)

    async def generate(self) -> LLMResponse:
        loop = asyncio.get_event_loop()
        rand = random.random()
        if not self.ai_aesthetic_scores:
            ai_response = await loop.run_in_executor(
                None,
                self.ai,
            )
        elif rand < 0.3:
            ai_best_idx = self.ai_aesthetic_scores.index(max(self.ai_aesthetic_scores))
            ai_response = await loop.run_in_executor(
                None,
                self.ai,
                self.ai_aesthetic_scores[-1],
                self.ai_aesthetic_scores[ai_best_idx],
                None,
                None,
                None,
                None,
                None,
                self.user_aesthetic_scores[-1],
                self.user_prompts[-1],
                self.user_negative_prompts[-1],
            )
        elif rand < 0.6 and self.user_aesthetic_scores:
            ai_best_idx = self.ai_aesthetic_scores.index(max(self.ai_aesthetic_scores))
            user_best_idx = self.user_aesthetic_scores.index(
                max(self.user_aesthetic_scores)
            )
            ai_response = await loop.run_in_executor(
                None,
                self.ai,
                self.ai_aesthetic_scores[-1],
                self.ai_aesthetic_scores[ai_best_idx],
                self.ai_prompts[ai_best_idx],
                self.ai_negative_prompts[ai_best_idx],
                self.user_aesthetic_scores[user_best_idx],
                self.user_prompts[user_best_idx],
                self.user_negative_prompts[user_best_idx],
                None,
                None,
                None,
            )
        else:
            ai_best_idx = self.ai_aesthetic_scores.index(max(self.ai_aesthetic_scores))
            user_best_idx = self.user_aesthetic_scores.index(
                max(self.user_aesthetic_scores)
            )
            ai_response = await loop.run_in_executor(
                None,
                self.ai,
                self.ai_aesthetic_scores[-1],
                self.ai_aesthetic_scores[ai_best_idx],
                None,
                None,
                self.user_aesthetic_scores[user_best_idx],
                None,
                None,
                None,
                None,
                None,
            )

        image_base64, aesthetic_score = await self._txt2img(
            ai_response.prompt, ai_response.negative_prompt
        )
        self.ai_prompts.append(ai_response.prompt)
        self.ai_negative_prompts.append(ai_response.negative_prompt)
        self.ai_aesthetic_scores.append(aesthetic_score)
        self.ai_images_base64.append(image_base64)

        return LLMResponse(
            prompt=ai_response.prompt,
            negative_prompt=ai_response.negative_prompt,
            comment=ai_response.comment,
            imageBase64=image_base64,
            aesthetic_score=aesthetic_score,
        )

    async def _txt2img(self, prompt: str, negative_prompt: str) -> tuple[str, float]:
        async with self._txt2img_lock:
            loop = asyncio.get_event_loop()
            output_image = await loop.run_in_executor(
                None,
                self.pipeline,
                prompt,
                negative_prompt,
            )
            if not self._detect_nudity(np.array(output_image)):
                aesthetic_score = await loop.run_in_executor(
                    None,
                    self.aesthetic_predictor,
                    [output_image],
                )
                aesthetic_score = aesthetic_score[0]
                buffered = BytesIO()
                output_image.save(buffered, format="WEBP")
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                image_base64 = self._never_gonna_let_you_down_base64
                aesthetic_score = 0

            return image_base64, aesthetic_score

    async def _best(self) -> BestResponse:
        if not self.user_aesthetic_scores and not self.ai_aesthetic_scores:
            return BestResponse(
                userImageBase64="",
                user_aesthetic_score=0,
                aiImageBase64="",
                ai_aesthetic_score=0,
            )
        elif not self.user_aesthetic_scores:
            ai_best_idx = self.ai_aesthetic_scores.index(max(self.ai_aesthetic_scores))
            return BestResponse(
                userImageBase64="",
                user_aesthetic_score=0,
                aiImageBase64=self.ai_images_base64[ai_best_idx],
                ai_aesthetic_score=self.ai_aesthetic_scores[ai_best_idx],
            )
        elif not self.ai_aesthetic_scores:
            user_best_idx = self.user_aesthetic_scores.index(
                max(self.user_aesthetic_scores)
            )
            return BestResponse(
                userImageBase64=self.user_images_base64[user_best_idx],
                user_aesthetic_score=self.user_aesthetic_scores[user_best_idx],
                aiImageBase64="",
                ai_aesthetic_score=0,
            )
        else:
            ai_best_idx = self.ai_aesthetic_scores.index(max(self.ai_aesthetic_scores))
            user_best_idx = self.user_aesthetic_scores.index(
                max(self.user_aesthetic_scores)
            )

            return BestResponse(
                userImageBase64=self.user_images_base64[user_best_idx],
                user_aesthetic_score=self.user_aesthetic_scores[user_best_idx],
                aiImageBase64=self.ai_images_base64[ai_best_idx],
                ai_aesthetic_score=self.ai_aesthetic_scores[ai_best_idx],
            )

    async def _end(self, background_tasks: BackgroundTasks) -> EndResponse:
        # update leaderboard with current user
        if self.user_aesthetic_scores:
            user_best_idx = self.user_aesthetic_scores.index(
                max(self.user_aesthetic_scores)
            )
            self.leaderboard = pd.concat(
                [
                    self.leaderboard,
                    pd.DataFrame(
                        {
                            "name": "あなた",
                            "aesthetic_score": self.user_aesthetic_scores[
                                user_best_idx
                            ],
                            "imageBase64": self.user_images_base64[user_best_idx],
                            "prompt": self.user_prompts[user_best_idx],
                            "negative_prompt": self.user_negative_prompts[
                                user_best_idx
                            ],
                        },
                        index=[0],
                    ),
                ]
            ).sort_values(by="aesthetic_score", ascending=False)

        # get rank of current user
        user_rank = self.leaderboard[self.leaderboard["name"] == "あなた"].index[0] + 1
        # get top 20 + current user to display
        # if あなた is not in top 20, append with preserving index (rank)
        if user_rank > 20:
            leaderboard = self.leaderboard.iloc[:20].assign(isCurrentUser=False)
            leaderboard = pd.concat(
                [
                    leaderboard,
                    pd.DataFrame(
                        {
                            "name": "あなた",
                            "aesthetic_score": self.user_aesthetic_scores[
                                user_best_idx
                            ],
                            "imageBase64": self.user_images_base64[user_best_idx],
                            "prompt": self.user_prompts[user_best_idx],
                            "negative_prompt": self.user_negative_prompts[
                                user_best_idx
                            ],
                            "isCurrentUser": True,
                        },
                        index=[user_rank - 1],
                    ),
                ]
            )
        else:
            leaderboard = self.leaderboard.iloc[:20].assign(isCurrentUser=False)
            leaderboard.loc[user_rank - 1].isCurrentUser = True
        leaderboard = leaderboard.to_dict(orient="records")

        background_tasks.add_task(self._reset)

        return EndResponse(rankings=leaderboard, yourRank=user_rank)

    def _reset(self) -> None:
        self.user_prompts = []
        self.user_negative_prompts = []
        self.user_aesthetic_scores = []
        self.user_images_base64 = []

        self.ai_prompts = []
        self.ai_negative_prompts = []
        self.ai_aesthetic_scores = []
        self.ai_images_base64 = []

        self.ai.reset()

        gc.collect()
        torch.cuda.empty_cache()

        self.pipeline._warmup()
        self.aesthetic_predictor._warmup()

    def _register(self, request: RegisterRequest) -> dict:
        # rename "あなた" to user's name
        self.leaderboard.loc[self.leaderboard["name"] == "あなた", "name"] = (
            request.name
        )
        self.leaderboard.to_json(
            self.leaderboard_path, orient="records", indent=2, force_ascii=False
        )

        return {"status": "success"}

    def _detect_nudity(self, image: np.ndarray) -> bool:
        res = self.nude_detector.detect(image)
        for r in res:
            if r["class"] in [
                "FEMALE_BREAST_EXPOSED",
                "FEMALE_GENITALIA_EXPOSED",
                "BUTTOCKS_EXPOSED",
                "ANUS_EXPOSED",
                "MALE_GENITALIA_EXPOSED",
            ]:
                return True

    def _calc_score_diff(self) -> float:
        if not self.user_aesthetic_scores or not self.ai_aesthetic_scores:
            return 0
        user_score = max(self.user_aesthetic_scores)
        ai_score = max(self.ai_aesthetic_scores)
        return user_score - ai_score


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pretrained_model_link_or_path", type=str, default="OnomaAIResearch/Illustrious-xl-early-release-v0")
    argparser.add_argument("--lora_model_link_or_path", type=str, default=None)
    argparser.add_argument("--server_host", type=str, default="0.0.0.0")
    argparser.add_argument("--server_port", type=int, default=9090)

    args = argparser.parse_args()

    api = BattleArenaAPI(
        pretrained_model_link_or_path=args.pretrained_model_link_or_path,
        lora_model_link_or_path=args.lora_model_link_or_path,
    )
    uvicorn.run(api.app, host=args.server_host, port=args.server_port)
