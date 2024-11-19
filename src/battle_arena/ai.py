import copy
import inspect
import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class AIResponse:
    prompt: str
    negative_prompt: str
    comment: str


class AI:
    def __init__(self):
        self.client = OpenAI(timeout=20)
        self.initial_messages = [
            {
                "role": "system",
                "content": inspect.cleandoc(
                    """
                    あなたはAIで、ユーザーとバトルしてもらいます。
                    対戦方式は、画像生成AIに指示を与え、最も高い審美性を持つ画像を生成した方が勝ちです。
                    画像生成AIには、Stable Diffusion XL が使用されます。このモデルは、danbooru2023データセットで事前学習されているため、アニメスタイルの画像を生成することが得意で、danbooruのtagをカンマ区切りに羅列したものをプロンプトとして受け付けます。
                    例えば、1人の赤い髪の女性キャラクターを生成する場合、"1girl, red_hair"というプロンプトを使用します。
                    また、画像生成AIはClassifier Free Guindaceを採用しており、プロンプトとは別に、望ましくない特徴を指定する"ネガティブプロンプト"も受け付けます。

                    ネットで拾ってきた文章では、以下のようなプロンプトの書き方が有効だとされています。しかし、審美性を競うバトルでは、この書き方が有効かどうかはわかりません。あなたの戦略に合わせて、プロンプトを考えてください。

                    ```
                    ポジティブプロンプト
                    masterpiece, best quality, good quality, newest

                    amazing qualityも有効だが、本家で使用すると高確率でNSFWな画像になる
                    自然言語のプロンプトに対応していないのでSD1.5時代のプロンプトの書き方が有効

                    品質
                    masterpiece > best quality > good quality > average quality > bad quality > worst quality

                    年代
                    oldest(~2017年)、old(~2019年)、modern(~2020年)、recent(~2022年)、newest(~2023年)

                    レーティング
                    general→sensitive→questionable→explicit

                    人間を描写する際のプロンプトの書く順番(絶対ではない)
                    ・画面の上から順に記述する
                    ・面積の広いものから順番に記述する
                        smile→open eyes→closed mouth

                    ・外見の特徴→動作の順
                    1boy and 1girlと言った書き方よりcoupleや1boy,1girl,と言った書き方の方が良い？

                    人数
                    ↓
                    キャラ名＋作品名
                    ↓
                    レーティング
                    ↓
                    プロンプトの主文
                    ↓
                    画風→品質→年代
                    ↓
                    ※構図・カメラ？

                    ネガティブプロンプト
                    lowres,worst quality, bad quality,bad anatomy,sketch,jpeg artifacts,signature,watermark,old,oldest
                    ```
                    (出典: https://note.com/robai104/n/n7a4801fb422c)

                    ここで、例示とともに出力のパターンについてもお伝えします。
                    例えば、あなたがハイクオリティーな1人の赤い髪の女性キャラクターを生成したい場合、以下のような出力を行うことができます。

                    ```json
                    {
                        "comment": "このプロンプトは、1人の赤い髪の女性キャラクターを生成するためのものです。プロンプトのmasterpiece, high_qualityというキーワードは、生成される画像が最高品質であることを示し、negative_promptのbad, ugly, worst_quality, low_qualityというキーワードは、生成される画像に含まれて欲しくない特徴を示しています。",
                        "prompt": "1girl, red_hair, masterpiece, high_quality",
                        "negative_prompt": "bad, ugly, worst_quality, low_quality",
                    }
                    ```

                    つまり、JSON形式でpromptとnegative_promptを指定します。commentは、プロンプトの説明を記述するためのフィールドです。私が書いたcommentの例はとても簡単なものですが、あなたが書くcommentは、プロンプトの意図や目的を明確に示し、いくらかの遊び心を持たせることが望ましいです。

                    また、バトルが始まると、画像生成AIは、あなたのプロンプトに対して画像を生成し、その画像の審美性を評価します。評価は、画像の品質、美しさ、クオリティー、特徴の適合度などを総合的に判断して行われ、0〜10点のスケールで評価されます。基本的には5.5点以上は好ましいと言えますが、とても素晴らしい画像は8点を叩き出すこともあります。そして、最終的に、あなたとユーザーの間で、画像生成AIが生成した画像の審美性を競い合います。
                    あなたは、過去に生成した自分の画像の審美性の点数がフィードバックとして与えられます。そして、そのフィードバックを元に、次のプロンプトを考えることができます。
                    また、たまに、あなたが対戦しているユーザーのプロンプト、ネガティブプロンプトと、それによって生成された画像の審美性の点数が与えられることもあります。その場合、あなたは、それをパクることもできますし、自分のプロンプトを考える際の参考にすることもできます。好きに使ってください。

                    それでは、バトルを始めましょう。まず、戦略についてのコメント、プロンプト、ネガティブプロンプトを先述したJSON形式で入力してください。
                    """
                ),
            }
        ]
        self.messages = copy.deepcopy(self.initial_messages)

    def __call__(
        self,
        aesthetic_score: float | None = None,
        best_aesthetic_score: float | None = None,
        best_prompt: str | None = None,
        best_negative_prompt: str | None = None,
        opponent_best_aesthetic_score: float | None = None,
        opponent_best_prompt: str | None = None,
        opponent_best_negative_prompt: str | None = None,
        opponent_aesthetic_score: float | None = None,
        opponent_prompt: str | None = None,
        opponent_negative_prompt: str | None = None,
    ) -> str:
        if (
            aesthetic_score is not None
            and best_aesthetic_score is not None
            and opponent_best_aesthetic_score is not None
        ):
            self.messages.append(
                {
                    "role": "system",
                    "content": f"今回のあなたのプロンプトで生成した画像の審美性の点数は、{aesthetic_score:.2f}点でした。あなたの最高点は{best_aesthetic_score:.2f}点、対戦相手の最高点は{opponent_best_aesthetic_score:.2f}点です。",
                }
            )
        if (
            best_prompt is not None
            and best_negative_prompt is not None
            and opponent_best_prompt is not None
            and opponent_best_negative_prompt is not None
        ):
            self.messages.append(
                {
                    "role": "system",
                    "content": f"リマインドです。参考にしてください。あなたが最高点を出したプロンプトは、{best_prompt}、ネガティブプロンプトは、{best_negative_prompt}です。対戦相手の最高点を出したプロンプトは、{opponent_best_prompt}、ネガティブプロンプトは、{opponent_best_negative_prompt}です。",
                }
            )
        if (
            opponent_prompt is not None
            and opponent_negative_prompt is not None
            and opponent_aesthetic_score is not None
        ):
            self.messages.append(
                {
                    "role": "system",
                    "content": f"あなたの対戦相手が使用したプロンプトが最近{opponent_aesthetic_score:.2f}点を出しています。プロンプトは、{opponent_prompt}、ネガティブプロンプトは、{opponent_negative_prompt}です。一方あなたの最高点は{best_aesthetic_score:.2f}点です。最高点によっては無視してもいいでしょうし、参考にしてもいいでしょう。",
                }
            )
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4o-mini", messages=self.messages
                )

                response = completion.choices[0].message.content

                if _response := self._parse_response(response):
                    self.messages.append({"role": "assistant", "content": response})
                    return _response
            except json.JSONDecodeError:
                self.messages.append(
                    {
                        "role": "system",
                        "content": inspect.cleandoc(
                            """
                            リマインドですが、出力形式は以下のようになります。
                            ```json
                            {
                                "comment": "このプロンプトは、1人の赤い髪の女性キャラクターを生成するためのものです。プロンプトのmasterpiece, high_qualityというキーワードは、生成される画像が最高品質であることを示し、negative_promptのbad, ugly, worst_quality, low_qualityというキーワードは、生成される画像に含まれて欲しくない特徴を示しています。",
                                "prompt": "1girl, red_hair, masterpiece, high_quality",
                                "negative_prompt": "bad, ugly, worst_quality, low_quality",
                            }
                            ```
                            それでは、再度お願いします。
                            """
                        ),
                    }
                )
                print("JSONDecodeError")
            except TimeoutError:
                print("TimeoutError")

    def reset(self) -> None:
        self.messages = copy.deepcopy(self.initial_messages)

    def _parse_response(self, response: str) -> AIResponse | None:
        json_objects = self._extract_json_from_text(response)

        if not json_objects:
            return None

        if (
            "prompt" in json_objects[0]
            and "negative_prompt" in json_objects[0]
            and "comment" in json_objects[0]
        ):
            return AIResponse(
                prompt=json_objects[0]["prompt"].replace("_", " "),
                negative_prompt=json_objects[0]["negative_prompt"].replace("_", " "),
                comment=json_objects[0]["comment"],
            )

    def _extract_json_from_text(self, text: str) -> list[dict[str, Any]]:
        """
        Extracts JSON objects from the given text.

        Parameters
        ----------
        text : str
            The text from which to extract JSON objects.

        Returns
        -------
        List[Dict[str, Any]]
            A list of extracted JSON objects.
        """
        json_objects = []
        stack = []
        start_index = None
        for i, char in enumerate(text):
            if char == "{":
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        json_str = text[start_index : i + 1]
                        # シングルクォートをダブルクォートに変換
                        json_fixed = json_str.replace("'", '"')
                        try:
                            parsed_json = json.loads(json_fixed)
                            json_objects.append(parsed_json)
                        except json.JSONDecodeError:
                            pass
        return json_objects
