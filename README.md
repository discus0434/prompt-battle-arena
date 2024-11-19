# Prompt Battle Arena

## 概要

第2回生成AIなんでも展示会で出展したアプリのソースコードです。
LLMと同時にリアルタイムで画像を生成し、制限時間内でどちらがより審美性スコアの高い画像を生成できるかを競うアプリです。

## ゲーム説明

- 1試合は120秒間です。
- 画面左側がプレイヤー画面、右側がAI画面です。
- プレイヤー画面の下部には2つのテキスト入力欄があり、それぞれが「プロンプト」「ネガティブプロンプト」の入力欄です。入力に応じてリアルタイムに画像が生成されます。
- 画面上部には「審美性スコア」が表示されます。審美性スコアは、画像の美しさを(ある一定の観点から)表す指標で、このゲームでは審美性スコアが高いほど良い画像とされます。制限時間の終了時に、より高い審美性スコアを持つ画像を生成したプレイヤーが勝利となります。
- LLMは10秒に1回、プロンプトとネガティブプロンプトを考えて入力し、画像を生成します。オリジナルでプロンプトを考えることもあれば、プレイヤーのプロンプトの一部をパクることもあります。


## 導入方法

1. このリポジトリをクローンします。

    ```bash
    git clone https://github.com/discus0434/battle-arena.git
    cd battle-arena
    ```

2. 依存関係をインストールします。

    2.1 venvの作成

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

    2.2 依存関係のインストール


    ```bash
    # CUDA 11.8の場合
    pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
    make install-dependency
    ```


    ```bash
    # CUDA 12.1の場合
    pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu121
    make install-dependency
    ```

    (Optional) 2.3 Stable Fastとlibtcmallocのインストール

    ```bash
    pip install -v -U git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast
    sudo apt install google-perftools libgoogle-perftools-dev
    ```

3. OpenAIのAPIキーを`.env`ファイルに記述します。

    ```bash
    # `sk-XXX`の部分は自分のAPIキーに置き換えてください
    echo "OPENAI_API_KEY=sk-XXX" >> .env
    ```

4. アプリを起動します。

    ```bash
    make launch
    ```
