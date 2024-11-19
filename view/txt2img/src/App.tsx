import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  useLayoutEffect,
} from "react";

import {
  TextField,
  Button,
  Dialog,
  DialogContent,
  DialogTitle,
} from "@mui/material";
import { styled } from "@mui/system";

// スタイル定義

// 全体のコンテナ
const Container = styled("div")({
  backgroundColor: "#F5F5F5",
  width: "100vw",
  height: "100vh",
  display: "flex",
  flexDirection: "column",
  margin: "0",
  color: "#333333",
  fontFamily: "'Poppins', sans-serif",
});

// タイマーのコンテナ
const TimerContainer = styled("div")({
  width: "100%",
  height: "10%",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
});

const Timer = styled("div")({
  fontSize: "2rem",
  fontWeight: "bold",
  color: "#FF0000",
  textAlign: "center",
});

// バトル用のコンテナ
const BattleContainer = styled("div")({
  width: "100%",
  height: "90%", // 高さを90%に設定
  display: "flex",
  justifyContent: "center",
  alignItems: "stretch",
  maxWidth: "1200px",
  margin: "0 auto",
});

// 左右のカラム
const Column = styled("div")({
  flex: "1",
  backgroundColor: "#FFFFFF",
  display: "flex",
  flexDirection: "column",
  padding: "10px",
  boxSizing: "border-box",
  alignItems: "stretch", // 子要素が横幅いっぱいに伸びる
  position: "relative",
  overflow: "hidden",
});

// スコア表示
const Score = styled("div")({
  marginBottom: "10px",
  fontWeight: "bold",
  fontSize: "1rem",
  color: "#555555",
});

// キャンバスコンテナ
const StyledCanvasContainer = styled("div")({
  width: "90%",
  flexGrow: 1,
  display: "flex",
  justifyContent: "center",
  alignItems: "stretch", // または "flex-start"
  position: "relative",
});
const StyledCanvas = styled("canvas")({
  width: "100%", // 親要素に合わせてキャンバスを拡大
  height: "90%",
});

// テキストフィールド
const StyledTextField = styled(TextField)({
  marginBottom: "10px",
  width: "90%",
  borderRadius: "5px",
  "& .MuiOutlinedInput-root": {
    borderRadius: "5px",
  },
  "& .MuiInputBase-input": {
    fontSize: "0.9rem",
    lineHeight: "1.2",
  },
});

// コメントボックスまたはスペース
const CommentBox = styled("div")({
  width: "90%",
  padding: "10px",
  borderRadius: "5px",
  backgroundColor: "#FFEFD5",
  color: "#333333",
  fontSize: "0.85rem",
  lineHeight: "1.4",
  boxSizing: "border-box",
  overflowY: "auto",
  maxHeight: "200px",
  minHeight: "200px",
  marginBottom: "10px",
});

// ローディング表示（透明度調整）
const LoadingOverlay = styled("div")(({ visible }: { visible: boolean }) => ({
  position: "absolute",
  top: 0,
  left: 0,
  width: "100%",
  height: "100%",
  backgroundColor: visible ? "rgba(255, 255, 255, 0.8)" : "transparent",
  display: visible ? "flex" : "none",
  alignItems: "center",
  justifyContent: "center",
  pointerEvents: visible ? "auto" : "none",
  transition: "background-color 0.3s ease",
  zIndex: 10,
}));

const LoadingText = styled("div")({
  fontSize: "1.2rem",
  color: "#333333",
  display: "flex",
  alignItems: "center",
});

const Dots = styled("span")({
  display: "inline-block",
  marginLeft: "5px",
  "& span": {
    animation: "blink 1.5s infinite",
    fontSize: "1.5rem",
  },
  "& span:nth-of-type(2)": {
    animationDelay: "0.3s",
  },
  "& span:nth-of-type(3)": {
    animationDelay: "0.6s",
  },
  "@keyframes blink": {
    "0%": { opacity: 0 },
    "50%": { opacity: 1 },
    "100%": { opacity: 0 },
  },
});

// 結果表示
const ResultContainer = styled("div")({
  width: "100%",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  fontSize: "2rem",
  fontWeight: "bold",
  color: "#333333",
});

// ランキングコンテナ
const RankingContainer = styled("div")({
  width: "100%",
  maxWidth: "800px",
  margin: "0 auto",
  padding: "20px",
  boxSizing: "border-box",
  overflowY: "auto",
  flexGrow: 1,
});

const RankingItem = styled("div")<{ isYou?: boolean }>(({ isYou }) => ({
  display: "flex",
  alignItems: "center",
  padding: "10px",
  backgroundColor: isYou ? "#FFFACD" : "#FFFFFF", // ユーザーの行をハイライト
  marginBottom: "10px",
  cursor: "pointer",
  borderRadius: "5px",
  boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
}));

const RankingImage = styled("img")({
  width: "50px",
  height: "50px",
  borderRadius: "5px",
  marginRight: "10px",
  objectFit: "cover",
});

const RankingText = styled("div")({
  flexGrow: 1,
});

const RankingName = styled("div")({
  fontWeight: "bold",
  fontSize: "1rem",
  color: "#333",
});

const RankingScore = styled("div")({
  fontSize: "0.9rem",
  color: "#666",
});

// モーダル内の画像
const ModalImage = styled("img")({
  width: "100%",
  height: "auto",
  maxWidth: "800px",
  objectFit: "contain",
  marginBottom: "20px",
});

function App() {
  // 状態管理
  // 画面状態を管理するステートを追加
  const [stage, setStage] = useState<
    "start" | "countdown" | "battle" | "result" | "next"
  >("start");
  const [countdown, setCountdown] = useState(5);
  const [battleTime, setBattleTime] = useState(120);
  const [winner, setWinner] = useState<"user" | "ai" | "draw">("draw");

  // 左側（ユーザー入力）の状態管理
  const [inputPrompt, setInputPrompt] = useState("");
  const [inputNegativePrompt, setInputNegativePrompt] = useState("");
  const [leftAestheticScore, setLeftAestheticScore] = useState<number | null>(
    null
  );

  const [leftImageData, setLeftImageData] = useState<string | null>(null);
  const [rightImageData, setRightImageData] = useState<string | null>(null);

  // 右側（AI生成）の状態管理
  const [aiPrompt, setAiPrompt] = useState("");
  const [aiNegativePrompt, setAiNegativePrompt] = useState("");
  const [aiComment, setAiComment] = useState("");
  const [rightAestheticScore, setRightAestheticScore] = useState<number | null>(
    null
  );
  const [isGenerating, setIsGenerating] = useState(false);

  // 最高スコアと画像の状態管理
  const [userBestImage, setUserBestImage] = useState<string | null>(null);
  const [userBestScore, setUserBestScore] = useState<number | null>(null);
  const [aiBestImage, setAiBestImage] = useState<string | null>(null);
  const [aiBestScore, setAiBestScore] = useState<number | null>(null);

  // ランキングの状態管理
  const [rankings, setRankings] = useState<any[]>([]);
  const [yourRank, setYourRank] = useState<number | null>(null);
  const [isRegisteringName, setIsRegisteringName] = useState(false);
  const [newName, setNewName] = useState("");
  const [selectedRankingItem, setSelectedRankingItem] = useState<any | null>(
    null
  );
  const [isModalOpen, setIsModalOpen] = useState(false);

  // タイマーIDを保持するref
  const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const battleIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const aiTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Canvasの参照
  const leftCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rightCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const redrawLeftImage = useCallback(() => {
    if (leftCanvasRef.current && leftImageData) {
      const canvas = leftCanvasRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const img = new Image();
        img.onload = () => {
          if (canvas) {
            // コンテキストを取得（既にスケーリング済み）
            const ctx = canvas.getContext("2d");
            if (ctx) {
              // キャンバスの表示サイズを取得（デバイスピクセル比を考慮しない）
              const canvasWidth = canvas.width;
              const canvasHeight = canvas.height;

              // 描画領域をクリア
              ctx.clearRect(0, 0, canvas.width, canvas.height);

              // アスペクト比を維持して画像を描画
              const hRatio = canvasWidth / img.width;
              const vRatio = canvasHeight / img.height;
              const ratio = Math.min(hRatio, vRatio);
              const centerShiftX = (canvasWidth - img.width * ratio) / 2;
              const centerShiftY = (canvasHeight - img.height * ratio) / 2;

              // 座標とサイズをデバイスピクセル比に合わせて調整
              ctx.drawImage(
                img,
                0,
                0,
                img.width,
                img.height,
                centerShiftX,
                centerShiftY,
                img.width * ratio,
                img.height * ratio
              );
            }
          }
        };
        img.src = `data:image/webp;base64,${leftImageData}`;
      }
    }
  }, [leftImageData]);

  const redrawRightImage = useCallback(() => {
    if (rightCanvasRef.current && rightImageData) {
      const canvas = rightCanvasRef.current;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const img = new Image();
        img.onload = () => {
          if (canvas) {
            // コンテキストを取得（既にスケーリング済み）
            const ctx = canvas.getContext("2d");
            if (ctx) {
              // キャンバスの表示サイズを取得（デバイスピクセル比を考慮しない）
              const canvasWidth = canvas.width;
              const canvasHeight = canvas.height;

              // 描画領域をクリア
              ctx.clearRect(0, 0, canvas.width, canvas.height);

              // アスペクト比を維持して画像を描画
              const hRatio = canvasWidth / img.width;
              const vRatio = canvasHeight / img.height;
              const ratio = Math.min(hRatio, vRatio);
              const centerShiftX = (canvasWidth - img.width * ratio) / 2;
              const centerShiftY = (canvasHeight - img.height * ratio) / 2;

              // 座標とサイズをデバイスピクセル比に合わせて調整
              ctx.drawImage(
                img,
                0,
                0,
                img.width,
                img.height,
                centerShiftX,
                centerShiftY,
                img.width * ratio,
                img.height * ratio
              );
            }
          }
        };
        img.src = `data:image/webp;base64,${rightImageData}`;
      }
    }
  }, [rightImageData]);

  useEffect(() => {
    if (leftImageData) {
      redrawLeftImage();
    }
  }, [leftImageData, redrawLeftImage]);

  useEffect(() => {
    if (rightImageData) {
      redrawRightImage();
    }
  }, [rightImageData, redrawRightImage]);

  const updateCanvasSize = useCallback(() => {
    if (leftCanvasRef.current) {
      const canvas = leftCanvasRef.current;
      const parent = canvas.parentElement;

      if (parent) {
        // キャンバスの表示サイズを取得
        const { width, height } = parent.getBoundingClientRect();

        // キャンバスの描画解像度を設定
        canvas.width = width;
        canvas.height = height;

        // CSSのサイズを維持
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }
    }

    if (rightCanvasRef.current) {
      const canvas = rightCanvasRef.current;
      const parent = canvas.parentElement;

      if (parent) {
        const { width, height } = parent.getBoundingClientRect();
        canvas.width = width;
        canvas.height = height;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }
    }
    redrawLeftImage();
    redrawRightImage();
  }, [redrawLeftImage, redrawRightImage]);

  // スタートボタンを押したときのハンドラを修正
  const handleStart = () => {
    setStage("countdown");
    setCountdown(5);
  };

  const handleLeftCanvasRef = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      leftCanvasRef.current = canvas;
      if (canvas) {
        console.log("Left canvas mounted");
        updateCanvasSize();
      }
    },
    []
  );
  const handleRightCanvasRef = useCallback(
    (canvas: HTMLCanvasElement | null) => {
      rightCanvasRef.current = canvas;
      if (canvas) {
        console.log("Right canvas mounted");
        updateCanvasSize();
      }
    },
    []
  );

  useEffect(() => {
    window.addEventListener("resize", updateCanvasSize);
    return () => {
      window.removeEventListener("resize", updateCanvasSize);
    };
  }, [updateCanvasSize]);

  // 非同期処理のフラグ
  const isFetchingLeftRef = useRef(false);

  // キューに入っているプロンプト
  const pendingLeftPromptRef = useRef<{
    prompt: string;
    negativePrompt: string;
  } | null>(null);

  // コンポーネントのマウント状態を管理
  const isMountedRef = useRef(true);

  // 最新のstageを保持するrefを追加
  const stageRef = useRef(stage);
  useEffect(() => {
    stageRef.current = stage;
  }, [stage]);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      // コンポーネントのアンマウント時にタイマーをクリア
      if (countdownIntervalRef.current)
        clearInterval(countdownIntervalRef.current);
      if (battleIntervalRef.current) clearInterval(battleIntervalRef.current);
      if (aiTimeoutRef.current) clearTimeout(aiTimeoutRef.current);
    };
  }, []);

  // ステージの変化に応じてタイマーを管理
  useEffect(() => {
    if (stage === "countdown") {
      // カウントダウンを開始
      let countdownValue = 5;
      setCountdown(countdownValue);

      // 既存のカウントダウンタイマーをクリア
      if (countdownIntervalRef.current)
        clearInterval(countdownIntervalRef.current);

      countdownIntervalRef.current = setInterval(() => {
        countdownValue -= 1;
        setCountdown(countdownValue);
        if (countdownValue === 0) {
          if (countdownIntervalRef.current)
            clearInterval(countdownIntervalRef.current);
          setStage("battle");
        }
      }, 1000);

      // クリーンアップ関数でタイマーをクリア
      return () => {
        if (countdownIntervalRef.current)
          clearInterval(countdownIntervalRef.current);
      };
    }

    if (stage === "battle") {
      // バトルタイマーを開始
      let timeLeft = 120;
      setBattleTime(timeLeft);

      // 既存のバトルタイマーをクリア
      if (battleIntervalRef.current) clearInterval(battleIntervalRef.current);

      battleIntervalRef.current = setInterval(() => {
        timeLeft -= 1;
        setBattleTime(timeLeft);

        if (timeLeft === 0) {
          if (battleIntervalRef.current)
            clearInterval(battleIntervalRef.current);
          // 勝者を判定
          determineWinner();
        }
      }, 1000);

      // 初回のAI画像生成を開始
      fetchAIImage();

      // クリーンアップ関数でタイマーをクリア
      return () => {
        if (battleIntervalRef.current) clearInterval(battleIntervalRef.current);
        if (aiTimeoutRef.current) clearTimeout(aiTimeoutRef.current);
      };
    } else {
      // stageが"battle"以外になったときに、AI画像生成のタイマーをクリア
      if (aiTimeoutRef.current) clearTimeout(aiTimeoutRef.current);
    }
  }, [stage]); // stageの変化に応じて実行

  // AIの画像とスコアを更新する関数
  const updateAICanvasAndScore = useCallback(
    (imageBase64: string, aestheticScore: number) => {
      setRightAestheticScore(aestheticScore);
      setRightImageData(imageBase64);
    },
    []
  );

  // 最高スコアと画像を取得する関数
  const fetchBestScores = useCallback(async () => {
    try {
      const response = await fetch("/api/best");
      const data = await response.json();
      const {
        userImageBase64,
        user_aesthetic_score,
        aiImageBase64,
        ai_aesthetic_score,
      } = data;

      if (isMountedRef.current) {
        setUserBestImage(userImageBase64);
        setUserBestScore(user_aesthetic_score);
        setAiBestImage(aiImageBase64);
        setAiBestScore(ai_aesthetic_score);
      }

      // 最新のスコアを返す
      return {
        user_aesthetic_score,
        ai_aesthetic_score,
      };
    } catch (error) {
      console.error("最高スコアの取得中にエラーが発生しました:", error);
      return null;
    }
  }, []);

  // fetchImage関数（ユーザー側のみ使用するように修正）
  const fetchImage = useCallback(
    async (prompt: string, negativePrompt: string): Promise<void> => {
      isFetchingLeftRef.current = true;

      try {
        const response = await fetch("/api/txt2img", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: prompt,
            negative_prompt: negativePrompt,
            source: "user", // ユーザー側の識別子
          }),
        });
        const data = await response.json();
        const imageBase64 = data.imageBase64;
        const aestheticScore = data.aesthetic_score;

        if (isMountedRef.current) {
          setLeftAestheticScore(aestheticScore);
          setLeftImageData(imageBase64); // 画像データを状態に保存
          // 画像生成後に最高スコアを取得
          await fetchBestScores();
        }
      } catch (error) {
        if (error instanceof Error) {
          console.error("画像の取得中にエラーが発生しました:", error);
        }
      } finally {
        isFetchingLeftRef.current = false;
        if (
          pendingLeftPromptRef.current &&
          (pendingLeftPromptRef.current.prompt !== prompt ||
            pendingLeftPromptRef.current.negativePrompt !== negativePrompt)
        ) {
          const nextPrompt = pendingLeftPromptRef.current;
          pendingLeftPromptRef.current = null;
          fetchImage(nextPrompt.prompt, nextPrompt.negativePrompt);
        }
      }
    },
    [fetchBestScores]
  );

  // AIの画像生成処理を修正
  const fetchAIImage = useCallback(() => {
    if (!isMountedRef.current || stageRef.current !== "battle") return;
    setIsGenerating(true);

    fetch("/api/generate")
      .then((response) => response.json())
      .then(async (data) => {
        if (isMountedRef.current) {
          setAiPrompt(data.prompt);
          setAiNegativePrompt(data.negative_prompt);
          setAiComment(data.comment);

          // 画像とスコアを取得して更新
          const imageBase64 = data.imageBase64;
          const aestheticScore = data.aesthetic_score;

          if (imageBase64 && typeof aestheticScore === "number") {
            updateAICanvasAndScore(imageBase64, aestheticScore);
          }

          // 最高スコアを取得
          await fetchBestScores();
        }
      })
      .catch((error) => {
        console.error("APIへのリクエスト中にエラーが発生しました:", error);
      })
      .finally(() => {
        if (isMountedRef.current) {
          setIsGenerating(false);
          // バトル中のみ次のAI画像生成を開始
          if (stageRef.current === "battle") {
            // 既存のタイマーをクリア
            if (aiTimeoutRef.current) clearTimeout(aiTimeoutRef.current);
            aiTimeoutRef.current = setTimeout(() => {
              fetchAIImage();
            }, 10000);
          }
        }
      });
  }, [fetchBestScores, updateAICanvasAndScore]);

  // ユーザー入力が変化したときの処理を修正
  useEffect(() => {
    if (stage !== "battle") return;

    if (inputPrompt.trim() === "" && inputNegativePrompt.trim() === "") {
      return;
    }

    if (!isFetchingLeftRef.current) {
      fetchImage(inputPrompt, inputNegativePrompt);
    } else {
      pendingLeftPromptRef.current = {
        prompt: inputPrompt,
        negativePrompt: inputNegativePrompt,
      };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputPrompt, inputNegativePrompt, stage]);

  // プロンプトの入力変更ハンドラ
  const handlePromptChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ): void => {
    setInputPrompt(event.target.value);
  };

  const handleNegativePromptChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ): void => {
    setInputNegativePrompt(event.target.value);
  };

  // 勝者を判定する関数
  const determineWinner = async () => {
    const bestScores = await fetchBestScores();

    if (bestScores) {
      const { user_aesthetic_score, ai_aesthetic_score } = bestScores;

      if (user_aesthetic_score !== null && ai_aesthetic_score !== null) {
        if (user_aesthetic_score > ai_aesthetic_score) {
          setWinner("user");
        } else if (user_aesthetic_score < ai_aesthetic_score) {
          setWinner("ai");
        } else {
          setWinner("draw");
        }
      } else {
        setWinner("draw");
      }
    } else {
      setWinner("draw");
    }

    setStage("result");
  };

  // 次の画面へボタンを押したときのハンドラを修正
  const handleNext = () => {
    // ランキングデータを取得
    fetch("/api/end")
      .then((response) => response.json())
      .then((data) => {
        setRankings(data.rankings);
        setYourRank(data.yourRank); // ユーザーの順位を取得

        setStage("next");
      })
      .catch((error) => {
        console.error("ランキングの取得中にエラーが発生しました:", error);
      });
  };

  // 名前登録用のハンドラ
  const handleNameClick = () => {
    setIsRegisteringName(true);
  };

  const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setNewName(event.target.value);
  };

  const handleNameSubmit = () => {
    // 名前を登録
    fetch("/api/register", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name: newName }),
    })
      .then((response) => response.json())
      .catch((error) => {
        console.error("名前の登録中にエラーが発生しました:", error);
      });
  };

  // トップへ戻るボタンのハンドラ
  const handleBackToTop = () => {
    // 状態を初期化
    setStage("start");
    setInputPrompt("");
    setInputNegativePrompt("");
    setLeftAestheticScore(null);
    setRightAestheticScore(null);
    setUserBestImage(null);
    setUserBestScore(null);
    setAiBestImage(null);
    setAiBestScore(null);
    setAiPrompt("");
    setAiNegativePrompt("");
    setAiComment("");
    setRankings([]);
    setYourRank(null);
    setIsRegisteringName(false);
    setNewName("");
  };

  // 画像クリック時のハンドラ
  const handleRankingItemClick = (item: any) => {
    setSelectedRankingItem(item);
    setIsModalOpen(true);
  };

  // モーダルを閉じるハンドラ
  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedRankingItem(null);
  };

  // レンダリング部分
  if (stage === "start") {
    // スタート画面
    return (
      <Container>
        <ResultContainer>
          <Button variant="contained" color="primary" onClick={handleStart}>
            スタートする
          </Button>
        </ResultContainer>
      </Container>
    );
  } else if (stage === "countdown") {
    // カウントダウン画面
    return (
      <Container>
        <ResultContainer>
          <div style={{ fontSize: "3rem", fontWeight: "bold" }}>
            {countdown}
          </div>
        </ResultContainer>
      </Container>
    );
  } else if (stage === "battle") {
    // バトル画面
    return (
      <Container>
        {/* タイマー表示 */}
        <TimerContainer>
          <Timer>残り時間: {battleTime}秒</Timer>
        </TimerContainer>

        {/* バトルコンテナ */}
        <BattleContainer>
          {/* 左側のコンテナ */}
          <Column>
            {/* 最高スコアと画像を右上に表示 */}
            {userBestScore !== null && userBestImage && (
              <div
                style={{
                  position: "absolute",
                  top: "10px",
                  right: "10px",
                  textAlign: "right",
                }}
              >
                <div style={{ fontSize: "0.8rem", color: "#555" }}>
                  最高スコア: {userBestScore.toFixed(2)}
                </div>
                <img
                  src={`data:image/webp;base64,${userBestImage}`}
                  alt="User Best"
                  style={{
                    width: "50px",
                    height: "50px",
                    borderRadius: "5px",
                    marginTop: "5px",
                  }}
                />
              </div>
            )}

            <Score>
              {leftAestheticScore !== null
                ? `審美性スコア: ${leftAestheticScore.toFixed(2)}`
                : "審美性スコア: --"}
            </Score>
            <StyledCanvasContainer>
              <StyledCanvas ref={handleLeftCanvasRef} />
            </StyledCanvasContainer>
            <StyledTextField
              variant="outlined"
              value={inputPrompt}
              onChange={handlePromptChange}
              multiline
              minRows={2}
              maxRows={4}
              placeholder="プロンプトを入力してください"
            />
            <StyledTextField
              variant="outlined"
              value={inputNegativePrompt}
              onChange={handleNegativePromptChange}
              multiline
              minRows={2}
              maxRows={4}
              placeholder="ネガティブプロンプトを入力してください"
            />
            {/* 左側のコメント欄の代わりに空のスペースを配置 */}
            <CommentBox style={{ visibility: "hidden" }} />
          </Column>

          {/* 右側のコンテナ */}
          <Column>
            {/* 最高スコアと画像を右上に表示 */}
            {aiBestScore !== null && aiBestImage && (
              <div
                style={{
                  position: "absolute",
                  top: "10px",
                  right: "10px",
                  textAlign: "right",
                }}
              >
                <div style={{ fontSize: "0.8rem", color: "#555" }}>
                  最高スコア: {aiBestScore.toFixed(2)}
                </div>
                <img
                  src={`data:image/webp;base64,${aiBestImage}`}
                  alt="AI Best"
                  style={{
                    width: "50px",
                    height: "50px",
                    borderRadius: "5px",
                    marginTop: "5px",
                  }}
                />
              </div>
            )}

            <Score>
              {rightAestheticScore !== null
                ? `審美性スコア: ${rightAestheticScore.toFixed(2)}`
                : "審美性スコア: --"}
            </Score>
            <StyledCanvasContainer>
              <StyledCanvas ref={handleRightCanvasRef} />
            </StyledCanvasContainer>
            <StyledTextField
              variant="outlined"
              value={aiPrompt}
              multiline
              minRows={2}
              maxRows={4}
              disabled
              placeholder="AIが生成したプロンプト"
            />
            <StyledTextField
              variant="outlined"
              value={aiNegativePrompt}
              multiline
              minRows={2}
              maxRows={4}
              disabled
              placeholder="AIが生成したネガティブプロンプト"
            />
            <CommentBox>{aiComment || "コメントはまだありません。"}</CommentBox>
            {/* ローディングオーバーレイ */}
            <LoadingOverlay visible={isGenerating}>
              <LoadingText>
                言語モデルが思考しています
                <Dots>
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </Dots>
              </LoadingText>
            </LoadingOverlay>
          </Column>
        </BattleContainer>
      </Container>
    );
  } else if (stage === "result") {
    // 結果画面
    let resultText = "";
    if (winner === "user") {
      resultText = "あなたの勝ちです！";
    } else if (winner === "ai") {
      resultText = "AIの勝ちです！";
    } else {
      resultText = "引き分けです！";
    }

    return (
      <Container>
        <ResultContainer>
          {resultText}
          <Button
            variant="contained"
            color="primary"
            onClick={handleNext}
            style={{ marginTop: "20px" }}
          >
            次の画面へ
          </Button>
        </ResultContainer>
      </Container>
    );
  } else if (stage === "next") {
    // ランキング画面
    return (
      <Container>
        <RankingContainer>
          <h2>ランキング</h2>
          {rankings.map((item, index) => {
            const isYou = item.name === "あなた" || item.isCurrentUser;
            return (
              <RankingItem
                key={index}
                onClick={() => handleRankingItemClick(item)}
                isYou={isYou}
              >
                <RankingImage
                  src={`data:image/webp;base64,${item.imageBase64}`}
                  alt={item.name}
                />
                <RankingText>
                  <RankingName>{`${index + 1}位: ${item.name}`}</RankingName>
                  <RankingScore>
                    最高スコア: {item.aesthetic_score.toFixed(2)}
                  </RankingScore>
                </RankingText>
              </RankingItem>
            );
          })}

          {/* ユーザーの順位を表示 */}
          {yourRank && yourRank > rankings.length && (
            <RankingItem
              onClick={() => handleRankingItemClick(rankings[yourRank - 1])}
              isYou
            >
              <RankingImage
                src={`data:image/webp;base64,${
                  rankings[yourRank - 1].imageBase64
                }`}
                alt="あなた"
              />
              <RankingText>
                <RankingName>{`${yourRank}位: あなた`}</RankingName>
                <RankingScore>
                  最高スコア:{" "}
                  {rankings[yourRank - 1].aesthetic_score.toFixed(2)}
                </RankingScore>
              </RankingText>
            </RankingItem>
          )}

          {/* 名前登録ボタンを追加 */}
          {!isRegisteringName && (
            <div style={{ marginTop: "20px" }}>
              <Button
                variant="outlined"
                color="primary"
                onClick={handleNameClick}
              >
                名前を登録する
              </Button>
            </div>
          )}

          {/* 名前登録フォーム */}
          {isRegisteringName && (
            <div style={{ marginTop: "20px" }}>
              <TextField
                variant="outlined"
                value={newName}
                onChange={handleNameChange}
                placeholder="名前を入力してください"
              />
              <Button
                variant="contained"
                color="primary"
                onClick={handleNameSubmit}
                style={{ marginLeft: "10px" }}
              >
                名前を決定
              </Button>
            </div>
          )}

          {/* トップへ戻るボタン */}
          <div style={{ marginTop: "20px" }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleBackToTop}
            >
              トップへ戻る
            </Button>
          </div>
        </RankingContainer>

        {/* モーダル */}
        <Dialog
          open={isModalOpen}
          onClose={handleModalClose}
          maxWidth="lg"
          fullWidth
        >
          {selectedRankingItem && (
            <>
              <DialogTitle>{selectedRankingItem.name}の作品</DialogTitle>
              <DialogContent>
                <ModalImage
                  src={`data:image/webp;base64,${selectedRankingItem.imageBase64}`}
                  alt={selectedRankingItem.name}
                />
                <div style={{ marginBottom: "10px" }}>
                  <strong>プロンプト:</strong>
                  <div>{selectedRankingItem.prompt || "情報がありません"}</div>
                </div>
                <div style={{ marginBottom: "10px" }}>
                  <strong>ネガティブプロンプト:</strong>
                  <div>
                    {selectedRankingItem.negative_prompt || "情報がありません"}
                  </div>
                </div>
              </DialogContent>
            </>
          )}
        </Dialog>
      </Container>
    );
  } else {
    // エラー時
    return null;
  }
}

export default App;
