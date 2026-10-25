"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --session  # セッション中は会話履歴を保持する
    uv run python -m app.ask --temperature 0.7 "LoRAとは何ですか？"

通常モードはツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from time import perf_counter
from uuid import uuid4

from google import genai
from google.genai import chats
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _print_response_metrics(
    response: genai.types.GenerateContentResponse, elapsed: float
) -> None:
    usage = response.usage_metadata
    print(f"[応答時間] {elapsed:.2f}秒")
    if usage is None:
        print("[トークン数] 利用量情報なし")
    else:
        print(
            f"[トークン数] 入力: {usage.prompt_token_count}, "
            f"出力: {usage.candidates_token_count}"
        )


def ask(question: str, model: str = MODEL, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    config = {"temperature": temperature} if temperature is not None else None
    started_at = perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=question,
        config=config,
    )
    _print_response_metrics(response, perf_counter() - started_at)

    return response.text


class SessionManager:
    """プロセス内だけで有効な会話セッションを管理する。"""

    def __init__(self, model: str = MODEL, temperature: float | None = None) -> None:
        self._client = genai.Client(api_key=get_api_key())
        self._model = model
        self._config = {"temperature": temperature} if temperature is not None else None
        self._conversation: chats.Chat | None = None
        self.session_id: str | None = None

    def start_session(self) -> str:
        if self._conversation is not None:
            raise RuntimeError("セッションはすでに開始されています")
        self._conversation = self._client.chats.create(
            model=self._model, config=self._config
        )
        self.session_id = uuid4().hex[:8]
        return self.session_id

    def send_message(self, message: str) -> str | None:
        if self._conversation is None:
            raise RuntimeError("メッセージを送るにはセッションを開始してください")
        started_at = perf_counter()
        response = self._conversation.send_message(message)
        _print_response_metrics(response, perf_counter() - started_at)
        return response.text

    def end_session(self) -> None:
        self._conversation = None
        self.session_id = None


def run_session(model: str = MODEL, temperature: float | None = None) -> None:
    session = SessionManager(model=model, temperature=temperature)
    session_id = session.start_session()
    print(
        f"セッション {session_id} を開始しました。"
        "終了するには exit または quit と入力してください。"
    )
    try:
        while True:
            message = input("あなた> ").strip()
            if message.lower() in {"exit", "quit"}:
                break
            if not message:
                continue
            print(f"モデル> {session.send_message(message)}\n")
    finally:
        session.end_session()
        print(f"セッション {session_id} を終了しました（履歴は破棄されます）。")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--model", default=MODEL, help="使用するモデル（既定: AIE_MODEL または講座の既定値）"
    )
    parser.add_argument(
        "--temperature", type=float, help="生成時の temperature（0〜2）"
    )
    parser.add_argument(
        "--session", action="store_true", help="会話履歴を保つ一時セッションを開始"
    )
    args = parser.parse_args()
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature は 0 から 2 の範囲で指定してください")

    print(f"[model] {args.model}")
    try:
        if args.session:
            run_session(model=args.model, temperature=args.temperature)
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question, model=args.model, temperature=args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
