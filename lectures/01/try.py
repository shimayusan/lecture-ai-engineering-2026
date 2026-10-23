"""第1回の宿題を試す実験スクリプト。

実行例:
    uv run python -m lectures.01.try
    uv run python -m lectures.01.try --model gemini-3.5-flash

基礎（temperature比較）、実装（セッション内での会話履歴）、
発展（複数モデル・質問の応答時間・トークン数比較）を順に実行する。
"""

import argparse

from google.genai import errors

from app.ask import SessionManager, ask
from app.config import MODEL, api_error_message


def run_basic_experiment(model: str) -> None:
    question = "LoRAを一文で説明してください。"
    print("\n=== 基礎: temperatureによる回答の違い ===")
    for temperature in (0.0, 0.7, 1.5):
        print(f"\n[temperature] {temperature}")
        print(ask(question, model=model, temperature=temperature))


def run_session_experiment(model: str) -> None:
    print("\n=== 実装: セッション内で会話履歴を保持 ===")
    session = SessionManager(model=model, temperature=0.2)
    session_id = session.start_session()
    print(f"[session] {session_id}")
    try:
        first_answer = session.send_message("私の名前はハナです。覚えておいてください。")
        print(f"[回答1] {first_answer}")
        second_answer = session.send_message("私の名前は何ですか？")
        print(f"[回答2] {second_answer}")
    finally:
        session.end_session()
        print(f"[session] {session_id} を終了し、履歴を破棄しました")


def run_advanced_experiment(models: list[str]) -> None:
    print("\n=== 発展: モデルごとの応答時間・トークン数 ===")
    for question in (
        "LoRAを一文で説明してください。",
        "RAGを一文で説明してください。",
        "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    ):
        print(f"\n[質問] {question}")
        for model in models:
            print(f"\n[model] {model}")
            print(ask(question, model=model, temperature=0.2))


def main() -> None:
    parser = argparse.ArgumentParser(description="第1回の宿題の実験を実行する")
    parser.add_argument("--model", default=MODEL, help="使用するモデル")
    parser.add_argument(
        "--compare-model",
        action="append",
        dest="compare_models",
        help="レベル3で比較するモデル（複数指定可）",
    )
    args = parser.parse_args()

    print(f"[model] {args.model}")
    try:
        run_basic_experiment(args.model)
        run_session_experiment(args.model)
        models = args.compare_models or [args.model, "gemini-3.8-flash"]
        run_advanced_experiment(models)
    except errors.APIError as error:
        print(api_error_message(error.code, error.message))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()