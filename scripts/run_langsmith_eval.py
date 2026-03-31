"""Run LangSmith offline evaluation against the local /chat API."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import httpx
from langsmith import Client
from langsmith.utils import LangSmithAuthError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402

DEFAULT_DATASET_NAME = "eval_case"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _pick_first(mapping: Optional[Dict[str, Any]], keys: Sequence[str], default: Any = "") -> Any:
    payload = mapping or {}
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return default


def _stringify_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_base_url(base_url: str) -> str:
    return str(base_url or DEFAULT_BASE_URL).rstrip("/")


def _resolve_question(inputs: Dict[str, Any], input_key: Optional[str]) -> str:
    if input_key:
        question = str(inputs.get(input_key, "")).strip()
        if question:
            return question
    return str(
        _pick_first(
            inputs,
            keys=("question", "input", "query", "content", "message"),
            default="",
        )
    ).strip()


def _resolve_reference_source(reference_outputs: Optional[Dict[str, Any]], source_key: Optional[str]) -> str:
    if source_key:
        value = str((reference_outputs or {}).get(source_key, "")).strip()
        if value:
            return value
    return str(
        _pick_first(
            reference_outputs,
            keys=("reference_source", "source", "expected_source", "citation"),
            default="",
        )
    ).strip()


def _build_client(
    *,
    api_key: Optional[str],
    endpoint: Optional[str],
    workspace_id: Optional[str],
) -> Client:
    return Client(
        api_key=(api_key or settings.langsmith_api_key or "").strip() or None,
        api_url=(endpoint or settings.langsmith_endpoint or "").strip() or None,
        workspace_id=(workspace_id or settings.langsmith_workspace_id or "").strip() or None,
        otel_enabled=False,
    )


def _ensure_service_available(base_url: str) -> None:
    health_url = f"{_normalize_base_url(base_url)}/health"
    try:
        response = httpx.get(health_url, timeout=httpx.Timeout(10.0, connect=5.0))
        response.raise_for_status()
    except Exception as error:
        raise RuntimeError(
            f"无法连接到本地服务：{health_url}\n"
            "请先启动 FastAPI 服务，再执行评估脚本。"
        ) from error


def _validate_dataset(client: Client, dataset_name: Optional[str], dataset_id: Optional[str]) -> Any:
    try:
        if dataset_id:
            return client.read_dataset(dataset_id=dataset_id)
        return client.read_dataset(dataset_name=dataset_name or DEFAULT_DATASET_NAME)
    except LangSmithAuthError as error:
        raise RuntimeError(
            "LangSmith 鉴权失败：请检查 LANGSMITH_API_KEY 是否有效，"
            "以及 LANGSMITH_WORKSPACE_ID 是否与当前 UI 中的数据集所在 workspace 一致。"
        ) from error


def _build_target(
    *,
    base_url: str,
    user_prefix: str,
    input_key: Optional[str],
    debug: bool,
) -> Any:
    api_url = f"{_normalize_base_url(base_url)}/chat"

    def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = _resolve_question(inputs, input_key=input_key)
        if not question:
            raise ValueError(f"未能从 dataset example 中解析问题字段，inputs={inputs}")

        response = httpx.post(
            api_url,
            json={
                "user_id": f"{user_prefix}-{uuid.uuid4().hex[:10]}",
                "content": question,
                "debug": debug,
            },
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result") or {}
        debug_payload = payload.get("debug") or {}
        langsmith_payload = debug_payload.get("langsmith") or {}

        return {
            "output": payload.get("message", ""),
            "answer": payload.get("message", ""),
            "citations": _stringify_list(result.get("citations")),
            "sources": _stringify_list(result.get("sources")),
            "run_status": payload.get("run_status", ""),
            "active_agent": payload.get("active_agent", ""),
            "intent": payload.get("intent", ""),
            "thread_id": payload.get("thread_id", ""),
            "trace_url": langsmith_payload.get("run_url", ""),
        }

    return target


def _build_source_match_evaluator(reference_source_key: Optional[str]) -> Any:
    def source_match(inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> Dict[str, Any]:
        expected_source = _resolve_reference_source(reference_outputs, source_key=reference_source_key)
        observed_sources = _stringify_list(outputs.get("citations")) + _stringify_list(outputs.get("sources"))
        observed_text = " | ".join(observed_sources)
        if not expected_source:
            return {
                "key": "source_match",
                "score": None,
                "comment": "reference output 中没有可用的 source 字段，已跳过。",
            }
        score = 1.0 if expected_source in observed_text else 0.0
        return {
            "key": "source_match",
            "score": score,
            "comment": (
                f"expected={expected_source} | observed={observed_text or 'EMPTY'}"
            ),
        }

    return source_match


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对当前 CustomerSupport-Agent 的 /chat 接口运行 LangSmith 离线评估。",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="LangSmith dataset 名称。")
    parser.add_argument("--dataset-id", default="", help="可选：直接指定 dataset ID。")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="本地 FastAPI 服务地址。")
    parser.add_argument("--experiment-prefix", default="customer-support-api", help="Experiment 前缀。")
    parser.add_argument("--description", default="Offline eval against /chat endpoint", help="Experiment 描述。")
    parser.add_argument("--max-concurrency", type=int, default=2, help="并发度，建议 1-3。")
    parser.add_argument("--num-repetitions", type=int, default=1, help="每条样本重复次数。")
    parser.add_argument("--user-prefix", default="eval", help="评测请求生成 user_id 时的前缀。")
    parser.add_argument("--input-key", default="", help="可选：显式指定 dataset 输入字段名。")
    parser.add_argument("--reference-source-key", default="reference_source", help="本地 source_match 使用的参考来源字段名。")
    parser.add_argument("--langsmith-api-key", default="", help="可选：覆盖环境变量中的 LANGSMITH_API_KEY。")
    parser.add_argument("--langsmith-endpoint", default="", help="可选：覆盖环境变量中的 LANGSMITH_ENDPOINT。")
    parser.add_argument("--langsmith-workspace-id", default="", help="可选：覆盖环境变量中的 LANGSMITH_WORKSPACE_ID。")
    parser.add_argument("--no-local-source-evaluator", action="store_true", help="不附带本地 source_match evaluator。")
    parser.add_argument("--no-debug", action="store_true", help="调用 /chat 时不请求 debug 信息。")
    parser.add_argument("--no-upload", action="store_true", help="仅本地执行 evaluate，不上传结果到 LangSmith。")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    base_url = _normalize_base_url(args.base_url)
    debug = not args.no_debug

    _ensure_service_available(base_url)

    client = _build_client(
        api_key=args.langsmith_api_key,
        endpoint=args.langsmith_endpoint,
        workspace_id=args.langsmith_workspace_id,
    )
    dataset = _validate_dataset(
        client,
        dataset_name=args.dataset_name,
        dataset_id=args.dataset_id or None,
    )

    evaluators = []
    if not args.no_local_source_evaluator:
        evaluators.append(_build_source_match_evaluator(args.reference_source_key or None))

    target = _build_target(
        base_url=base_url,
        user_prefix=args.user_prefix,
        input_key=args.input_key or None,
        debug=debug,
    )

    print("开始运行 LangSmith 评估...")
    print(f"- dataset = {dataset.name}")
    print(f"- base_url = {base_url}")
    print(f"- experiment_prefix = {args.experiment_prefix}")
    print(f"- max_concurrency = {args.max_concurrency}")
    print(f"- local_source_evaluator = {not args.no_local_source_evaluator}")
    print("")

    try:
        results = client.evaluate(
            target,
            data=dataset,
            evaluators=evaluators or None,
            experiment_prefix=args.experiment_prefix,
            description=args.description,
            max_concurrency=args.max_concurrency,
            num_repetitions=args.num_repetitions,
            upload_results=not args.no_upload,
            blocking=True,
            metadata={
                "base_url": base_url,
                "dataset_name": dataset.name,
                "local_source_evaluator": not args.no_local_source_evaluator,
            },
        )
    except LangSmithAuthError as error:
        raise RuntimeError(
            "LangSmith 鉴权失败：请检查本地 .env 中的 LANGSMITH_API_KEY / LANGSMITH_WORKSPACE_ID。"
        ) from error

    print("评估完成。")
    print(f"- experiment_name = {results.experiment_name}")
    if args.no_upload:
        print("- upload_results = false（本次未上传到 LangSmith UI）")
    else:
        print("- 现在可以去 LangSmith -> Datasets & Experiments -> 该 dataset -> Experiments 查看结果")
        print("- 你在 UI 里绑定的 Correctness evaluator 会自动出现在这次 experiment 中")
        if not args.no_local_source_evaluator:
            print("- 这次脚本还额外上传了一个本地 evaluator：source_match")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[run_langsmith_eval] {error}", file=sys.stderr)
        raise SystemExit(1)
