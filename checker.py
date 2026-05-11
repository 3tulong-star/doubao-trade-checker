import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests


DEFAULT_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
DEFAULT_MODEL = os.getenv("ARK_MODEL", "doubao-seed-2-0-lite-260215").strip()
DEFAULT_API_KEY = os.getenv("ARK_API_KEY", "").strip()


@dataclass
class CompanyResult:
    company_name: str
    foreign_trade: str
    total_score: int
    reasoning: str
    dimension_scores: Dict[str, int]
    evidence_by_dimension: Dict[str, List[str]]
    evidence: List[str]
    status: str
    raw_response: str


def normalize_lines(lines: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    items: List[str] = []
    for raw in lines:
        item = raw.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        items.append(item)
    return items


def load_companies(input_path: Path) -> List[str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return normalize_lines(input_path.read_text(encoding="utf-8").splitlines())


def build_prompt(company_name: str) -> str:
    return f"""
你需要结合联网搜索结果，判断下面这个企业是否“做外贸”。

企业抬头：
{company_name}

请按以下维度分别判断并打分。每个维度返回 0-20 的整数分，证据越强分越高，无证据返回 0：

1. customs_trade
是否有海关、进出口收发货人、报关备案、进出口经营权等证据

2. overseas_sales
是否有官网、产品介绍、新闻、案例中明确写出口、海外市场、国际客户、销往海外等证据

3. b2b_platform
是否有 1688 外贸订单、阿里国际站、中国制造网、环球资源、跨境平台等证据

4. hiring_signal
是否有外贸业务员、国际销售、海外销售、跨境运营等招聘或团队信息

5. international_compliance
是否有 CE、FDA、海外认证、国际物流、海外仓、国际参展等辅助证据

如果证据不足，不要瞎猜，输出 uncertain。

判断规则：
- total_score = 五个维度分数之和，总分 0-100
- 如果 total_score >= 60，foreign_trade 倾向 yes
- 如果 total_score <= 25，foreign_trade 倾向 no
- 中间区间或证据冲突时可输出 uncertain

只返回一个 JSON 对象，不要返回 markdown，不要返回解释性前缀：
{{
  "company_name": "{company_name}",
  "foreign_trade": "yes | no | uncertain",
  "total_score": 0,
  "reasoning": "简短中文说明",
  "dimension_scores": {{
    "customs_trade": 0,
    "overseas_sales": 0,
    "b2b_platform": 0,
    "hiring_signal": 0,
    "international_compliance": 0
  }},
  "evidence_by_dimension": {{
    "customs_trade": ["证据1"],
    "overseas_sales": ["证据2"],
    "b2b_platform": [],
    "hiring_signal": [],
    "international_compliance": []
  }},
  "evidence": ["关键证据1", "关键证据2"]
}}
""".strip()


def extract_output_text(response_json: Dict) -> str:
    output = response_json.get("output") or []
    chunks: List[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def extract_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")
    return json.loads(text[start:end + 1])


def _normalize_score(value: object, *, max_score: int) -> int:
    try:
        numeric = float(value)
    except Exception:
        return 0
    if 0 <= numeric <= 1:
        numeric = numeric * max_score
    return max(0, min(max_score, int(round(numeric))))


def validate_payload(company_name: str, payload: Dict, raw_response: str) -> CompanyResult:
    foreign_trade = str(payload.get("foreign_trade", "uncertain")).strip().lower()
    if foreign_trade not in {"yes", "no", "uncertain"}:
        foreign_trade = "uncertain"
    raw_dimension_scores = payload.get("dimension_scores")
    if not isinstance(raw_dimension_scores, dict):
        raw_dimension_scores = {}
    dimension_scores = {
        "customs_trade": _normalize_score(raw_dimension_scores.get("customs_trade", 0), max_score=20),
        "overseas_sales": _normalize_score(raw_dimension_scores.get("overseas_sales", 0), max_score=20),
        "b2b_platform": _normalize_score(raw_dimension_scores.get("b2b_platform", 0), max_score=20),
        "hiring_signal": _normalize_score(raw_dimension_scores.get("hiring_signal", 0), max_score=20),
        "international_compliance": _normalize_score(raw_dimension_scores.get("international_compliance", 0), max_score=20),
    }
    total_score = _normalize_score(payload.get("total_score", sum(dimension_scores.values())), max_score=100)
    computed_total = sum(dimension_scores.values())
    if abs(total_score - computed_total) > 10:
        total_score = computed_total
    raw_evidence_by_dimension = payload.get("evidence_by_dimension")
    if not isinstance(raw_evidence_by_dimension, dict):
        raw_evidence_by_dimension = {}
    evidence_by_dimension: Dict[str, List[str]] = {}
    for key in dimension_scores:
        values = raw_evidence_by_dimension.get(key)
        if not isinstance(values, list):
            values = []
        evidence_by_dimension[key] = [str(item).strip() for item in values if str(item).strip()]
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(item).strip() for item in evidence if str(item).strip()]
    reasoning = str(payload.get("reasoning", "")).strip()
    return CompanyResult(
        company_name=company_name,
        foreign_trade=foreign_trade,
        total_score=total_score,
        reasoning=reasoning,
        dimension_scores=dimension_scores,
        evidence_by_dimension=evidence_by_dimension,
        evidence=evidence,
        status="ok",
        raw_response=raw_response,
    )


def extract_output_text_from_sse_lines(lines: Iterable[str]) -> str:
    deltas: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        payload_str = line[6:]
        if payload_str == "[DONE]":
            continue
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "response.output_text.delta":
            delta = payload.get("delta", "")
            if delta:
                deltas.append(delta)
    return "".join(deltas).strip()


def call_ark(company_name: str, *, api_key: str, base_url: str, model: str, timeout_seconds: int) -> CompanyResult:
    request_payload = {
        "model": model,
        "stream": True,
        "tools": [{"type": "web_search", "max_keyword": 2}],
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_prompt(company_name),
                    }
                ],
            }
        ]
    }
    response = requests.post(
        f"{base_url}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=request_payload,
        timeout=timeout_seconds,
        stream=True,
    )
    if response.status_code >= 400:
        try:
            error_payload = response.json()
        except Exception:
            error_payload = {}
        error = error_payload.get("error") or {}
        error_code = error.get("code")
        error_message = error.get("message") or response.text
        if error_code == "ToolNotOpen":
            raise RuntimeError(
                "Ark web_search is not activated for this account. "
                "Activate it at https://console.volcengine.com/common-buy/CC_content_plugin "
                f"then rerun. Raw message: {error_message}"
            )
        raise RuntimeError(f"Ark API HTTP {response.status_code}: {error_message}")
    raw_lines = [line.decode("utf-8", errors="replace") for line in response.iter_lines() if line]
    raw_text = extract_output_text_from_sse_lines(raw_lines)
    payload = extract_json(raw_text)
    return validate_payload(company_name, payload, raw_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-check whether companies appear to do foreign trade using Doubao + Web Search.")
    parser.add_argument("--input", required=True, help="Input txt file, one company per line.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSONL/checkpoint output.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ark model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Ark base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Ark API key. Can also use ARK_API_KEY env var.")
    parser.add_argument("--sleep-seconds", type=float, default=1.5, help="Sleep between requests. Default: 1.5")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="HTTP timeout per request. Default: 120")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of companies to process.")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent requests. Default: 3")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> Set[str]:
    if not checkpoint_path.exists():
        return set()
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    completed = payload.get("completed_companies")
    if not isinstance(completed, list):
        return set()
    return {str(item) for item in completed}


def save_checkpoint(checkpoint_path: Path, completed_companies: Set[str]) -> None:
    checkpoint_path.write_text(
        json.dumps({"completed_companies": sorted(completed_companies)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, item: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: CompanyResult) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow([
                "company_name",
                "foreign_trade",
                "total_score",
                "customs_trade",
                "overseas_sales",
                "b2b_platform",
                "hiring_signal",
                "international_compliance",
                "reasoning",
                "evidence",
                "status",
                "raw_response",
            ])
        writer.writerow(
            [
                row.company_name,
                row.foreign_trade,
                row.total_score,
                row.dimension_scores.get("customs_trade", 0),
                row.dimension_scores.get("overseas_sales", 0),
                row.dimension_scores.get("b2b_platform", 0),
                row.dimension_scores.get("hiring_signal", 0),
                row.dimension_scores.get("international_compliance", 0),
                row.reasoning,
                " | ".join(row.evidence),
                row.status,
                row.raw_response,
            ]
        )


def run_one_company(company_name: str, args: argparse.Namespace) -> CompanyResult:
    return call_ark(
        company_name,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("Missing ARK API key. Set ARK_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "results.csv"
    results_jsonl = output_dir / "results.jsonl"
    errors_jsonl = output_dir / "errors.jsonl"
    checkpoint_json = output_dir / "checkpoint.json"

    companies = load_companies(Path(args.input))
    if args.limit > 0:
        companies = companies[: args.limit]

    completed = load_checkpoint(checkpoint_json)
    pending = [company for company in companies if company not in completed]
    total = len(companies)

    print(f"Loaded {total} companies, pending {len(pending)}, skipped {len(completed & set(companies))}.")
    if not pending:
        print(f"Done. Results written to {output_dir}")
        return 0

    max_workers = max(1, args.concurrency)
    total_pending = len(pending)
    submitted = 0
    finished = 0
    future_to_company: Dict[Future, str] = {}
    company_iter = iter(pending)

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        nonlocal submitted
        try:
            company_name = next(company_iter)
        except StopIteration:
            return False
        submitted += 1
        print(f"[submit {submitted}/{total_pending}] {company_name}")
        future = executor.submit(run_one_company, company_name, args)
        future_to_company[future] = company_name
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx in range(max_workers):
            if not submit_next(executor):
                break
            if args.sleep_seconds > 0 and idx < max_workers - 1:
                time.sleep(args.sleep_seconds)

        while future_to_company:
            done, _ = wait(set(future_to_company.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                company_name = future_to_company.pop(future)
                finished += 1
                print(f"[done {finished}/{total_pending}] {company_name}")
                try:
                    result = future.result()
                    append_csv(results_csv, result)
                    append_jsonl(results_jsonl, asdict(result))
                    completed.add(company_name)
                    save_checkpoint(checkpoint_json, completed)
                    print(f"  -> {result.foreign_trade} (score={result.total_score})")
                except Exception as exc:
                    error_payload = {
                        "company_name": company_name,
                        "status": "error",
                        "error": str(exc),
                    }
                    append_jsonl(errors_jsonl, error_payload)
                    print(f"  -> error: {exc}")
                if submit_next(executor) and args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    print(f"Done. Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
