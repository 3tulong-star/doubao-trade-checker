import argparse
import csv
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Set

import requests


DEFAULT_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
DEFAULT_MODEL = os.getenv("ARK_MODEL", "doubao-seed-2-0-lite-260215").strip()
DEFAULT_API_KEY = os.getenv("ARK_API_KEY", "").strip()


FIELD_NAMES = [
    "official_website",
    "company_type",
    "business_scope_keywords",
    "main_products",
    "export_markets",
    "b2b_platforms",
    "customs_or_export_record",
    "import_export_license",
    "foreign_trade_job_roles",
    "international_certifications",
    "contact_emails",
    "contact_phones",
    "contact_person_names",
    "factory_or_trader_signal",
    "addresses",
    "brand_names",
    "social_or_content_channels",
]


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
    return normalize_lines(input_path.read_text(encoding="utf-8").splitlines())


def build_prompt(company_name: str) -> str:
    return f"""
你需要结合联网搜索结果，分析下面这个企业在公开互联网信息里，哪些“外贸相关字段”可以被稳定结构化。

企业抬头：
{company_name}

任务目标：
1. 不要先做 yes/no 判断。
2. 重点是从公开网页里提取“可以结构化入库的独立字段”。
3. 只保留你能从联网搜索结果中找到明确证据支持的字段。
4. 没搜到就返回空值，不要臆造。

请按以下字段返回：
- official_website: 官网 URL，找不到返回 ""
- company_type: 从公开信息推断，枚举 ["manufacturer","trading_company","service_provider","retailer","unknown"] 之一
- business_scope_keywords: 工商/简介里能抽出的主营关键词数组
- main_products: 主要产品数组
- export_markets: 明确出现的出口国家/地区/海外市场数组
- b2b_platforms: 数组，每项包含 platform、url、store_name
- customs_or_export_record: 是否能看到海关、进出口收发货人、出口记录等证据，布尔值
- import_export_license: 是否能看到经营范围含进出口/外贸备案/进出口权等证据，布尔值
- foreign_trade_job_roles: 招聘中出现的外贸相关岗位数组
- international_certifications: 国际认证数组，例如 CE/FDA/ROHS/REACH/UL/ISO
- contact_emails: 邮箱数组
- contact_phones: 电话数组
- contact_person_names: 联系人姓名数组
- factory_or_trader_signal: 枚举 ["factory","trader","both","unknown"]
- addresses: 公开地址数组
- brand_names: 品牌名数组
- social_or_content_channels: 数组，可包含微信公众号、视频号、LinkedIn、Facebook、独立站博客等

同时返回：
- source_types: 这家公司本次搜索里实际命中的信息源类型数组，例如 ["official_site","b2b_platform","job_posting","customs","yellow_pages","news","social"]
- evidence_snippets: 最多 8 条中文证据摘要
- notes: 对字段可结构化性的简短说明

只返回一个 JSON 对象，不要返回 markdown，不要返回解释性前缀：
{{
  "company_name": "{company_name}",
  "fields": {{
    "official_website": "",
    "company_type": "unknown",
    "business_scope_keywords": [],
    "main_products": [],
    "export_markets": [],
    "b2b_platforms": [],
    "customs_or_export_record": false,
    "import_export_license": false,
    "foreign_trade_job_roles": [],
    "international_certifications": [],
    "contact_emails": [],
    "contact_phones": [],
    "contact_person_names": [],
    "factory_or_trader_signal": "unknown",
    "addresses": [],
    "brand_names": [],
    "social_or_content_channels": []
  }},
  "source_types": [],
  "evidence_snippets": [],
  "notes": ""
}}
""".strip()


def extract_json(text: str) -> Dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")
    return json.loads(text[start:end + 1])


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


def call_ark(company_name: str, *, api_key: str, base_url: str, model: str, timeout_seconds: int) -> Dict:
    request_payload = {
        "model": model,
        "stream": True,
        "tools": [{"type": "web_search", "max_keyword": 3}],
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": build_prompt(company_name)}],
            }
        ],
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
        raise RuntimeError(f"Ark API HTTP {response.status_code}: {response.text}")
    raw_lines = [line.decode("utf-8", errors="replace") for line in response.iter_lines() if line]
    raw_text = extract_output_text_from_sse_lines(raw_lines)
    payload = extract_json(raw_text)
    payload["_raw_response"] = raw_text
    return payload


def run_one_company(company_name: str, args: argparse.Namespace) -> Dict:
    result = call_ark(
        company_name,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    result.setdefault("company_name", company_name)
    return result


def append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: Dict, *, write_header: bool) -> None:
    fieldnames = [
        "company_name",
        "official_website",
        "company_type",
        "factory_or_trader_signal",
        "customs_or_export_record",
        "import_export_license",
        "business_scope_keywords",
        "main_products",
        "export_markets",
        "b2b_platforms",
        "foreign_trade_job_roles",
        "international_certifications",
        "contact_emails",
        "contact_phones",
        "contact_person_names",
        "addresses",
        "brand_names",
        "social_or_content_channels",
        "source_types",
        "evidence_snippets",
        "notes",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def summarize(records: List[Dict]) -> Dict:
    field_presence = {name: 0 for name in FIELD_NAMES}
    source_presence: Dict[str, int] = {}
    enum_counts = {
        "company_type": {},
        "factory_or_trader_signal": {},
    }
    for record in records:
        fields = record.get("fields") or {}
        for field_name in FIELD_NAMES:
            value = fields.get(field_name)
            present = False
            if isinstance(value, bool):
                present = value
            elif isinstance(value, str):
                present = bool(value.strip())
            elif isinstance(value, list):
                present = len(value) > 0
            elif value is not None:
                present = True
            if present:
                field_presence[field_name] += 1
        for key in enum_counts:
            value = str(fields.get(key, "unknown")).strip() or "unknown"
            enum_counts[key][value] = enum_counts[key].get(value, 0) + 1
        for source in record.get("source_types") or []:
            source = str(source).strip()
            if not source:
                continue
            source_presence[source] = source_presence.get(source, 0) + 1
    total = max(len(records), 1)
    fill_rates = {
        field_name: {
            "count": count,
            "fill_rate": round(count / total, 4),
        }
        for field_name, count in field_presence.items()
    }
    return {
        "sample_size": len(records),
        "field_fill_rates": fill_rates,
        "source_type_counts": source_presence,
        "enum_counts": enum_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover which foreign-trade-related fields can be structured from web search results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set ARK_API_KEY.")
    companies = load_companies(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "field_results.jsonl"
    results_csv = output_dir / "field_results.csv"
    errors_jsonl = output_dir / "errors.jsonl"
    summary_json = output_dir / "field_summary.json"

    all_records: List[Dict] = []
    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    submitted = 0
    finished = 0
    company_iter = iter(companies)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        in_flight: Dict[Future, str] = {}
        while len(in_flight) < args.concurrency:
            try:
                company_name = next(company_iter)
            except StopIteration:
                break
            future = executor.submit(run_one_company, company_name, args)
            in_flight[future] = company_name
            submitted += 1
            print(f"[submit {submitted}/{len(companies)}] {company_name}")
            time.sleep(args.sleep_seconds)

        while in_flight:
            done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                company_name = in_flight.pop(future)
                finished += 1
                try:
                    payload = future.result()
                    fields = payload.get("fields") or {}
                    row = {
                        "company_name": payload.get("company_name", company_name),
                        "official_website": fields.get("official_website", ""),
                        "company_type": fields.get("company_type", ""),
                        "factory_or_trader_signal": fields.get("factory_or_trader_signal", ""),
                        "customs_or_export_record": fields.get("customs_or_export_record", False),
                        "import_export_license": fields.get("import_export_license", False),
                        "business_scope_keywords": " | ".join(fields.get("business_scope_keywords") or []),
                        "main_products": " | ".join(fields.get("main_products") or []),
                        "export_markets": " | ".join(fields.get("export_markets") or []),
                        "b2b_platforms": json.dumps(fields.get("b2b_platforms") or [], ensure_ascii=False),
                        "foreign_trade_job_roles": " | ".join(fields.get("foreign_trade_job_roles") or []),
                        "international_certifications": " | ".join(fields.get("international_certifications") or []),
                        "contact_emails": " | ".join(fields.get("contact_emails") or []),
                        "contact_phones": " | ".join(fields.get("contact_phones") or []),
                        "contact_person_names": " | ".join(fields.get("contact_person_names") or []),
                        "addresses": " | ".join(fields.get("addresses") or []),
                        "brand_names": " | ".join(fields.get("brand_names") or []),
                        "social_or_content_channels": " | ".join(fields.get("social_or_content_channels") or []),
                        "source_types": " | ".join(payload.get("source_types") or []),
                        "evidence_snippets": " | ".join(payload.get("evidence_snippets") or []),
                        "notes": payload.get("notes", ""),
                    }
                    append_jsonl(results_jsonl, payload)
                    append_csv(results_csv, row, write_header=write_header)
                    write_header = False
                    all_records.append(payload)
                    print(f"[done {finished}/{len(companies)}] {company_name}")
                except Exception as exc:
                    append_jsonl(errors_jsonl, {"company_name": company_name, "error": str(exc)})
                    print(f"[done {finished}/{len(companies)}] {company_name}")
                    print(f"  -> error: {exc}")

                try:
                    next_company = next(company_iter)
                except StopIteration:
                    continue
                next_future = executor.submit(run_one_company, next_company, args)
                in_flight[next_future] = next_company
                submitted += 1
                print(f"[submit {submitted}/{len(companies)}] {next_company}")
                time.sleep(args.sleep_seconds)

    summary_json.write_text(json.dumps(summarize(all_records), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
