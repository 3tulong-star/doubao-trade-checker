# Doubao Trade Checker

Batch-check whether companies appear to do foreign trade using **Doubao + Volcengine Ark Responses API + Web Search tool**.

## Official Basis

This project follows Volcengine Ark's documented approach for web search:

- Responses API endpoint: `https://ark.cn-beijing.volces.com/api/v3/responses`
- Web Search tool via:
  - `"tools": [{"type": "web_search"}]`

Reference:

- Web Search tool docs: https://www.volcengine.com/docs/82379/1756990?lang=zh
- Plugin/tool usage example snippet surfaced in Volcengine docs: https://www.volcengine.com/docs/82379/1338552

## What It Does

- Reads up to 1000 company names from a text file
- Calls Doubao with Web Search enabled for each company
- Asks the model to classify whether the company appears to do foreign trade
- Saves:
  - `results.csv`
  - `results.jsonl`
  - `checkpoint.json`
  - `errors.jsonl`

## Output Fields

- `company_name`
- `foreign_trade`
  - `yes`
  - `no`
  - `uncertain`
- `total_score`
- `dimension_scores`
- `evidence_by_dimension`
- `reasoning`
- `evidence`
- `status`
- `raw_response`

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Environment Variables

```bash
export ARK_API_KEY="your_ark_api_key"
```

Optional:

```bash
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export ARK_MODEL="doubao-seed-2-0-lite-260215"
```

## Input File

One company name per line:

```text
SHENZHEN EXAMPLE TECHNOLOGY CO., LTD.
MEXICO IMPORTACIONES SA DE CV
NINGBO ABC TRADING CO LTD
```

## Run

```bash
python3 checker.py \
  --input companies.example.txt \
  --output-dir runs/run1
```

For concurrent runs:

```bash
python3 checker.py \
  --input companies.example.txt \
  --output-dir runs/run1 \
  --concurrency 3
```

## Field Discovery

`field_discovery.py` extracts structured foreign-trade-related fields from public web search results.

```bash
python3 field_discovery.py \
  --input companies.example.txt \
  --output-dir runs/field_discovery
```

## Resume

Rerun with the same `--output-dir`. The script uses `checkpoint.json` to skip already completed company names.

## Notes

- This script depends on model judgment after web search, so output should be reviewed rather than blindly trusted.
- For very large runs, use a conservative `--sleep-seconds` to reduce rate-limit pressure.
- Do not commit API keys, private company lists, or generated run outputs. Local outputs under `runs/` are ignored by git.
