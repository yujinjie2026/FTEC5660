#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": (
                            "You are a pure OCR data extractor. DO NOT calculate anything. DO NOT output JSON.\n\n"
                            
                            "Step 1: Find the final net payment amount after rounding.\n"
                            "Step 2: Extract EVERY original positive item price into a Markdown table.\n"
                            "- List every item row by row. If an identical item appears 3 times, write 3 separate rows.\n"
                            "- Do NOT include negative discounts, rounding, subtotals, cash, or change in the table.\n\n"
                            
                            "STRICT OUTPUT FORMAT:\n"
                            "<thinking_process>\n"
                            "Scan the receipt line by line to map names to their exact prices.\n"
                            "</thinking_process>\n"
                            "FINAL_PAYMENT: <amount>\n\n"
                            "| Item Name | Price |\n"
                            "|---|---|\n"
                            "| Exact Name 1 | 24.90 |\n"
                            "| Exact Name 2 | 12.00 |"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                ],
            )
        ]
    )

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0.0,
    )
    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    batch_inputs = [{"image_url": image_data_url(img_path)} for img_path in images]
    results = chain.batch(batch_inputs,config={"max_concurrency": 2})
    
    total_spend = Decimal("0.00")
    total_without_discount = Decimal("0.00")
    
    print("\n--- DEBUG LOG: Model Step-by-Step Outputs ---")
    for img_path, res in zip(images, results):
        text = response_text(res)
        print(f"\n[{img_path.name} RAW OUTPUT]:\n{text}\n")
        
        fp = Decimal("0.00")
        op = Decimal("0.00")
        
        # 1. 提取 Final Payment (Q1)
        fp_match = re.search(r"FINAL_PAYMENT:\s*?\$?\s*?(-?\d+\.\d+)", text, re.IGNORECASE)
        if fp_match:
            fp = Decimal(fp_match.group(1))
            
        # 2. 提取 Markdown 表格中的正数 (Q2)
        for line in text.splitlines():
            # 识别 Markdown 表格行，排除表头和分割线
            if line.strip().startswith("|") and "Price" not in line and "---" not in line:
                # 分割表格列并清理空格
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    price_str = parts[-1].replace(",", "")
                    # 抓取包含可能的负号的数字
                    amounts = re.findall(r"-?\d+\.\d+", price_str)
                    if amounts:
                        val = Decimal(amounts[-1])
                        # 只有大于0的才累加，自带免疫负号折扣幻觉的能力
                        if val > 0:
                            op += val
                            
        # 备选提取逻辑：极端防御
        if fp == Decimal("0.00") and op == Decimal("0.00"):
            decimals = re.findall(r"-?\d+\.\d{2}", text)
            if len(decimals) >= 2:
                fp = Decimal(decimals[-2])
                op = Decimal(decimals[-1])
            elif len(decimals) == 1:
                fp = Decimal(decimals[-1])
                op = Decimal(decimals[-1])
                
        print(f"[{img_path.name} PARSED] -> Q1(Spent): {fp}, Q2(Without Discount): {op}")
        total_spend += fp
        total_without_discount += op

    print("--- DEBUG LOG END ---\n")

    return {
        QUERY_1: f"HK${total_spend:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
