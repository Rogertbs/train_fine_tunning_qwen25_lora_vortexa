"""Avalia um modelo servido por uma API compativel com OpenAI."""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REGISTRY_PATTERN = re.compile(r"VX-\d+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.lower().split())


def ask_model(url: str, model: str, messages: list[dict], max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Falha ao consultar o vLLM: {error}") from error

    return result["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um adapter LoRA servido pelo vLLM.")
    parser.add_argument("--url", default="http://localhost:8085/v1/chat/completions")
    parser.add_argument("--model", default="vortexa")
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry_hits = 0
    answer_hits = 0

    for index, record in enumerate(records, start=1):
        messages = record["messages"]
        expected = messages[-1]["content"]
        response = ask_model(args.url, args.model, messages[:-1], args.max_tokens)
        expected_registry = REGISTRY_PATTERN.search(expected)
        registry_ok = bool(expected_registry and expected_registry.group() in response)
        answer_ok = normalize(expected) in normalize(response)
        registry_hits += registry_ok
        answer_hits += answer_ok

        if args.verbose:
            status = "OK" if registry_ok else "ERRO"
            print(f"[{status}] {index}: {response}")

    total = len(records)
    print(f"Modelo: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Registros corretos: {registry_hits}/{total} ({registry_hits / total:.1%})")
    print(f"Respostas exatas: {answer_hits}/{total} ({answer_hits / total:.1%})")


if __name__ == "__main__":
    main()
