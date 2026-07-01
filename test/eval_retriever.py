import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["IS_TESTING"] = "true"

from backend.scripts.retriever import retriever_logic


def load_eval_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_file_name(file_name: str) -> str:
    return file_name.replace("\\", "/").split("/")[-1]


def hit_at_k(retrieved_files, expected_files, k: int) -> bool:
    return bool(set(retrieved_files[:k]) & set(expected_files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-file",
        default=PROJECT_ROOT / "test" / "eval_questions.json",
    )
    parser.add_argument("--match-count", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_eval_cases(Path(args.eval_file))
    if args.limit:
        cases = cases[: args.limit]

    total = len(cases)
    recall_1 = 0
    recall_k = 0
    failures = []

    for i, case in enumerate(cases, 1):
        q = case["question"]
        expected = [normalize_file_name(x) for x in case["expected_files"]]

        docs = retriever_logic(
            q,
            match_count=args.match_count,
            match_threshold=args.threshold,
            include_metadata=True,
        )

        retrieved = [normalize_file_name(d["file_name"]) for d in docs]

        r1 = hit_at_k(retrieved, expected, 1)
        rk = hit_at_k(retrieved, expected, args.match_count)

        recall_1 += int(r1)
        recall_k += int(rk)

        status = "PASS" if rk else "FAIL"

        print(f"\n[{i}/{total}] {status} {q}")
        print(f"expected: {', '.join(expected)}")
        print(f"retrieved: {', '.join(retrieved) or '(none)'}")

        if not rk:
            failures.append(
                {
                    "question": q,
                    "expected_files": expected,
                    "retrieved_files": retrieved,
                }
            )

        time.sleep(1.5)

    if failures:
        print("\n=== Failures ===")
        for f in failures:
            print(f"- question: {f['question']}")
            print(f"  expected: {', '.join(f['expected_files'])}")
            print(f"  retrieved: {', '.join(f['retrieved_files'])}")

    print("\n=== Summary ===")
    print(f"Total: {total}")
    print(f"Recall@1: {recall_1 / total:.2%} ({recall_1}/{total})")
    print(f"Recall@{args.match_count}: {recall_k / total:.2%} ({recall_k}/{total})")


if __name__ == "__main__":
    main()