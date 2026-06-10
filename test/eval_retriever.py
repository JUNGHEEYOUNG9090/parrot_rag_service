## 검색품질평가기

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.scripts.retriever import retriever_logic


def load_eval_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_file_name(file_name: str) -> str:
    return file_name.replace("\\", "/").split("/")[-1]


def hit_at_k(retrieved_files, expected_files, k: int) -> bool:
    top_k = set(retrieved_files[:k])
    expected = set(expected_files)
    return bool(top_k & expected)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever Recall@k.")
    parser.add_argument(
        "--eval-file",
        default="test/eval_questions.json",
        help="Path to evaluation question JSON.",
    )
    parser.add_argument("--match-count", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cases = load_eval_cases(Path(args.eval_file))
    total = len(cases)
    recall_1_hits = 0
    recall_k_hits = 0
    failures = []

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        expected_files = [normalize_file_name(name) for name in case["expected_files"]]
        docs = retriever_logic(
            question,
            match_count=args.match_count,
            match_threshold=args.threshold,
            include_metadata=True,
        )
        retrieved_files = [
            normalize_file_name(doc["file_name"])
            for doc in docs
        ]

        recall_1 = hit_at_k(retrieved_files, expected_files, 1)
        recall_k = hit_at_k(retrieved_files, expected_files, args.match_count)
        recall_1_hits += int(recall_1)
        recall_k_hits += int(recall_k)

        status = "PASS" if recall_k else "FAIL"
        print(f"\n[{index}/{total}] {status} {question}")
        print(f"expected: {', '.join(expected_files)}")
        print(f"retrieved: {', '.join(retrieved_files) or '(none)'}")

        if not recall_k:
            failures.append(
                {
                    "question": question,
                    "expected_files": expected_files,
                    "retrieved_files": retrieved_files,
                }
            )

    print("\n=== Summary ===")
    print(f"Total: {total}")
    print(f"Recall@1: {recall_1_hits / total:.2%} ({recall_1_hits}/{total})")
    print(
        f"Recall@{args.match_count}: "
        f"{recall_k_hits / total:.2%} ({recall_k_hits}/{total})"
    )

    if failures:
        print("\n=== Failures ===")
        for failure in failures:
            print(f"- question: {failure['question']}")
            print(f"  expected: {', '.join(failure['expected_files'])}")
            print(f"  retrieved: {', '.join(failure['retrieved_files']) or '(none)'}")


if __name__ == "__main__":
    main()
