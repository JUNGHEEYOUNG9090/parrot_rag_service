# test/search_test.py
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# .env 파일 로드
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from backend.scripts.retriever import retriever_logic 


query = "앵무새가 깃털을 자꾸 뽑는데 원인이 뭐야?"

docs = retriever_logic(query, include_metadata=True)

print(f"검색된 문서 개수: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- 문서 {i+1} ---")
    print(f"파일: {doc['file_name']}")
    print(f"점수: {doc['score']}")
    print(doc["content"][:500])
