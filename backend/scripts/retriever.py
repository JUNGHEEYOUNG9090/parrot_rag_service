import os
import re
import unicodedata
from functools import lru_cache
from collections import defaultdict

from dotenv import load_dotenv
from langsmith import traceable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from supabase.client import create_client


# ================= ENV =================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, "../.env")
load_dotenv(env_path, override=False)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-large")


# ================= HYDE =================
HYDE_TEMPLATE = """
당신은 반려 조류 전문가입니다.
질문에 대한 검색용 문서를 작성하세요.

질문: {question}
"""

@lru_cache(maxsize=1)
def hyde_chain():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt = ChatPromptTemplate.from_template(HYDE_TEMPLATE)
    return prompt | llm | StrOutputParser()


def build_query(query: str):
    return hyde_chain().invoke({"question": query})


# ================= SUPABASE =================
class Retriever:
    def __init__(self, client):
        self.client = client

    def search(self, query, embedding, k=30, threshold=0.5):
        return self.client.rpc(
            "hybrid_search",
            {
                "query_text": query,
                "query_embedding": embedding,
                "match_count": k,
                "match_threshold": threshold,
            },
        ).execute()


retriever = Retriever(supabase)


# ================= MAIN =================
@traceable(name="retriever_logic")
def retriever_logic(
    query: str,
    match_count: int = 3,
    match_threshold: float = 0.5,
    include_metadata: bool = False,
):

    # 1. HyDE
    q = build_query(query)
    q_emb = embeddings.embed_query(q)

    # 2. retrieval (ONLY supabase)
    res = retriever.search(query, q_emb, k=30, threshold=match_threshold)
    docs = res.data or []

    if not docs:
        return []

    # 3. rerank (top N only for speed)
    docs = docs[:20]

    pairs = [[query, d["content"]] for d in docs]
    scores = reranker.predict(pairs)

    for d, s in zip(docs, scores):
        d["score"] = float(s)

    # 4. sort by rerank
    docs = sorted(docs, key=lambda x: x["score"], reverse=True)

    # 5. SIMPLE DIVERSITY (file cap only)
    file_count = defaultdict(int)
    final = []

    for d in docs:
        fn = d["metadata"].get("file_name", "unknown")

        if file_count[fn] >= 2:
            continue

        final.append(d)
        file_count[fn] += 1

        if len(final) >= match_count:
            break

    # fallback
    if not final:
        final = docs[:match_count]

    # 6. output
    if include_metadata:
        return [
            {
                "content": d["content"],
                "file_name": d["metadata"].get("file_name"),
                "score": d["score"],
            }
            for d in final
        ]

    return [d["content"] for d in final]


# ================= TEST =================
if __name__ == "__main__":
    print(
        retriever_logic("앵무새가 밤에 무서워하지 않게 하려면?")
    )