import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable
from sentence_transformers import CrossEncoder
from supabase.client import create_client


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, "../.env")
load_dotenv(dotenv_path=env_path, override=False)

IS_TESTING = os.getenv("IS_TESTING", "false").lower() == "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false" if IS_TESTING else "true"
os.environ["LANGCHAIN_PROJECT"] = "parrot_rag"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_KEY", "")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-large")

HYDE_TEMPLATE = """
당신은 반려 조류 전문가입니다. 다음 질문에 답하기 위해 관련 문서에서 찾을 법한 이상적인 답변을 한국어로 작성하세요.
이 답변은 실제 사용자에게 보여주기 위한 것이 아니라 검색 정확도를 높이기 위한 가상의 문서입니다.

질문: {question}
"""


@lru_cache(maxsize=1)
def get_hyde_chain():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    hyde_prompt = ChatPromptTemplate.from_template(HYDE_TEMPLATE)
    return hyde_prompt | llm | StrOutputParser()


def build_embedding_query(query: str) -> str:
    if IS_TESTING:
        print("[retriever] IS_TESTING=true: HyDE를 건너뛰고 원 질문으로 임베딩합니다.")
        return query

    return get_hyde_chain().invoke({"question": query})


class Retriever:
    def __init__(self, supabase, embeddings):
        self.supabase = supabase
        self.embeddings = embeddings

    def hybrid_search(self, query_text, query_embedding, match_count, match_threshold):
        return self.supabase.rpc(
            "hybrid_search",
            {
                "query_text": query_text,
                "match_count": match_count,
                "match_threshold": match_threshold,
                "query_embedding": query_embedding,
            },
        ).execute()


retriever = Retriever(supabase, embeddings)


@traceable(name="retriever_logic_with_rerank", project_name="parrot_rag")
def retriever_logic(
    query: str,
    match_count: int = 3,
    match_threshold: float = 0.5,
    include_metadata: bool = False,
):
    print(f"[retriever] query -> {query}")

    embedding_query = build_embedding_query(query)
    query_embedding = embeddings.embed_query(embedding_query)

    candidate_count = max(10, match_count)
    response = retriever.hybrid_search(
        query,
        query_embedding,
        candidate_count,
        match_threshold,
    )

    candidate_docs = response.data
    if not candidate_docs:
        print("[retriever] 검색된 후보 문서가 없습니다.")
        return []

    pairs = [[query, doc["content"]] for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs)

    for doc, score in zip(candidate_docs, rerank_scores):
        doc["rerank_score"] = float(score)

    reranked_docs = sorted(
        candidate_docs,
        key=lambda row: row["rerank_score"],
        reverse=True,
    )
    final_docs = reranked_docs[:match_count]

    if include_metadata:
        result = [
            {
                "content": row.get("content", ""),
                "file_name": row.get("metadata", {}).get("file_name", "unknown"),
                "score": row.get("rerank_score"),
            }
            for row in final_docs
        ]
    else:
        result = [row["content"] for row in final_docs]

    print(f"[retriever] selected docs -> {len(result)}")
    return result


if __name__ == "__main__":
    result = retriever_logic("앵무새가 밤에 무서워하지 않게 하려면 어떻게 해야 해?")
    print(f"최종 결과: {result}")
