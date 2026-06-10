import os
from dotenv import load_dotenv
from supabase.client import create_client
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

# 상위 폴더의 .env 경로 지정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, "../.env")
load_dotenv(dotenv_path=env_path, override=True)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "parrot_rag"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_KEY")

# Supabase 설정
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 데코레이터 
@traceable(name="retriever_logic", project_name="parrot_rag")
def retriever_logic(
    query: str,
    match_count: int = 3,
    match_threshold: float = 0.5,
    include_metadata: bool = False,
):
    print(f"디버그: 쿼리 실행 중 -> {query}")
    query_embedding = embeddings.embed_query(query)
    
    response = supabase.rpc("hybrid_search", {
        "query_text":query,
        "match_count": match_count,
        "match_threshold": match_threshold, 
        "query_embedding": query_embedding
    }).execute()

    if include_metadata:
        result = [
            {
                "content": row.get("content", ""),
                "file_name": row.get("metadata", {}).get("file_name", "unknown"),
                "score": row.get("similarity") or row.get("score") or row.get("rank"),
            }
            for row in response.data
        ]
    else:
        result = [row['content'] for row in response.data]

    print(f"디버그: 검색된 데이터 개수 -> {len(result)}")
    return result

if __name__ == "__main__":
    result = retriever_logic("나의 프로젝트 내용은 뭐야?")
    print(f"최종 결과: {result}")
