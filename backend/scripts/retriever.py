import os
from dotenv import load_dotenv
from supabase.client import create_client
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# 상위 폴더의 .env 경로 지정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, "../.env")
load_dotenv(dotenv_path=env_path, override=True)

IS_TESTING = os.getenv("IS_TESTING", "false") == "true"
if not IS_TESTING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

os.environ["LANGCHAIN_PROJECT"] = "parrot_rag"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_KEY")

# Supabase, LLM, 리랭커 모델 설정
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
reranker = CrossEncoder('BAAI/bge-reranker-large')

# HyDE를 위한 프롬프트 템플릿
hyde_template = """
당신은 앵무새 박사입니다. 다음 질문에 대해, 가장 이상적인 답변을 상세하고 친절하게 작성해주세요.
이 답변은 실제 사실이 아니어도 괜찮습니다. 오직 관련 문서를 찾기 위한 목적으로만 사용됩니다.
답변은 반드시 한국어로 작성해주세요.

질문: {question}
"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
hyde_chain = hyde_prompt | llm | StrOutputParser()

class Retriever:
    def __init__(self, supabase, embeddings):
        self.supabase = supabase
        self.embeddings = embeddings

    def hybrid_search(self, query_text, query_embedding, match_count, match_threshold):
        # DB 호출
        return self.supabase.rpc("hybrid_search", {
            "query_text": query_text,
            "match_count": match_count,
            "match_threshold": match_threshold,
            "query_embedding": query_embedding
        }).execute()

# 서비스에서 사용할 인스턴스 생성
retriever = Retriever(supabase, embeddings)

@traceable(name="retriever_logic_with_rerank", project_name="parrot_rag")
def retriever_logic(
    query: str,
    match_count: int = 3,
    match_threshold: float = 0.5,
    include_metadata: bool = False,
):
    print(f"디버그: 원본 쿼리 -> {query}")

    # 1. HyDE: 가상의 답변 생성
    hypothetical_document = hyde_chain.invoke({"question": query})
    query_embedding = embeddings.embed_query(hypothetical_document)
    
    # 2. 리랭킹을 위해 더 많은 후보군 확보 (예: 10개)
    CANDIDATE_COUNT = 10
    response = retriever.hybrid_search(query, query_embedding, CANDIDATE_COUNT, match_threshold)
    
    candidate_docs = response.data
    if not candidate_docs:
        print("디버그: 검색된 후보 문서 없음.")
        return []

    # 3. 리랭킹 수행
    pairs = [[query, doc['content']] for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs)
    
    # 리랭킹 점수를 후보군에 추가
    for doc, score in zip(candidate_docs, rerank_scores):
        doc['rerank_score'] = score
        
    # 리랭킹 점수 기준으로 내림차순 정렬
    reranked_docs = sorted(candidate_docs, key=lambda x: x['rerank_score'], reverse=True)
    
    # 4. 최종 결과 선택 (match_count 만큼)
    final_docs = reranked_docs[:match_count]

    if include_metadata:
        result = [
            {
                "content": row.get("content", ""),
                "file_name": row.get("metadata", {}).get("file_name", "unknown"),
                "score": row.get("rerank_score"), # 점수를 리랭킹 점수로 변경
            }
            for row in final_docs
        ]
    else:
        result = [row['content'] for row in final_docs]

    print(f"디버그: 최종 선택된 데이터 개수 -> {len(result)}")
    return result

if __name__ == "__main__":
    # .env 파일 로드
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
    result = retriever_logic("앵무새가 밤에 무서워하지 않게 하려면 어떻게 해야 해?")
    print(f"최종 결과: {result}")
