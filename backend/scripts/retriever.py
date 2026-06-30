import os
import re
import unicodedata
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable
from sentence_transformers import CrossEncoder
from supabase.client import create_client

# 로컬 BM25 구성을 위해 추가된 라이브러리
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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


# ----------------- 한국어 조사 제거 토크나이저 -----------------
def ko_tokenizer(text):
    # 유니코드 NFC 정규화 수행 (자소 분리 방지)
    normalized = unicodedata.normalize('NFC', text)
    # 특수기호 제거 및 영문/한글/숫자 토큰 추출
    tokens = re.findall(r'[가-힣a-zA-Z0-9]+', normalized)
    
    # 대표적인 한국어 조사 패턴
    josa_pattern = re.compile(r'(을|를|이|가|은|는|의|에|에게|에서|과|와|으로|로|라고|고|만|도|의)$')
    
    processed_tokens = []
    for token in tokens:
        if len(token) > 2:
            cleaned = josa_pattern.sub('', token)
            processed_tokens.append(cleaned)
        else:
            processed_tokens.append(token)
    return processed_tokens


# ----------------- 로컬 BM25 Retriever 빌드 -----------------
def init_local_bm25():
    print("[retriever] 로컬 BM25 인덱스를 빌드합니다...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.abspath(os.path.join(current_dir, "..", "data", "processed"))
    
    if not os.path.exists(folder_path):
        print(f"[retriever] 경고: processed 폴더가 없습니다: {folder_path}")
        return None
        
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
    )
    
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                md_chunks = markdown_splitter.split_text(content)
                chunks = text_splitter.split_documents(md_chunks)
                
                for chunk in chunks:
                    chunk.metadata['file_name'] = filename
                    all_chunks.append(chunk)
            except Exception as e:
                print(f"[retriever] 문서 로드 실패 ({filename}): {e}")
                
    if not all_chunks:
        print("[retriever] 로드된 문서 조각이 없어 BM25를 생성하지 못했습니다.")
        return None
        
    # 한국어 형태소 조사 제거 토크나이저를 넘겨주어 키워드 매칭율 향상
    local_bm25 = BM25Retriever.from_documents(all_chunks, preprocess_func=ko_tokenizer)
    local_bm25.k = 10
    print(f"[retriever] 로컬 BM25 인덱스 빌드 완료 (총 {len(all_chunks)}개 청크)")
    return local_bm25

local_bm25_retriever = init_local_bm25()

HYDE_TEMPLATE = """
당신은 반려 조류 전문가입니다. 다음 질문에 답하기 위해 관련 문서에서 찾을 법한 이상적인 답변을 한국어로 작성하세요.
이 답변은 실제 사용자에게 보여주기 위한 것이 아니라 검색 정확도를 높이기 위한 가상의 문서입니다.

질문: {question}
"""


@lru_cache(maxsize=1)
def get_hyde_chain():
    # HyDE 용도로는 속도가 빠르고 TPM 한도가 매우 높은 8B 모델(llama-3.1-8b-instant)을 사용해 최적화합니다.
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
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


def compute_rrf_docs(supabase_docs, bm25_docs, candidate_count=10, k=10):
    def normalize_text(text):
        normalized = unicodedata.normalize('NFC', text)
        return " ".join(normalized.split()).strip()

    rrf_dict = {}

    # 1. Supabase 문서 랭킹 부여
    for rank, doc in enumerate(supabase_docs, start=1):
        content = doc.get("content", "")
        norm_content = normalize_text(content)
        if not norm_content:
            continue
            
        metadata = doc.get("metadata", {})
        if norm_content not in rrf_dict:
            rrf_dict[norm_content] = {
                "content": content,
                "metadata": metadata,
                "rrf_score": 0.0
            }
        rrf_dict[norm_content]["rrf_score"] += 1.0 / (k + rank)

    # 2. BM25 문서 랭킹 부여
    for rank, doc in enumerate(bm25_docs, start=1):
        content = doc.page_content
        norm_content = normalize_text(content)
        if not norm_content:
            continue
            
        metadata = doc.metadata
        if norm_content not in rrf_dict:
            rrf_dict[norm_content] = {
                "content": content,
                "metadata": metadata,
                "rrf_score": 0.0
            }
        rrf_dict[norm_content]["rrf_score"] += 1.0 / (k + rank)

    # 3. RRF 점수 기준 정렬 및 상위 후보군 선택
    sorted_rrf = sorted(rrf_dict.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_rrf[:candidate_count]


@traceable(name="retriever_logic_with_rerank", project_name="parrot_rag")
def retriever_logic(
    query: str,
    match_count: int = 3,
    match_threshold: float = 0.5,
    include_metadata: bool = False,
):
    print(f"[retriever] query -> {query}")

    # 1. Supabase Hybrid Search 후보군 추출
    embedding_query = build_embedding_query(query)
    query_embedding = embeddings.embed_query(embedding_query)

    candidate_count = max(10, match_count)
    supabase_response = retriever.hybrid_search(
        query,
        query_embedding,
        candidate_count,
        match_threshold,
    )
    supabase_docs = supabase_response.data or []

    # 2. 로컬 BM25 Search 후보군 추출
    bm25_docs = []
    if local_bm25_retriever is not None:
        try:
            local_bm25_retriever.k = candidate_count
            bm25_docs = local_bm25_retriever.invoke(query)
        except Exception as e:
            print(f"[retriever] BM25 검색 중 오류 발생: {e}")

    # 3. RRF (Reciprocal Rank Fusion)를 통한 결과 병합
    combined_candidates = compute_rrf_docs(supabase_docs, bm25_docs, candidate_count=candidate_count)
    
    if not combined_candidates:
        print("[retriever] 검색된 후보 문서가 없습니다.")
        return []

    # 4. CrossEncoder Rerank
    pairs = [[query, doc["content"]] for doc in combined_candidates]
    rerank_scores = reranker.predict(pairs)

    for doc, score in zip(combined_candidates, rerank_scores):
        doc["rerank_score"] = float(score)

    reranked_docs = sorted(
        combined_candidates,
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
