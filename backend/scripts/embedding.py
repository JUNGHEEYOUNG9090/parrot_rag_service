import os
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. 환경 변수 및 클라이언트 설정
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
model = SentenceTransformer('BAAI/bge-m3')

# 2. 기존 데이터 삭제
print("기존 'documents' 테이블의 데이터를 모두 삭제합니다...")
supabase.table("documents").delete().neq("id", -1).execute()
print("데이터 삭제 완료.")

# 3. Markdown 분할 설정
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

# 4. 문서 처리 및 임베딩
# 수정 후 (코드 파일의 위치를 기준으로 절대 경로 생성)
current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts 폴더 위치
folder_path = os.path.abspath(os.path.join(current_dir, "..", "..", "backend", "data", "processed"))
total_chunks = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".md"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Markdown 문서를 의미 단위로 분할
        chunks = markdown_splitter.split_text(content)
        print(f"\n처리 중인 파일: {filename} ({len(chunks)}개 조각)")
        
        for i, chunk in enumerate(chunks):
            chunk_content = chunk.page_content
            # 메타데이터에 원본 파일 이름 추가
            chunk_metadata = chunk.metadata
            chunk_metadata['file_name'] = filename
            
            # 각 조각(chunk)을 임베딩
            embedding = model.encode(chunk_content).tolist()
            
            # Supabase에 저장
            data = {"content": chunk_content, "metadata": chunk_metadata, "embedding": embedding}
            supabase.table("documents").insert(data).execute()
            print(f"  -> 조각 {i+1}/{len(chunks)} 저장 완료")
            total_chunks += 1

print(f"\n\n총 {total_chunks}개의 조각(Chunk)을 데이터베이스에 저장했습니다.")
print("모든 문서의 분할 및 임베딩 작업이 완료되었습니다.")