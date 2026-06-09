import os
from dotenv import load_dotenv
from supabase import create_client  # <--- 이 줄이 반드시 있어야 합니다!
from sentence_transformers import SentenceTransformer

load_dotenv()

# .env 파일에서 데이터베이스 연결 주소를 가져옵니다.
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")
# 모델 로드 (BAAI/bge-m3는 1024차원)
model = SentenceTransformer('BAAI/bge-m3')

# (데이터 삽입 부분)
folder_path = "../data/processed"
for filename in os.listdir(folder_path):
    if filename.endswith(".md"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            embedding = model.encode(content).tolist()
            
            # DB 직접 접근 대신 API로 데이터 전송
            data = {"content": content, "metadata": {"file_name": filename}, "embedding": embedding}
            supabase.table("documents").insert(data).execute()
            print(f"저장 완료: {filename}")