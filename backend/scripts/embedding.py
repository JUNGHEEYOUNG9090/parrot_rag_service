import os
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")
# 모델 로드 (BAAI/bge-m3는 1024차원)
model = SentenceTransformer('BAAI/bge-m3')

folder_path = "../data/processed"
for filename in os.listdir(folder_path):
    if filename.endswith(".md"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            embedding = model.encode(content).tolist()
            
            data = {"content": content, "metadata": {"file_name": filename}, "embedding": embedding}
            supabase.table("documents").insert(data).execute()
            print(f"저장 완료: {filename}")