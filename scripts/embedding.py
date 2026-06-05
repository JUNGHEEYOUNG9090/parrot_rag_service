import os
import psycopg2
import json # 추가
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

# 모델 선정 이유 : ../test/emb_compare_models.py에 서술
model = SentenceTransformer('BAAI/bge-m3')

load_dotenv()

db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")

conn = psycopg2.connect(
    dbname=db_name,
    user=db_user,
    password=db_password,
    host=db_host
)

cur = conn.cursor()


folder_path = "../data/processed" 
for filename in os.listdir(folder_path):
    if filename.endswith(".md"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            # 임베딩 생성 (1024차원)
            embedding = model.encode(content).tolist()
            
            # DB 저장
            cur.execute(
                "INSERT INTO parrot_rag.document_embeddings (content, metadata, embedding) VALUES (%s, %s, %s)",
                (content, f'{{"file_name": "{filename}"}}', embedding)
            )

conn.commit()
cur.close()
conn.close()