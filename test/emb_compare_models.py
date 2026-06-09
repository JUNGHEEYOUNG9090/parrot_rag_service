from sentence_transformers import SentenceTransformer
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 모델 로드
models = {
    "BGE-M3": SentenceTransformer('BAAI/bge-m3'),
    "KoSimCSE": SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
}

# 테스트 쿼리
test_query = "특수동물병원에서 조류를 진료할 때 주의할 점은 무엇인가요?"

def compare_models(text_sample):
    results = {}
    for name, model in models.items():
        start_time = time.time()
        embedding = model.encode(text_sample)
        latency = time.time() - start_time
        results[name] = {"embedding": embedding, "latency": latency}
    return results

def measure_latency(model, text, iterations=10):
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model.encode(text)
        latencies.append(time.perf_counter() - start)
    return np.mean(latencies) * 1000 # ms 단위

for name, model in models.items():
    avg_latency = measure_latency(model, test_query)
    print(f"{name} 평균 추론 속도: {avg_latency:.2f}ms")

test_doc = "조류는 아픈 것을 숨기는 습성이 있어 보호자의 세심한 관찰과 일상적인 기록이 필수적입니다."

def evaluate_quality(models, query, doc):
    print(f"\n품질 비교 (질문: {query})")
    for name, model in models.items():
        q_vec = model.encode([query])
        d_vec = model.encode([doc])
        score = cosine_similarity(q_vec, d_vec)[0][0]
        print(f"{name} 유사도 점수: {score:.4f}")

evaluate_quality(models, test_query, test_doc)



# BGE-M3 평균 추론 속도: 199.05ms
# KoSimCSE 평균 추론 속도: 56.42ms

# 품질 비교 (질문: 특수동물병원에서 조류를 진료할 때 주의할 점은 무엇인가요?)
# BGE-M3 유사도 점수: 0.6104
# KoSimCSE 유사도 점수: 0.5748

# BGE-M3으로 임베딩 모델 선정