# test/search_test.py
from backend.scripts.retriever import get_retriever # retriever.py를 만들었다고 가정

retriever = get_retriever()
query = "앵무새가 깃털을 자꾸 뽑는데 원인이 뭐야?"

docs = retriever.invoke(query)

print(f"검색된 문서 개수: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- 문서 {i+1} ---")
    print(doc.page_content)