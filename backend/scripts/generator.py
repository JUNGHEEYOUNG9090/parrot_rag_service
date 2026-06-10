import os
import re
from dotenv import load_dotenv
from backend.scripts.retriever import retriever_logic
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, "../.env")

# 1. 모델 설정 (Groq의 Llama 3 사용)
llm = ChatGroq(model="llama-3.3-70b-versatile")
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
# 2. 질문에 답변하기 위한 프롬프트 템플릿
template = """
당신은 앵무새를 사랑하는 다정한 전문가입니다. 
아래 '문맥'을 활용하여 사용자의 질문에 다정하게 대화하듯 답변하세요.

규칙:
1. '문맥'에 포함된 정보는 답변의 절대적 근거입니다. 반드시 이를 우선하여 사용하세요.
2. 이전 대화 맥락을 파악하여 질문의 의도를 정확히 이해하고 대화하세요.
3. 딱딱한 정보 나열보다는 사용자의 상황에 공감하는 다정한 어조로 답변하세요.
4. '문맥'에서 답을 찾을 수 없거나 관련 없는 질문이라면, 모르는 척하지 말고 솔직하게 "해당 내용은 제 정보에 없습니다"라고 말하세요.
5. 오직 한국어로만 작성하세요. 한자, 영어, 일본어 등 다른 언어는 절대 포함하지 마세요.
6. 이전에 했던 말을 그대로 반복하지 말고, 질문에 직접적인 해결책을 명확히 제시하세요.
7. 정보를 나열할 때는 반드시 항목별로 줄바꿈(`\n`)을 하고 불렛 포인트(-)를 사용하세요.
8. 각 문단 사이에는 반드시 빈 줄을 하나씩 넣어 가독성을 높이세요.
9. 핵심 키워드는 **굵게** 처리하여 강조하세요.

[출력 규칙 - 반드시 지킬 것]
1. 답변은 절대로 한 문단으로 작성하지 마세요. 
2. 반드시 주제별로 2~3문장씩 끊어서 '엔터(줄바꿈)'를 두 번씩 치세요.
3. 중요한 내용은 불렛 포인트(-)를 사용하여 리스트 형태로 출력하세요.
4. 모든 문단 사이에는 빈 줄을 하나씩 반드시 포함하세요.

문맥: {context}
이전 대화 맥락: {history}
질문: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. 답변 생성 함수 (LangSmith 추적 가능)
@traceable(name="generator_logic", project_name="parrot_rag")
async def generate_answer_stream(query: str, chat_history: list):
    history_text = "\n".join([f"{msg['type']}: {msg['text']}" for msg in chat_history[-3:]])
    retrieval_query = f"{history_text}\n현재 질문: {query}" if history_text else query
    context_docs = retriever_logic(retrieval_query)
    unique_context = list(set(context_docs))
    context_text = "\n".join(unique_context)
    
    chain = prompt | llm | StrOutputParser()
    inputs = {
        "context": context_text, 
        "history": history_text, 
        "question": query
    }
    answer = await chain.ainvoke(inputs)
    
    # 2. 정보 부족 시 웹 검색 및 스트리밍 분기
    if "해당 내용이 없습니다" in answer or len(context_text) < 50:
        print("\n[알림] 내부 정보 부족, 웹 검색을 시작합니다...")
        try:
            search_result = tavily.search(query=query, search_depth="advanced")
            web_context = "\n".join([r['content'] for r in search_result['results']])
            
            # 웹 검색 결과는 LLM을 통해 스트리밍으로 출력
            final_prompt = f"다음 웹 정보를 바탕으로 질문에 답하세요: {web_context}\n\n질문: {query}"
            async for chunk in llm.astream(final_prompt):
                filter_chunk = re.sub(r'[^가-힣a-zA-Z0-9\s.,!?\n*\-()]', '', chunk)
                if filter_chunk:
                    yield filter_chunk
                
        except Exception as e:
            yield "죄송합니다. 정보를 찾을 수가 없습니다."
            
    else:
        # 기존 답변이 있는 경우도 스트리밍으로 출력
        async for chunk in chain.astream(inputs):
            filter_chunk = re.sub(r'[^가-힣a-zA-Z0-9\s.,!?\n*\-()]', '', chunk)
            if filter_chunk:
                yield filter_chunk
                
# 4. 실행
if __name__ == "__main__":None
