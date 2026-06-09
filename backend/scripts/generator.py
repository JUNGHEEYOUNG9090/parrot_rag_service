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
사용자의 질문에 대해 아래 '문맥'을 참고하여 대화하듯 답변하세요.

규칙:
1. 사용자의 질문이 이전 대화와 연결된다면, 그 맥락을 충분히 반영하세요.
2. 정보만 딱딱하게 나열하지 말고, 사용자의 상황을 먼저 공감하고 다정하게 조언하세요.
3. 제공된 문맥에서 답을 찾기 어렵다면, 솔직하게 모르겠다고 말하거나 검색 결과를 정중히 요약하세요.
4. 답변은 반드시 완벽한 '한국어' 문장으로만 작성하세요. 
5. 한자, 일본어, 베트남어 등 한국어가 아닌 문자는 절대 포함하지 마세요.
6. 이전에 생성한 내용을 반복하지 말고, 간결하고 명확하게 답변하세요.

문맥: {context}
이전 대화 맥락: {history}
질문: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. 답변 생성 함수 (LangSmith 추적 가능)
@traceable(name="generator_logic", project_name="parrot_rag")
async def generate_answer_stream(query: str, chat_history: list):
    history_text = "\n".join([f"{msg['type']}: {msg['text']}" for msg in chat_history[-3:]])
    context_docs = retriever_logic(query)
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
                filter_chunk = re.sub(r'[^가-힣a-zA-Z0-9\s.,!?]', '', chunk)
                if filter_chunk:
                    yield filter_chunk
                
        except Exception as e:
            yield "죄송합니다. 현재 외부 정보 검색 기능을 사용할 수 없습니다."
            
    else:
        # 기존 답변이 있는 경우도 스트리밍으로 출력
        async for chunk in chain.astream(inputs):
            filter_chunk = re.sub(r'[^가-힣a-zA-Z0-9\s.,!?]', '', chunk)
            if filter_chunk:
                yield filter_chunk
                
# 4. 실행
if __name__ == "__main__":None
