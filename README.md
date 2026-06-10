# Parrot RAG Chatbot

앵무새 관련 문서를 기반으로 질문에 답변하는 RAG 챗봇 프로젝트입니다.

PDF/Markdown 문서를 임베딩해 Supabase에 저장하고, 사용자의 질문이 들어오면 hybrid search로 관련 문서를 검색한 뒤 Groq LLM으로 한국어 답변을 생성합니다. React 프론트엔드는 FastAPI 서버의 스트리밍 응답을 받아 채팅 UI로 보여줍니다.

## 주요 기능

- 앵무새 관련 문서 기반 질의응답
- Supabase RPC 기반 hybrid search
- `BAAI/bge-m3` 임베딩 모델 사용
- Groq `llama-3.3-70b-versatile` 기반 답변 생성
- FastAPI `StreamingResponse`를 통한 답변 스트리밍
- React + Vite + Tailwind 기반 채팅 UI
- 대화 이력을 활용한 후속 질문 검색 보강
- 내부 문서 정보가 부족할 때 Tavily 웹 검색 fallback
- 검색 품질 평가 스크립트 제공

## 기술 스택

### Backend

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)
![Tavily](https://img.shields.io/badge/Tavily-111827?style=for-the-badge&logoColor=white)

### Frontend

![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)

## 프로젝트 구조

```text
.
├── main.py
├── backend
│   ├── data
│   │   ├── raw
│   │   └── processed
│   └── scripts
│       ├── embedding.py
│       ├── generator.py
│       └── retriever.py
├── frontend
│   ├── src
│   │   ├── App.tsx
│   │   ├── main.jsx
│   │   └── index.css
│   └── package.json
└── test
    ├── eval_questions.json
    ├── eval_retriever.py
    ├── search_test.py
    └── test_embed.py
```

## RAG 동작 흐름

<img src="./img/동작흐름.png" width="400" />

1. 문서를 Markdown으로 가공
2. `embedding.py`가 문서를 읽고 `BAAI/bge-m3`로 임베딩
3. 문서 본문, 파일명 metadata, embedding을 Supabase `documents` 테이블에 저장
4. 사용자가 React UI에서 질문을 입력
5. FastAPI `/ask` 엔드포인트가 질문과 대화 이력을 받아옴
6. `generator.py`가 최근 대화 이력을 현재 질문에 붙여 검색용 query를 만듦
7. `retriever.py`가 Supabase `hybrid_search` RPC로 관련 문서를 검색
8. 검색된 문서를 context로 넣어 Groq LLM이 답변을 생성
9. 답변은 스트리밍으로 프론트엔드에 전달

## 환경 변수

루트의 `.env.example`을 복사해 `.env` 파일을 만들고 값을 설정

```powershell
copy .env.example .env
```

```env
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
DATABASE_URL=
GROQ_API_KEY=
LANGSMITH_KEY=
TAVILY_API_KEY=
```

## 실행 방법

### Backend

```powershell
.\venv\Scripts\uvicorn.exe main:app --reload
```

기본 서버 주소:

```text
http://localhost:8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

프론트엔드 / 백엔드를 한 번에 실행:

```powershell
cd frontend
npm run start-all
```

## 문서 임베딩

Markdown 문서 Supabase에 저장:

```powershell
.\venv\Scripts\python.exe backend\scripts\embedding.py
```

문서는 `backend/data/processed`의 `.md` 파일

## 검색 품질 평가

검색 성능은 `test/eval_questions.json`의 질문-기대문서 평가셋으로 측정하였음.

```powershell
.\venv\Scripts\python.exe test\eval_retriever.py
```

현재 평가 결과:

```text
Total: 10
Recall@1: 90.00% (9/10)
Recall@3: 90.00% (9/10)
```

이 평가는 리랭커를 바로 붙이기 전에 현재 hybrid search만으로 어느 정도 검색 품질이 나오는지 확인하기 위해 추가.

문서 수가 많지 않기 때문에 리랭커 도입보다 실패 케이스 분석, threshold 조정, query 보강이 우선이라고 판단.

```text
retriever 설정
query: str
match_count: int = 3
match_threshold: float = 0.5
include_metadata: bool = False
```

## 실행 화면

### 채팅 UI

<img src="./img/실행화면.PNG" width="400" />

### 검색 품질 평가

<img src="./img/품질평가.PNG" width="400" />

## 주요 트러블슈팅

### 후속 질문 검색 실패

사용자가 "이게 왜 중요한데?"처럼 짧은 후속 질문을 하면 검색기가 현재 질문만 보고 검색해서 결과가 0개가 되는 문제

이를 해결하기 위해 `generator.py`에서 최근 대화 이력과 현재 질문을 합쳐 검색용 query 생성

```python
retrieval_query = f"{history_text}\n현재 질문: {query}" if history_text else query
context_docs = retriever_logic(retrieval_query)
```

답변 생성에는 원래 사용자 질문을 그대로 사용하고, 검색에만 보강된 query를 사용

### UI 줄바꿈과 말풍선 크기

LLM 답변에 줄바꿈이 포함되어도 React 화면에서는 한 줄로 접혀 보이는 문제

메시지 말풍선에 Tailwind를 수정하여 해결

```tsx
whitespace-pre-wrap break-words w-fit max-w-[80%]
```

### 문서 출력 문제

#### Markdown 굵게 표시

LLM이 `**중요 키워드**` 형식으로 답변했지만 화면에는 `**`가 그대로 보임.

React에서 메시지 렌더링 시 `**텍스트**`를 `<strong>`으로 변환하도록 처리

#### 괄호 제거 문제

백엔드 출력 필터 정규식에서 괄호가 허용되지 않아 `채집 활동(Foraging)`이 `채집 활동Foraging`처럼 붙어 출력

정규식 사용하여 `()`를 추가해 해결

```python
r'[^가-힣a-zA-Z0-9\s.,!?\n*\-()]'
```

## 앞으로 개선할 점

- 답변 하단에 참고 문서 출처 표시
- 평가 질문셋을 20~30개 이상으로 확장
- 실패 케이스 기반 문서 chunking 개선
- Supabase `hybrid_search` score 반환값 정리
- 필요 시 LLM 기반 query rewrite 도입
- 배포 환경 구성
