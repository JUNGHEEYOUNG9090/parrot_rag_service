import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.scripts.generator import generate_answer_stream
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    chat_history: list = []

@app.post("/ask")
async def ask(query: Query):
    return StreamingResponse(
        generate_answer_stream(query.question, query.chat_history), 
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    # 포트를 8000으로 지정합니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)