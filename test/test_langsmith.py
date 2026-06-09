import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Groq
llm = ChatGroq(model="llama-3.3-70b-versatile")
response = llm.invoke("랭스미스 연동 테스트")

print(response.content)