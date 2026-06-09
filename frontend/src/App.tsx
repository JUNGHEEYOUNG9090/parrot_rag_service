import { useState, useRef, KeyboardEvent } from "react";

interface ChatMessage {
  type: "user" | "bot";
  text: string;
}

function App() {
  const [question, setQuestion] = useState<string>("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  // 1. 엔터 키 처리를 위한 함수
  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSend();
  };

  // 2. 스트리밍 지원하는 전송 함수
  const handleSend = async () => {
    if (!question.trim()) return;

    const userQuestion = question;
    const newHistory: ChatMessage[] = [
      ...chatHistory,
      { type: "user", text: userQuestion },
    ];
    setChatHistory(newHistory);
    setQuestion("");
    setChatHistory((prev) => [...prev, { type: "bot", text: "" }]);

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userQuestion,
          chat_history: chatHistory,
        }),
      });

      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // 스트림 인코딩 처리
        const chunk = decoder.decode(value, { stream: true });

        setChatHistory((prev) => {
          const updated = [...prev];
          const lastIndex = updated.length - 1;
          // 기존 텍스트에 새로운 청크를 더할 때,
          // 혹시라도 중복되는 데이터가 있는지 확인하거나
          // 단순히 누적되도록 확실하게 보장합니다.
          updated[lastIndex] = {
            ...updated[lastIndex],
            text: updated[lastIndex].text + chunk,
          };
          return updated;
        });
      }
    } catch (e) {
      setChatHistory((prev) => [
        ...prev.slice(0, -1),
        { type: "bot", text: "서버 연결 오류!" },
      ]);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4 bg-gray-50">
      <h1 className="text-2xl font-bold text-center mb-6">🦜 앵무새-BOT</h1>
      <div className="flex-1 overflow-y-auto space-y-4 p-4 bg-white rounded-3xl shadow-lg">
        {chatHistory.map((chat, i) => (
          <div
            key={i}
            className={`p-4 rounded-2xl max-w-[80%] ${chat.type === "user" ? "bg-blue-500 text-white self-end ml-auto" : "bg-gray-200"}`}
          >
            {chat.text}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="mt-4 flex gap-2">
        <input
          className="flex-1 p-4 rounded-full border border-gray-300 outline-none"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyPress} // 여기서 연결!
          placeholder="질문을 입력하세요..."
        />
        <button
          className="px-6 py-4 bg-blue-600 text-white rounded-full"
          onClick={handleSend}
        >
          전송
        </button>
      </div>
    </div>
  );
}

export default App;
