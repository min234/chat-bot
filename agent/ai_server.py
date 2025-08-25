# ai_server.py
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import asyncio

# your AI logic
from open import generate_content

app = FastAPI()

# 정적 파일(클라이언트 JS, CSS 등)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 요청/응답 스키마
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    reply: str
_history_lock = asyncio.Lock()
HISTORY_PATH = "chat_history.jsonl"
# 1) 채팅 화면 제공
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 2) 챗 API
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_q = req.query
    print("User query:", user_q)

    # AI에게 답변 요청
    reply_text = await generate_content(user_request=user_q)
    print("AI reply:", reply_text)

    # 기록 저장 (JSONL: 한 줄에 하나의 JSON 객체)
    async with _history_lock:
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            record = {"query": user_q, "reply": reply_text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return ChatResponse(reply=reply_text)
    
if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8080, reload=True)
