from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from langdetect import detect
from ve_db import search
 

load_dotenv(dotenv_path=".env", override=True)
api_key = os.getenv("OPENAI_API_KEY")

chat_system = """\
  You are a friendly Vietnamese QA assistant. Always consider the "context line" provided as your sole source of truth.
 
For each user query, do the following:
0. When using language, always be respectful,When calling a user, always call them customer in Vietnamese.
1. If CONTEXT is not empty, write the answer using all lines _internally_.
2. Do not display or quote any lines from CONTEXT in the output.
3. Write a concise, conversational response that fully integrates the content of CONTEXT.
4. If CONTEXT is empty, reply exactly as follows: "Bạn có thể liên hđ Bộ phận Chăm sóc khách hàng trực tiếp qua Livechat trên trên app hoặc website https://bitruth.com/"
5. For the query "Xin chào", display only the sentence "Xin chào, Bitruth's trợ thôngtin gì ạ?"
6. If you don't understand the website, please click here to visit our website: https://bitruth.com/ 
7. You don't say anything vi
8. If it says buy and sell, show sentence 4.
 9.If you ask what kind of games there are, you will answer "Trò chòi hiđen đang đuợc tiến hành là FUTURES X1000 (MÔ PHỎNG GIAO DỊCH)".
,Option Game(DỰ Don Gia 1 PÚT)
CAO/THẤP (DỰ дOÁN ẾN NGẮN ẠN)."
Make it come out like this.
       """

chat_system_template = """\
  User Query: {user_request}
  lang: {lang}

  Context lines:
  {sim_sent_block}

  Using ONLY the information above, provide your answer. Do NOT include or quote the CONTEXT lines themselves—only their semantic content.
If the question is about long-term, short-term, investment or trading, please do not say anything and just show the answer below. Do not include context lines and just show the answer. 
Answer: Bitruth xin phép không đua ra bất kỳ lời khuyên đầu tù nào. There are many questions and answers. There is no Bitrus.
  
"""

async def generate_content(user_request: str) -> str:
    client = AsyncOpenAI(api_key=api_key)
    lang = detect(user_request)
    if lang not in ("en", "vi"):
        lang = "vi"

    hits = search(user_request, k_records=5)
    texts = []
    print(hits)
    for rec in hits:
        parts = [
            rec['big_title'],        # 최상위 모듈 이름
            rec['section'],          # 섹션 제목
            rec['subsection'],       # 서브섹션 제목
            rec['sub_title_id'],     # 세부 제목 (빈 문자열일 수도)
            " ".join(rec['items'])   # 실제 항목들
        ]
        # 빈 문자열 제거 후, 모두 합쳐서 하나의 문장으로
        text = " | ".join([p for p in parts if p])
        texts.append(text)
            

    sim_sent_block = "\n".join(f"- {t}" for t in texts)
    print(sim_sent_block)
    user_content = chat_system_template.format(
        user_request=user_request,
        lang=lang,
        sim_sent_block=sim_sent_block
    )

    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",  "content": chat_system},
            {"role": "user",    "content": user_content},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    print(f"[tokens] prompt: {resp.usage.prompt_tokens}, completion: {resp.usage.completion_tokens}, total: {resp.usage.total_tokens}")
    return resp.choices[0].message.content.strip()

async def main():
    client = AsyncOpenAI(api_key=api_key)
    print("Bitruth QA Assistant (exit 또는 quit 입력 시 종료)")
    while True:
        user_input = input("\nYour query> ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        try:
            answer = await generate_content( user_input)
            print("\nAI Response:", answer)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
