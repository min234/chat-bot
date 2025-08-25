 
const BASE_URL = window.location.origin; 

// 채팅 UI 엘리먼트
const chatEl  = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');

function appendMessage(text, cls) {
  const div = document.createElement('div');
  div.className = `message ${cls}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

sendBtn.onclick = async () => {
  const q = inputEl.value.trim();
  if (!q) return;

  appendMessage(`You: ${q}`, 'user');
  inputEl.value = '';

  // ② fetch 호출주소를 BASE_URL로 교체
  const res = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: q })
  });

  if (!res.ok) {
    appendMessage('Bot: 오류가 발생했습니다.', 'bot');
    return;
  }

  const { reply } = await res.json();
  appendMessage(`Bot: ${reply}`, 'bot');
};
