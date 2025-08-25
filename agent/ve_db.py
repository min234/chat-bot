import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(dotenv_path=".env", override=True)

# 메타 파일 경로
META_PATH = Path("last_data/meta.jsonl")

# 1) 메타 데이터 로드
meta = {}
with open(META_PATH, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        rec = json.loads(line)
        meta[str(idx)] = rec

# 2) 임베딩 모델 초기화
embed = SentenceTransformer("sentence-transformers/LaBSE")

# 3) big_title+subsection 임베딩 & FAISS 인덱스 생성
texts = []
idxs = []
for idx, rec in meta.items():
    txt = rec.get("big_title", "")
    sub = rec.get("subsection", "")
    if sub:
        txt += " | " + sub
    texts.append(txt)
    idxs.append(int(idx))

coarse_emb = embed.encode(texts, normalize_embeddings=True).astype("float32")
dim = coarse_emb.shape[1]
coarse_index = faiss.IndexFlatIP(dim)
coarse_index.add(coarse_emb)

def search(query: str, k_records: int = 10):
    """
    big_title + subsection 으로만 상위 k_records 레코드 반환
    """
    # 1) 쿼리 임베딩
    q_vec = embed.encode([query], normalize_embeddings=True).astype("float32")

    # 2) coarse 검색: top k_records id
    scores, id_arr = coarse_index.search(q_vec, k_records)
    top_ids = id_arr[0]
    top_scores = scores[0]

    # 3) 결과 리스트 구성
    results = []
    for idx, score in zip(top_ids, top_scores):
        if idx < 0:
            continue
        rec = meta.get(str(int(idx)))
        if not rec:
            continue

        # JSON 형태로 리턴
        results.append({
            "score":      float(score),
            "big_title":  rec["big_title"],
            "section":    rec["section"],
            "subsection": rec["subsection"],
            "sub_title_id": rec.get("sub_title_id", ""),
            "items":      rec.get("items", []),
        })

    return results

# 예제 실행
if __name__ == "__main__":
    q = "Đang có game gì"
    out = search(q, k_records=10)
    print(json.dumps(out, ensure_ascii=False, indent=2))
