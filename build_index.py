import json
import pathlib
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1) 경로 · 모델 설정
SRC        = "game.json"
OUT_DIR    = pathlib.Path("last_data")
OUT_DIR.mkdir(exist_ok=True)
INDEX_PATH = OUT_DIR / "game_pairs.index"
META_PATH  = OUT_DIR / "meta.jsonl"
MODEL_NAME = "sentence-transformers/LaBSE"

# 2) JSON 로드
records = json.load(open(SRC, encoding="utf-8"))

# 3) flat_records 생성
flat_records = []

modules = records.get("modules", [])
for module in modules:
    big_title = module.get("big_title", "")

    for section in module.get("sections", []):
        section_title = section.get("section", "")

        # 첫 번째 레벨 subsections
        for subsection in section.get("subsections", []):
            parent_title = subsection.get("title", section_title)
             
            # case 1: 키가 "sub"인 경우
            for sub in subsection.get("sub", []):
                flat_records.append({
            
                    "big_title":  big_title,
                    "section":    section_title,
                    "subsection": parent_title,
                    "sub_title_id":  sub.get("title", ""),
                    "items":      sub.get("items", [])
                })

            # case 2: 키가 "subsections"인 경우
            for sub in subsection.get("subsections", []):
                flat_records.append({
            
                    "big_title":  big_title,
                    "section":    section_title,
                    "subsection": parent_title,
                    "sub_title_id":  sub.get("title", ""),
                    "items":      sub.get("items", [])
                })

            # case 3: 둘 다 없는 경우 (직접 items 처리)
            if not subsection.get("sub") and not subsection.get("subsections"):
                flat_records.append({
            
                    "big_title":  big_title,
                    "section":    section_title,
                    "subsection": parent_title,
                    "sub_title_id":  "",
                    "items":      subsection.get("items", [])
                })
# 4) DataFrame
df = pd.DataFrame(flat_records)
print("=== 샘플 레코드 ===")
print(df.head(), "\n")
texts = [
    # big_title → section → subsection → sub_title_id → items 순서로 모두 포함
    f"{rec['big_title']} {rec['section']} {rec['subsection']} {rec['sub_title_id']} "
    + " ".join(rec['items'])
    for rec in flat_records
]
# 5) 임베딩 생성
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
).astype("float32")

# 6) FAISS 인덱스 구축
d = embeddings.shape[1]
index_flat = faiss.IndexFlatIP(d)
index = faiss.IndexIDMap(index_flat)

ids = np.arange(len(df), dtype="int64")
index.add_with_ids(embeddings, ids)
print("벡터 개수:", index.ntotal)

# 7) 인덱스 저장
faiss.write_index(index, str(INDEX_PATH))
print("✅  인덱스 저장 →", INDEX_PATH)

# 8) meta.jsonl 쓰기
with META_PATH.open("w", encoding="utf-8") as f:
    for rec in flat_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("✅  메타 저장 →", META_PATH)
