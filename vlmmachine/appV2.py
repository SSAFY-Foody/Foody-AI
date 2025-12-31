import io
import os
import json
import re
from typing import Optional, List, Dict, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from sqlalchemy import create_engine, Column, String, Float, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from dotenv import load_dotenv

# .env 로드
load_dotenv()

# SQL 설정
DATABASE_URL = os.getenv("FOODY_DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("FOODY_DATABASE_URL 환경변수가 설정되어 있지 않습니다.")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 기존 FOOD 테이블 매핑
class Foods(Base):
    __tablename__ = "foods"

    code = Column(String(45), primary_key=True)
    name = Column(String(30), nullable=False)
    standard = Column(String(10), nullable=False)
    kcal = Column(Float, nullable=False, default=0)

    # 실제 DB 컬럼 매핑
    carb = Column("carb_g", Float, nullable=False, default=0)
    protein = Column("protein_g", Float, nullable=False, default=0)
    fat = Column("fat_g", Float, nullable=False, default=0)
    sugar = Column("sugar_g", Float, nullable=False, default=0)
    natrium = Column("natrium_g", Float, nullable=False, default=0)


# 응답 모델(JSON)
class FoodResponse(BaseModel):
    name: str
    standard: str
    kcal: float
    carb: float
    protein: float
    fat: float
    sugar: float
    natrium: float


def round2(value: float) -> float:
    return round(float(value), 2)


# Qwen VLM 클라이언트
from pathlib import Path

class QwenClient:
    def __init__(self):
        base_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        local_dir = os.getenv("FOODY_VLM_MODEL_DIR", "").strip()

        if local_dir:
            ckpt_path = Path(local_dir)

            if ckpt_path.is_dir():
                final_path = ckpt_path / "final"
                if final_path.exists():
                    ckpt_path = final_path
                else:
                    checkpoints = sorted(
                        ckpt_path.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1
                    )
                    if checkpoints:
                        ckpt_path = checkpoints[-1]

            print(f"[INFO] Loading VLM from local checkpoint: {ckpt_path}")

            try:
                self.processor = AutoProcessor.from_pretrained(str(ckpt_path))
                print("[INFO] Processor loaded from checkpoint.")
            except Exception:
                self.processor = AutoProcessor.from_pretrained(base_id)
                print("[WARN] Processor not found in checkpoint. Loaded from base model.")

            adapter_cfg = ckpt_path / "adapter_config.json"
            is_lora = adapter_cfg.exists()

            if is_lora:
                print("[INFO] Detected LoRA adapter checkpoint. Loading base + adapter...")
                from peft import PeftModel

                base_model = AutoModelForVision2Seq.from_pretrained(
                    base_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

                self.model = PeftModel.from_pretrained(
                    base_model,
                    str(ckpt_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                print("[INFO] LoRA adapter loaded successfully.")
            else:
                print("[INFO] Detected full-model checkpoint. Loading directly...")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    str(ckpt_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                print("[INFO] Full model loaded successfully.")
        else:
            print("[INFO] Loading base model only (no checkpoint path provided).")
            self.processor = AutoProcessor.from_pretrained(base_id)
            self.model = AutoModelForVision2Seq.from_pretrained(
                base_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.model.eval()
        print("[INFO] VLM ready.")

    def _is_valid_food_name(self, s: str) -> bool:
        if not s:
            return False
        s = s.strip()
        if s in {"-", "—", "_", "?", "없음", "모름", "알수없음", "알 수 없음", "unknown"}:
            return False
        if not re.search(r"[가-힣]", s):
            return False
        return True

    def _generate_one(self, pil_image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

        text = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        text = text.split()[0].strip()
        text = re.sub(r"[\"'()\[\]{}<>]", "", text).strip()
        return text

    def _post_process_food_name(self, food_name: str) -> str:
        translations = {
            "omelette": "오믈렛",
            "omelet": "오믈렛",
            "eggroll": "계란말이",
            "egg roll": "계란말이",
            "koreaneggroll": "계란말이",
            "korean egg roll": "계란말이",
            "rolledegg": "계란말이",
            "rolled egg": "계란말이",
            "salad": "샐러드",
            "rice": "밥",
            "kimchi": "김치",
            "kimbap": "김밥",
            "ramen": "라면",
        }

        lower_name = food_name.lower().strip()
        if lower_name in translations:
            return translations[lower_name]
        for eng, kor in translations.items():
            if eng in lower_name:
                return kor
        return food_name

    def predict_food_name(self, pil_image: Image.Image) -> str:
        prompt1 = (
            "너는 한국 음식 이미지 분류기다.\n"
            "이미지에서 '가장 중심이 되는 음식 1개'의 이름만 한국어로 출력해라.\n\n"
            "출력 규칙(매우 중요):\n"
            "1) 한국어 음식명만 출력\n"
            "2) 조사/문장/설명 금지 (예: '입니다', '같아요', '.' 금지)\n"
            "3) 따옴표/괄호/슬래시/이모지 금지\n"
            "4) 공백 없이 음식명만 출력 (최대 12자)\n\n"
            "예시(정답 형식):\n"
            "김밥\n"
            "계란말이\n"
            "떡볶이\n\n"
            "혼동 주의 규칙:\n"
            "- 계란말이: 달걀을 말아 네모/원통 형태, 단면이 말린 층\n"
            "- 오믈렛: 접힌 형태, 둥글고 납작함\n\n"
            "중요: 출력할 음식명이 없다고 판단되더라도 '-', '없음', '?'를 출력하지 말고\n"
            "가장 유사한 한국 음식명 1개를 반드시 출력하라.\n\n"
            "정답(음식명만):"
        )

        out1 = self._generate_one(pil_image, prompt1)
        out1 = self._post_process_food_name(out1)
        if self._is_valid_food_name(out1):
            return out1

        prompt2 = (
            "너는 한국 음식 이미지 분류기다.\n"
            "모르겠어도 '-', '?', '없음'을 출력하지 마라.\n"
            "가장 유사한 한국 음식명 1개를 반드시 출력하라.\n"
            "음식명만 출력(설명 금지).\n"
            "정답:"
        )

        out2 = self._generate_one(pil_image, prompt2)
        out2 = self._post_process_food_name(out2)
        if self._is_valid_food_name(out2):
            return out2

        print(f"[WARN] VLM invalid outputs: out1='{out1}', out2='{out2}' -> fallback='음식'")
        return "음식"

    def estimate_nutrition_llm(self, food_name: str) -> dict:
        prompt = f"""
당신은 전문 영양학자입니다.
"{food_name}" 음식의 100g 기준 영양성분을 추정하세요.

 반드시 아래 JSON 형식으로만 출력하세요.
 설명, 문장, 코드블록, 여분의 텍스트는 절대로 넣지 마세요.
 모든 수치는 number 로 출력하세요 (따옴표 금지)

예시 출력:
{{
  "standard": "100g",
  "kcal": 154,
  "carb": 3.2,
  "protein": 11.2,
  "fat": 10.1,
  "sugar": 1.1,
  "natrium": 250
}}
        """

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=200)

        text = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        try:
            json_block = re.search(r"\{.*\}", text, flags=re.S).group(0)
            data = json.loads(json_block)
        except Exception:
            data = {
                "standard": "100g",
                "kcal": 0,
                "carb": 0,
                "protein": 0,
                "fat": 0,
                "sugar": 0,
                "natrium": 0,
            }

        return data


qwen = QwenClient()

# Chroma RAG
# - Chroma에는 "검색용 텍스트 + code"만 저장
# - 영양값은 항상 RDB에서 가져옴 (source of truth)
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_DIR = os.getenv("FOODY_CHROMA_DIR", "./chroma_foods")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

ko_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=os.getenv("FOODY_EMBED_MODEL", "jhgan/ko-sroberta-multitask")
)

food_collection = chroma_client.get_or_create_collection(
    name="food_nutrition",
    embedding_function=ko_embedding,
)


def build_chroma_from_db():
    """
    서버 시작 시 foods 테이블 내용을 Chroma에 인덱싱.
    ✅ Chroma에는 중복 영양값 저장 X (code + name/standard 정도만)
    """
    count = food_collection.count()
    if count > 0:
        print(f"[INFO] Existing Chroma collection already has {count} items. Skip building.")
        return

    print("[INFO] Building Chroma index from foods table (lightweight metadata)...")

    db = SessionLocal()
    try:
        foods: List[Foods] = db.query(Foods.code, Foods.name, Foods.standard).all()
        total = len(foods)
        print(f"[INFO] Loaded {total} foods from DB.")
    finally:
        try:
            db.close()
        except Exception as e:
            print(f"[WARN] Failed to close DB session cleanly: {e}")

    if total == 0:
        print("[INFO] No foods found in DB to index.")
        return

    batch_size = 256
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = foods[start:end]

        ids, docs, metas = [], [], []
        for code, name, standard in batch:
            ids.append(code)
            # 검색용 문서(이름 기반 유사검색) + 단위(선택)
            docs.append(f"{name} {standard}")
            metas.append({"code": code, "name": name, "standard": standard})

        food_collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f"[INFO] Indexed {end}/{total} foods into Chroma...")

    print("[INFO] Finished building Chroma index.")


def _fetch_foods_by_codes(db: Session, codes: List[str]) -> Dict[str, Foods]:
    """codes 목록을 RDB에서 조회해서 {code: Foods}로 반환"""
    if not codes:
        return {}
    rows = db.query(Foods).filter(Foods.code.in_(codes)).all()
    return {r.code: r for r in rows}


def rag_pick_codes(
    query_text: str,
    top_k: int = 5,
    hard_threshold: float = 1.0,
    soft_threshold: float = 0.75,
) -> Optional[List[Tuple[str, float]]]:
    """
    Chroma에서 유사 검색 결과를 code + distance로 반환.
    - best_distance > hard_threshold -> None (RAG 폐기)
    - dist <= soft_threshold 후보들만 리턴 (없으면 Top-1만)
    """
    result = food_collection.query(query_texts=[query_text], n_results=top_k)
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    documents = result.get("documents", [[]])[0]

    if not metadatas:
        print(f"[WARN] No RAG results found for '{query_text}'")
        return None

    print(f"[DEBUG] RAG Search Results for '{query_text}':")
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"  [{i+1}] {doc} | code={meta.get('code')} | distance={dist:.4f}")

    best_distance = distances[0]
    if best_distance > hard_threshold:
        print(f"[WARN] Best match distance ({best_distance:.4f}) > hard_threshold ({hard_threshold}).")
        print("[WARN] Discarding RAG result -> fallback to LLM.")
        return None

    candidates: List[Tuple[str, float]] = []
    for meta, dist in zip(metadatas, distances):
        code = meta.get("code")
        if not code:
            continue
        if dist <= soft_threshold:
            candidates.append((code, float(dist)))

    if not candidates:
        # soft_threshold 안쪽 후보가 없으면 Top-1 code만
        top1_code = metadatas[0].get("code")
        if not top1_code:
            return None
        print(f"[INFO] No candidates within soft_threshold ({soft_threshold}). Using Top-1 only.")
        return [(top1_code, float(distances[0]))]

    print(f"[INFO] Candidate codes within soft_threshold: {len(candidates)}")
    return candidates


def rag_estimate_nutrition_from_rdb(
    db: Session,
    food_name: str,
    top_k: int = 5,
    hard_threshold: float = 1.0,
    soft_threshold: float = 0.75,
    eps: float = 1e-6
) -> Optional[dict]:
    """
    Chroma -> (code, distance) 후보 얻기
    영양값은 RDB에서 code로 조회
    inverse-distance 가중 평균
    """
    candidates = rag_pick_codes(
        query_text=food_name,
        top_k=top_k,
        hard_threshold=hard_threshold,
        soft_threshold=soft_threshold,
    )
    if not candidates:
        return None

    codes = [c for c, _ in candidates]
    code_to_food = _fetch_foods_by_codes(db, codes)

    # RDB에 실제로 존재하는 것만 남김
    valid: List[Tuple[Foods, float]] = []
    for code, dist in candidates:
        f = code_to_food.get(code)
        if f:
            valid.append((f, dist))

    if not valid:
        print("[WARN] RAG returned codes but none found in RDB.")
        return None

    # Top-1 기준 standard를 따라가되, 필요하면 여기서 통일 규칙 적용 가능
    best_food = valid[0][0]
    best_dist = valid[0][1]

    # hard threshold는 이미 통과했지만, 혹시 DB쪽만 남기면서 이상해질 수도 있으니 로그
    print(f"[INFO] Weighted RAG from RDB (best_code={best_food.code}, best_distance={best_dist:.4f}, "
          f"used_candidates={len(valid)}/{len(candidates)})")

    weights = [1.0 / (d + eps) for _, d in valid]
    wsum = sum(weights)

    def wavg(attr: str) -> float:
        s = 0.0
        for (food, _), w in zip(valid, weights):
            s += float(getattr(food, attr)) * w
        return s / wsum if wsum > 0 else float(getattr(best_food, attr))

    return {
        "standard": best_food.standard,
        "kcal": wavg("kcal"),
        "carb": wavg("carb"),
        "protein": wavg("protein"),
        "fat": wavg("fat"),
        "sugar": wavg("sugar"),
        "natrium": wavg("natrium"),
        # 디버깅/추적 필요하면 code도 같이 내려도 됨(지금 response 모델엔 없음)
        # "code": best_food.code,
    }


# FastAPI
app = FastAPI(title="Foody - Qwen2.5-VL Analyzer API")


@app.on_event("startup")
def on_startup():
    build_chroma_from_db()


@app.post("/api/vlm/food", response_model=FoodResponse)
async def predict_food(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 1) 이미지 체크
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드 가능합니다.")

    # 2) 이미지 로드
    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 열 수 없습니다.")

    # 3) 음식 이름 추론 (VLM)
    print(f"\n{'='*80}")
    food_name = qwen.predict_food_name(pil_image)
    normalized_name = food_name.replace(" ", "")
    print(f"[INFO] 🔍 VLM Prediction: '{food_name}' (normalized: '{normalized_name}')")

    # 4) RDB exact match (공백 제거 비교)
    food = (
        db.query(Foods)
        .filter(func.replace(Foods.name, " ", "") == normalized_name)
        .first()
    )

    # 5) 영양 성분 결정 로직 (리팩터링)
    if food:
        print(f"[INFO] Found exact match in RDB: '{food.name}' (code: {food.code})")
        response_name = food.name
        est = {
            "standard": food.standard,
            "kcal": float(food.kcal),
            "carb": float(food.carb),
            "protein": float(food.protein),
            "fat": float(food.fat),
            "sugar": float(food.sugar),
            "natrium": float(food.natrium),
        }
    else:
        # RAG: Chroma는 "code 후보"만 주고, 영양값은 RDB에서 가져와 평균
        print(f"[INFO] No exact match in RDB for '{normalized_name}'")
        print(f"[INFO] Starting RAG search (codes) for '{food_name}'...")

        response_name = food_name
        est = rag_estimate_nutrition_from_rdb(
            db=db,
            food_name=food_name,
            top_k=5,
            hard_threshold=1.0,
            soft_threshold=0.75,
        )

        if est:
            print("[INFO] RAG->RDB nutrition success")
        else:
            print(f"[INFO] RAG failed, using LLM estimation for '{food_name}'")
            est = qwen.estimate_nutrition_llm(food_name)

    # 최종 결과 로깅
    print(f"[INFO] Final Response:")
    print(f"name='{response_name}', standard='{est.get('standard')}'")
    print(f"kcal={round2(est.get('kcal', 0))}, carb={round2(est.get('carb', 0))}g")
    print(f"protein={round2(est.get('protein', 0))}g, fat={round2(est.get('fat', 0))}g")
    print(f"sugar={round2(est.get('sugar', 0))}g, natrium={round2(est.get('natrium', 0))}mg")
    print(f"{'='*80}\n")

    return FoodResponse(
        name=response_name,
        standard=est.get("standard", "100g"),
        kcal=round2(est.get("kcal", 0)),
        carb=round2(est.get("carb", 0)),
        protein=round2(est.get("protein", 0)),
        fat=round2(est.get("fat", 0)),
        sugar=round2(est.get("sugar", 0)),
        natrium=round2(est.get("natrium", 0)),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
