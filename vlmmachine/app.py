import io
import os
import json
import re
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from sqlalchemy import create_engine, Column, String, Float, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from dotenv import load_dotenv

# =========================
# .env 로드
# =========================
load_dotenv()

# =========================
# SQL 설정
# =========================
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


# =========================
# 기존 FOOD 테이블 매핑
# =========================
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


# =========================
# 응답 모델(JSON)
# =========================
class FoodResponse(BaseModel):
    name: str
    standard: str
    kcal: float
    carb: float
    protein: float
    fat: float
    sugar: float
    natrium: float


# =========================
# 유틸: 소수점 둘째자리 반올림
# =========================
def round2(value: float) -> float:
    return round(float(value), 2)


# =========================
# Qwen VLM 클라이언트 (Base model only)
# =========================
class QwenClient:
    def __init__(self):
        print("[INFO] Loading Qwen2.5-VL-3B-Instruct....")

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",  # GPU 없으면 자동으로 CPU
        )
        self.model.eval()
        print("[INFO] Base model loaded successfully (LoRA disabled).")

    def _is_valid_food_name(self, s: str) -> bool:
        if not s:
            return False
        s = s.strip()

        # 흔한 실패 토큰들
        if s in {"-", "—", "_", "?", "없음", "모름", "알수없음", "알 수 없음", "unknown"}:
            return False

        # 한글이 1글자 이상은 있어야 음식명으로 취급
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

        # 공백/문장 섞여 나올 때 대비
        text = text.split()[0].strip()

        # 따옴표/괄호 등 제거
        text = re.sub(r"[\"'()\[\]{}<>]", "", text).strip()

        return text

    def _post_process_food_name(self, food_name: str) -> str:
        """영어 음식 이름을 한국어로 변환하고 검증"""
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

        # 영어 단어 그대로 매칭
        if lower_name in translations:
            return translations[lower_name]

        # 부분 일치 처리
        for eng, kor in translations.items():
            if eng in lower_name:
                return kor

        return food_name

    # 1) 음식 이미지를 보고 음식 명 추론
    def predict_food_name(self, pil_image: Image.Image) -> str:
        # ✅ 1차 프롬프트 (네가 작성한 개선본을 "문장 연결" 제대로 되도록 정리)
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

        # ✅ 2차 프롬프트(재시도) - 더 강하게 "반드시 음식명"
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

        # ✅ 최후 fallback: 예전처럼 이상값 노출을 막기 위한 기본값
        print(f"[WARN] VLM invalid outputs: out1='{out1}', out2='{out2}' -> fallback='음식'")
        return "음식"

    # 2) (Fallback) 순수 LLM 기반 영양 추론 (RAG 실패 시 마지막 보루)
    def estimate_nutrition_llm(self, food_name: str) -> dict:
        prompt = f"""
당신은 전문 영양학자입니다.
"{food_name}" 음식의 100g 기준 영양성분을 추정하세요.

❗ 반드시 아래 JSON 형식으로만 출력하세요.
❗ 설명, 문장, 코드블록, 여분의 텍스트는 절대로 넣지 마세요.
❗ 모든 수치는 number 로 출력하세요 (따옴표 금지)

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

# =========================
# Chroma RAG 설정
# =========================
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
    """서버 시작 시 foods 테이블 내용을 Chroma에 인덱싱 (이미 있으면 스킵)"""
    count = food_collection.count()
    if count > 0:
        print(f"[INFO] Existing Chroma collection already has {count} items. Skip building.")
        return

    print("[INFO] Building Chroma index from foods table...")

    db = SessionLocal()
    try:
        foods: List[Foods] = db.query(Foods).all()
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

    batch_size = 128
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = foods[start:end]

        ids = []
        docs = []
        metas = []

        for f in batch:
            ids.append(f.code)
            docs.append(f"{f.name} {f.standard}")
            metas.append(
                {
                    "name": f.name,
                    "standard": f.standard,
                    "kcal": float(f.kcal),
                    "carb": float(f.carb),
                    "protein": float(f.protein),
                    "fat": float(f.fat),
                    "sugar": float(f.sugar),
                    "natrium": float(f.natrium),
                }
            )

        food_collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f"[INFO] Indexed {end}/{total} foods into Chroma...")

    print("[INFO] Finished building Chroma index.")


def rag_estimate_nutrition(food_name: str, top_k: int = 3, distance_threshold: float = 1.0) -> Optional[dict]:
    """Chroma로 유사 음식 영양정보 검색 및 추정 (Top-1 선택)"""
    result = food_collection.query(query_texts=[food_name], n_results=top_k)

    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    documents = result.get("documents", [[]])[0]

    if not metadatas:
        print(f"[WARN] No RAG results found for '{food_name}'")
        return None

    # 상세 로깅: 검색 결과 출력
    print(f"[DEBUG] RAG Search Results for '{food_name}':")
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"  [{i+1}] {doc} | distance={dist:.4f}")
        print(f"      kcal={meta.get('kcal')}, carb={meta.get('carb')}g, "
              f"protein={meta.get('protein')}g, fat={meta.get('fat')}g")

    # 가장 유사한 결과 선택 (Top-1)
    best_meta = metadatas[0]
    best_distance = distances[0]

    # 거리 임계값 체크
    if best_distance > distance_threshold:
        print(f"[WARN] Best match distance ({best_distance:.4f}) exceeds threshold ({distance_threshold})")
        print(f"[WARN] Result may be inaccurate for '{food_name}'")

    # Top-1만 사용 (평균 제거)
    result_data = {
        "standard": best_meta.get("standard", "100g"),
        "kcal": float(best_meta.get("kcal", 0)),
        "carb": float(best_meta.get("carb", 0)),
        "protein": float(best_meta.get("protein", 0)),
        "fat": float(best_meta.get("fat", 0)),
        "sugar": float(best_meta.get("sugar", 0)),
        "natrium": float(best_meta.get("natrium", 0)),
    }

    print(f"[INFO] Selected nutrition data: kcal={result_data['kcal']}, "
          f"carb={result_data['carb']}g, protein={result_data['protein']}g, fat={result_data['fat']}g")

    return result_data


# =========================
# FastAPI 앱 생성
# =========================
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

    # 5) 영양 성분 결정 로직
    if food:
        # ✅ DB에 정확히 일치하는 음식이 있으면 DB 데이터 사용
        print(f"[INFO] ✅ Found exact match in DB: '{food.name}' (code: {food.code})")
        print(f"[INFO] 💾 Using DB nutrition data directly (skipping RAG)")
        
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
        # DB에 없으면 RAG로 유사 음식 검색
        print(f"[INFO] No exact match in DB for '{normalized_name}'")
        print(f"[INFO] Starting RAG search for '{food_name}'...")
        
        response_name = food_name
        est = rag_estimate_nutrition(food_name)
        
        if est:
            print(f"[INFO] RAG search successful")
        else:
            # RAG도 실패하면 LLM으로 추정
            print(f"[INFO] RAG failed, using LLM estimation for '{food_name}'")
            est = qwen.estimate_nutrition_llm(food_name)

    # 최종 결과 로깅
    print(f"[INFO] 📊 Final Response:")
    print(f"       name='{response_name}', standard='{est.get('standard')}'")
    print(f"       kcal={round2(est.get('kcal', 0))}, carb={round2(est.get('carb', 0))}g")
    print(f"       protein={round2(est.get('protein', 0))}g, fat={round2(est.get('fat', 0))}g")
    print(f"       sugar={round2(est.get('sugar', 0))}g, natrium={round2(est.get('natrium', 0))}mg")
    print(f"{'='*80}\n")

    return FoodResponse(
        name=response_name,
        standard=est.get("standard", "100"),
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

