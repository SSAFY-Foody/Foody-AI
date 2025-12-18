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

    code     = Column(String(45), primary_key=True)
    name     = Column(String(30), nullable=False)
    standard = Column(String(10), nullable=False)
    kcal     = Column(Float, nullable=False, default=0)

    # 첫 번째 인자에 "실제 DB 컬럼 이름"을 넣어주면 됨
    carb     = Column("carb_g", Float, nullable=False, default=0)
    protein  = Column("protein_g", Float, nullable=False, default=0)
    fat      = Column("fat_g", Float, nullable=False, default=0)
    sugar    = Column("sugar_g", Float, nullable=False, default=0)
    natrium  = Column("natrium_g", Float, nullable=False, default=0)



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
# Qwen VLM 클라이언트
# =========================
class QwenClient:
    def __init__(self):
        print("[INFO] Loading Qwen2.5-VL-3B-Instruct...")
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        
        # 기본 모델 로드
        self.model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",  # 필요시 "cpu" 로 변경
        )
        
        # Fine-tuned LoRA 어댑터 로드 (있는 경우)
        import os
        lora_path = "./qwen_vlm_model"
        if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            print(f"[INFO] Loading fine-tuned LoRA adapter from {lora_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print("[INFO] Fine-tuned model loaded successfully!")
        else:
            print("[INFO] No fine-tuned model found. Using base model.")


    # 1) 음식 이미지를 보고 음식 명 추론
    def predict_food_name(self, pil_image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {
                        "type": "text",
                        "text": (
                            "한국 음식 이미지를 분류합니다. 음식 이름을 한국어로만 출력하세요.\n\n"
                            "중요: 절대로 영어 단어를 사용하지 마세요!\n"
                            "예: 'omelette' (X) → '오믈렛' (O)\n\n"
                            "한국 음식 예시:\n"
                            "- 계란말이: 계란으로 만든 네모난 한국식 계란 요리 (돌돌 말려있음)\n"
                            "- 김밥: 김으로 싼 밥\n"
                            "- 불고기: 양념한 고기 구이\n"
                            "- 떡볶이: 빨간 국물에 떡\n\n"
                            "주의사항:\n"
                            "- 계란말이 ≠ 오믈렛 (계란말이는 네모나고 말려있음)\n"
                            "- 반드시 한국어로만 답변\n"
                            "- 한 단어로만 출력\n\n"
                            "이미지의 음식 이름:"
                        ),
                    },
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
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        # "김밥입니다." 같은 경우 대비 → 첫 단어만 사용
        food_name = text.split()[0]
        
        # 후처리: 영어 단어를 한국어로 변환
        food_name = self._post_process_food_name(food_name)
        
        return food_name
    
    def _post_process_food_name(self, food_name: str) -> str:
        """영어 음식 이름을 한국어로 변환하고 검증"""
        
        # 영어 → 한국어 매핑
        translations = {
            "omelette": "오믈렛",
            "omelet": "오믈렛",
            "egg roll": "계란말이",
            "korean egg roll": "계란말이",
            "rolled egg": "계란말이",
            "salad": "샐러드",
            "rice": "밥",
            "kimchi": "김치",
            "kimbap": "김밥",
            "ramen": "라면",
        }
        
        # 소문자로 변환하여 비교
        lower_name = food_name.lower().strip()
        
        # 영어 단어 발견 시 한국어로 변환
        if lower_name in translations:
            return translations[lower_name]
        
        # 부분 일치 처리
        for eng, kor in translations.items():
            if eng in lower_name:
                return kor
        
        return food_name

    # 2) (Fallback) 순수 LLM 기반 영양 추론 (RAG 실패 시 마지막 보루)
    def estimate_nutrition_llm(self, food_name: str) -> dict:
        prompt = f"""
당신은 전문 영양학자입니다.
"{food_name}" 음식의 100g 기준 영양성분을 추정하세요.

❗ 반드시 아래 JSON 형식으로만 출력하세요.
❗ 설명, 문장, 코드블록, 여분의 텍스트는 절대로 넣지 마세요.

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

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
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
            max_new_tokens=200,
        )

        text = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1]:],
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

# Chroma 저장 경로 (.env 에서 FOODY_CHROMA_DIR 로 지정 가능)
CHROMA_DB_DIR = os.getenv("FOODY_CHROMA_DIR", "./chroma_foods")

# Chroma 클라이언트 (영구 저장)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# 한국어 지원 임베딩 함수 (원하는 SentenceTransformer 로 변경 가능)
ko_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=os.getenv(
        "FOODY_EMBED_MODEL",
        "jhgan/ko-sroberta-multitask"  # or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
)

# food_nutrition 컬렉션 생성/로드
food_collection = chroma_client.get_or_create_collection(
    name="food_nutrition",
    embedding_function=ko_embedding,
)

def build_chroma_from_db():
    """
    서버 시작 시, foods 테이블 내용을 Chroma에 인덱싱.
    이미 데이터가 있다면 스킵 (중복 방지).
    """
    count = food_collection.count()
    if count > 0:
        print(f"[INFO] Existing Chroma collection already has {count} items. Skip building.")
        return

    print("[INFO] Building Chroma index from foods table...")

    # 1) DB에서만 빨리 긁어오기
    db = SessionLocal()
    try:
        foods: List[Foods] = db.query(Foods).all()
        total = len(foods)
        print(f"[INFO] Loaded {total} foods from DB.")
    finally:
        # 연결이 이미 끊겼어도 여기서 또 에러 안 터지게 방어
        try:
            db.close()
        except Exception as e:
            print(f"[WARN] Failed to close DB session cleanly: {e}")

    if total == 0:
        print("[INFO] No foods found in DB to index.")
        return

    # 2) 세션은 이미 닫힌 상태에서 → 임베딩 + Chroma 인덱싱
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

        food_collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
        )
        print(f"[INFO] Indexed {end}/{total} foods into Chroma...")

    print("[INFO] Finished building Chroma index.")



def rag_estimate_nutrition(food_name: str, top_k: int = 3) -> Optional[dict]:
    """
    Chroma를 사용해서 유사한 음식들의 영양정보를 가져와
    Top-k 를 평균해서 통합 추정.
    """
    result = food_collection.query(
        query_texts=[food_name],
        n_results=top_k,
    )

    metadatas = result.get("metadatas", [[]])[0]
    if not metadatas:
        return None

    # kcal 등은 평균, standard는 제일 유사한(첫 번째) 것 사용
    n = len(metadatas)
    sum_kcal = sum(m.get("kcal", 0.0) for m in metadatas)
    sum_carb = sum(m.get("carb", 0.0) for m in metadatas)
    sum_protein = sum(m.get("protein", 0.0) for m in metadatas)
    sum_fat = sum(m.get("fat", 0.0) for m in metadatas)
    sum_sugar = sum(m.get("sugar", 0.0) for m in metadatas)
    sum_natrium = sum(m.get("natrium", 0.0) for m in metadatas)

    best = metadatas[0]  # 가장 유사한 것 하나
    estimated = {
        "standard": best.get("standard", "100g"),
        "kcal": sum_kcal / n,
        "carb": sum_carb / n,
        "protein": sum_protein / n,
        "fat": sum_fat / n,
        "sugar": sum_sugar / n,
        "natrium": sum_natrium / n,
    }
    return estimated



# =========================
# FastAPI 앱 생성
# =========================
app = FastAPI(title="Foody - Qwen2.5-VL Analyzer API")


@app.on_event("startup")
def on_startup():
    """
    서버 시작 시 1번만 실행:
    - foods 테이블 → Chroma 인덱싱
    """
    build_chroma_from_db()


@app.post("/api/vlm/food", response_model=FoodResponse)
async def predict_food(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 1) 이미지 체크
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드 가능합니다.")

    # 2) 이미지 로드
    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 열 수 없습니다.")

    # 3) 음식 이름 추론 (VLM)
    food_name = qwen.predict_food_name(pil_image)
    normalized_name = food_name.replace(" ", "")
    print(f"[INFO] Predicted food name: {food_name} (normalized: {normalized_name})")

    # 4) RDB에서 정확히 같은 이름이 있는지 먼저 체크
    food = (
        db.query(Foods)
        .filter(func.replace(Foods.name, " ", "") == normalized_name)
        .first()
    )

    # 응답에 보여줄 기준 이름 (DB에 있으면 DB 이름 기준, 없으면 모델 추론 그대로)
    if food:
        print(f"[INFO] Found exact match in DB: {food.name}")
        response_name = food.name
    else:
        response_name = food_name

    # 5) RAG로 Chroma에서 유사 음식들 검색 + 통합
    est = rag_estimate_nutrition(food_name)

    if est:
        print(f"[INFO] Estimated via Chroma RAG: {food_name}")
    else:
        # 6) Chroma에도 없으면
        if food:
            # (1) RDB에 값이 있으니 그걸 그대로 사용
            print(f"[WARN] Not found in Chroma, but found in DB. Using DB values for {food.name}")
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
            # (2) RDB도 없으면 완전 LLM 추론
            print(f"[WARN] Not found in DB or Chroma. Falling back to pure LLM for {food_name}")
            est = qwen.estimate_nutrition_llm(food_name)

    return FoodResponse(
        name=response_name,
        standard=est["standard"],
        kcal=est["kcal"],
        carb=est["carb"],
        protein=est["protein"],
        fat=est["fat"],
        sugar=est["sugar"],
        natrium=est["natrium"],
    )
