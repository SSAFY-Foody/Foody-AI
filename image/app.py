
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from sqlalchemy import create_engine, Column, String, Float, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 내용 읽어오기

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


#기존 FOOD 테이블 매핑

class Foods(Base):
    __tablename__ = "foods" 

    code      = Column(String(45), primary_key=True)   # PK
    name      = Column(String(30), nullable=False)     # 음식 이름
    category  = Column(String(30), nullable=False)     # 카테고리
    standard  = Column(String(10), nullable=False)     # 기준량
    kcal      = Column(Float, nullable=False, default=0)
    carb_g    = Column(Float, nullable=False, default=0)
    protein_g = Column(Float, nullable=False, default=0)
    fat_g     = Column(Float, nullable=False, default=0)
    sugar_g   = Column(Float, nullable=False, default=0)
    natrium_g = Column(Float, nullable=False, default=0)


#응답 모델(JSON) 생성

class FoodResponse(BaseModel):
    name: str
    category: str
    standard: str
    kcal: float
    carb_g: float
    protein_g: float
    fat_g: float
    sugar_g: float
    natrium_g: float


# ai 모델 로드

class QwenClient:

    def __init__(self):
        print("[INFO] Loading Qwen2.5-VL-3B-Instruct...")
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"  # GPU 안 되면 여기 cpu로 바꿔줘야 함
        )

    # 음식 이미지를 보고 음식 명 추론
    def predict_food_name(self, pil_image: Image.Image) -> str:

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {
                        "type": "text",
                        "text": (
                            "당신은 한국 음식 이미지 분류기입니다.\n"
                            "절대로 설명하지 말고, 절대로 영어를 섞지 말고,\n"
                            "음식 이름을 한국어 한 단어로만 출력하세요.\n\n"
                            "예시:\n"
                            "김밥\n"
                            "라면\n"
                            "불고기\n"
                            "오믈렛\n"
                            "샐러드\n\n"
                            "정확히 한 단어만 출력하세요."
                        )
                    }
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
            do_sample=False
        )

        text = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        # "김밥입니다." 같은 경우 대비 → 첫 단어만 사용
        return text.split()[0]

   # DB 에 없는 경우 추론
    def estimate_nutrition(self, food_name: str) -> dict:

        prompt = f"""
당신은 전문 영양학자입니다.
"{food_name}" 음식의 100g 기준 영양성분을 추정하세요.

❗ 반드시 아래 JSON 형식으로만 출력하세요.
❗ 설명, 문장, 코드블록, 여분의 텍스트는 절대로 넣지 마세요.

예시 출력:
{{
  "category": "한식",
  "standard": "100g",
  "kcal": 154,
  "carb_g": 3.2,
  "protein_g": 11.2,
  "fat_g": 10.1,
  "sugar_g": 1.1,
  "natrium_g": 250
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
            max_new_tokens=200
        )

        text = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        import json, re
        try:
            # 혹시 모델이 앞뒤에 # & * 같은 이상한 문자 붙이는 경우 안에 있는 값만 추출하기
            json_block = re.search(r"\{.*\}", text, flags=re.S).group(0)
            data = json.loads(json_block)
        except: # 오류 나면 기본값 반환하기
            data = {
                "category": "기타",
                "standard": "100g",
                "kcal": 0,
                "carb_g": 0,
                "protein_g": 0,
                "fat_g": 0,
                "sugar_g": 0,
                "natrium_g": 0
            }

        return data


qwen = QwenClient()


# FastAPI 앱 생성
# 실제 로직이 돌아가는 부분
app = FastAPI(title="Foody - Qwen2.5-VL Analyzer API")


@app.post("/predict", response_model=FoodResponse)
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

    # 1) 음식 이름 추론
    food_name = qwen.predict_food_name(pil_image)
    normalized_name = food_name.replace(" ", "")



    # 4) DB 조회 - 부분 일치 기반
    # 공백 제거 후에 like 구문 통해서 일부 일치하는 값 찾기
    food = (
        db.query(Foods)
        .filter(func.replace(Foods.name, " ", "").like(f"%{normalized_name}%"))
        .order_by(func.length(Foods.name))
        .first()
    )

    # 5) DB에 있으면 → DB 값 그대로 응답
    if food:
        print("[INFO] Food found in DB:", food.name)
        return FoodResponse(
            name=food.name,
            category=food.category,
            standard=food.standard,
            kcal=food.kcal,
            carb_g=food.carb_g,
            protein_g=food.protein_g,
            fat_g=food.fat_g,
            sugar_g=food.sugar_g,
            natrium_g=food.natrium_g,
        )

    # 6) DB에 없으면  Qwen으로 영양소 추론 (DB INSERT 없음)
    est = qwen.estimate_nutrition(food_name)
    print("[INFO] Food not found in DB. Estimated:", food_name)
    return FoodResponse(
        name=food_name,
        category=est["category"],
        standard=est["standard"],
        kcal=est["kcal"],
        carb_g=est["carb_g"],
        protein_g=est["protein_g"],
        fat_g=est["fat_g"],
        sugar_g=est["sugar_g"],
        natrium_g=est["natrium_g"],
    )
