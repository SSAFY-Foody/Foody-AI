
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

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

chromaclient = chromadb.PersistentClient(path="./chroma_foods")
embedding_function = SentenceTransformerEmbeddingFunction(

    model_name="sentence-transformers/all-mpnet-base-v2" # 다국어 지원 임베딩
)

foods_collection = chroma_client.get_or_create_collection(
    name = "foods",
    embedding_function= embedding_function,
)

# Chroma 클라이언트 및 컬렉션 전역 변수
chroma_client = chromadb.PersistentClient(path="./chroma_foods")  # 원하는 경로
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"  # 다국어 지원 임베딩
)

foods_collection = chroma_client.get_or_create_collection(
    name="foods",
    embedding_function=embedding_function,
)

from sqlalchemy.orm import Session

def build_chroma_index(db: Session):
    """
    MySQL foods 테이블 전체를 Chroma에 upsert하는 함수.
    - document: 음식 이름 + 카테고리 + 설명성 텍스트
    - metadata: code, name, category, standard, 영양성분 전부
    """
    # 이미 데이터가 있다면 skip
    count = foods_collection.count()
    if count > 0:
        print(f"[INFO] Chroma collection already has {count} items. Skip building.")
        return

    print("[INFO] Building Chroma index from MySQL foods table...")

    foods = db.query(Foods).all()
    if not foods:
        print("[WARN] foods table is empty. Nothing to index.")
        return

    ids = []
    documents = []
    metadatas = []

    for food in foods:
        ids.append(food.code)
        # 검색용 document (텍스트 기반)
        doc = f"{food.name} ({food.category}) 기준량 {food.standard}"
        documents.append(doc)

        meta = {
            "code": food.code,
            "name": food.name,
            "category": food.category,
            "standard": food.standard,
            "kcal": float(food.kcal),
            "carb_g": float(food.carb_g),
            "protein_g": float(food.protein_g),
            "fat_g": float(food.fat_g),
            "sugar_g": float(food.sugar_g),
            "natrium_g": float(food.natrium_g),
        }
        metadatas.append(meta)

    # 한 번에 업서트
    foods_collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"[INFO] Chroma index built. Inserted {len(ids)} items.")


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
    standard: str
    kcal: float
    carb_g: float
    protein_g: float
    fat_g: float
    sugar_g: float
    natrium_g: float


# ai 모델 로드
class QwenClient:
    # ... 기존 __init__, predict_food_name 유지

    def estimate_nutrition_with_context(self, food_name: str, context: str) -> dict:
        """
        RAG: Chroma에서 찾은 유사 음식들의 영양성분(context)을 기반으로
        Qwen에게 "{food_name}"의 영양성분을 JSON으로 추론하게 함.
        """
        prompt = f"""
당신은 전문 영양학자입니다.

아래는 이미 DB에 존재하는 유사 음식들과 그 영양성분입니다:

{context}

위 음식들과 가장 비슷한 "{food_name}" 음식의
100g 기준 영양성분을 추정하세요.

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
            json_block = re.search(r"\{.*\}", text, flags=re.S).group(0)
            data = json.loads(json_block)
        except:
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



# class QwenClient:

#     def __init__(self):
#         print("[INFO] Loading Qwen2.5-VL-3B-Instruct...")
#         self.processor = AutoProcessor.from_pretrained(
#             "Qwen/Qwen2.5-VL-3B-Instruct"
#         )
#         self.model = AutoModelForVision2Seq.from_pretrained(
#             "Qwen/Qwen2.5-VL-3B-Instruct",
#             torch_dtype=torch.float16,
#             device_map="auto"  # GPU 안 되면 여기 cpu로 바꿔줘야 함
#         )

#     # 음식 이미지를 보고 음식 명 추론
#     def predict_food_name(self, pil_image: Image.Image) -> str:

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": pil_image},
#                     {
#                         "type": "text",
#                         "text": (
#                             "당신은 한국 음식 이미지 분류기입니다.\n"
#                             "절대로 설명하지 말고, 절대로 영어를 섞지 말고,\n"
#                             "음식 이름을 한국어 한 단어로만 출력하세요.\n\n"
#                             "예시:\n"
#                             "김밥\n"
#                             "라면\n"
#                             "불고기\n"
#                             "오믈렛\n"
#                             "샐러드\n\n"
#                             "정확히 한 단어만 출력하세요."
#                         )
#                     }
#                 ],
#             }
#         ]

#         inputs = self.processor.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt",
#         ).to(self.model.device)

#         output = self.model.generate(
#             **inputs,
#             max_new_tokens=20,
#             do_sample=False
#         )

#         text = self.processor.decode(
#             output[0][inputs["input_ids"].shape[-1]:],
#             skip_special_tokens=True
#         ).strip()

#         # "김밥입니다." 같은 경우 대비 → 첫 단어만 사용
#         return text.split()[0]

#    # DB 에 없는 경우 추론
#     def estimate_nutrition(self, food_name: str) -> dict:

#         prompt = f"""
# 당신은 전문 영양학자입니다.
# "{food_name}" 음식의 100g 기준 영양성분을 추정하세요.

# ❗ 반드시 아래 JSON 형식으로만 출력하세요.
# ❗ 설명, 문장, 코드블록, 여분의 텍스트는 절대로 넣지 마세요.

# 예시 출력:
# {{
#   "standard": "100g",
#   "kcal": 154,
#   "carb_g": 3.2,
#   "protein_g": 11.2,
#   "fat_g": 10.1,
#   "sugar_g": 1.1,
#   "natrium_g": 250
# }}
#         """

#         messages = [
#             {"role": "user", "content": [{"type": "text", "text": prompt}]}
#         ]

#         inputs = self.processor.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt",
#         ).to(self.model.device)

#         output = self.model.generate(
#             **inputs,
#             max_new_tokens=200
#         )

#         text = self.processor.decode(
#             output[0][inputs["input_ids"].shape[-1]:],
#             skip_special_tokens=True
#         ).strip()

#         import json, re
#         try:
#             # 혹시 모델이 앞뒤에 # & * 같은 이상한 문자 붙이는 경우 안에 있는 값만 추출하기
#             json_block = re.search(r"\{.*\}", text, flags=re.S).group(0)
#             data = json.loads(json_block)
#         except: # 오류 나면 기본값 반환하기
#             data = {
#                 "standard": "100g",
#                 "kcal": 0,
#                 "carb_g": 0,
#                 "protein_g": 0,
#                 "fat_g": 0,
#                 "sugar_g": 0,
#                 "natrium_g": 0
#             }

#         return data


qwen = QwenClient()

# FastAPI 앱 생성
# 실제 로직이 돌아가는 부분
app = FastAPI(title="Foody - Qwen2.5-VL Analyzer API")

@app.on_event("startup")
def on_startup():
    # 앱 시작할때, Chroma 컬렉션에 아무것도 없으면 MySQL 에서 빌드
    with SessionLocal() as db:
        build_chroma_index(db)



@app.post("/food", response_model=FoodResponse)
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

    # 3) 음식 이름 추론
    food_name = qwen.predict_food_name(pil_image)
    print("[INFO] Predicted food name:", food_name)

    # --- (선택) 가장 먼저, 정확히 그 이름이 DB에 있으면 그냥 그걸 쓰고 싶다면 ---
    # normalized_name = food_name.replace(" ", "")
    # food = (
    #     db.query(Foods)
    #     .filter(func.replace(Foods.name, " ", "") == normalized_name)
    #     .first()
    # )
    # if food:
    #     print("[INFO] Exact food found in DB:", food.name)
    #     return FoodResponse(
    #         name=food.name,
    #         category=food.category,
    #         standard=food.standard,
    #         kcal=food.kcal,
    #         carb_g=food.carb_g,
    #         protein_g=food.protein_g,
    #         fat_g=food.fat_g,
    #         sugar_g=food.sugar_g,
    #         natrium_g=food.natrium_g,
    #     )

    # 4) Chroma RAG 검색
    try:
        rag_k = 5  # 유사 음식 몇 개 가져올지
        query_result = foods_collection.query(
            query_texts=[food_name],
            n_results=rag_k,
        )
    except Exception as e:
        print("[ERROR] Chroma query failed:", e)
        query_result = None

    # query_result 구조 예:
    # {
    #   "ids": [[...]],
    #   "documents": [[...]],
    #   "metadatas": [[{...}, {...}, ...]],
    #   "distances": [[...]]
    # }

    metadatas = []
    if query_result and query_result.get("metadatas"):
        metadatas = query_result["metadatas"][0]  # 첫 번째 query에 대한 결과
    else:
        print("[WARN] No results from Chroma. Fallback to no-context estimation.")

    # 5) 컨텍스트 텍스트 만들기
    context_lines = []
    for m in metadatas:
        context_lines.append(
            f"- 이름: {m.get('name')}, 카테고리: {m.get('category')}, 기준량: {m.get('standard')}, "
            f"칼로리: {m.get('kcal')}, 탄수화물: {m.get('carb_g')}, 단백질: {m.get('protein_g')}, "
            f"지방: {m.get('fat_g')}, 당류: {m.get('sugar_g')}, 나트륨: {m.get('natrium_g')}"
        )
    context_text = "\n".join(context_lines)

    # 6) Qwen으로 영양소 추론 (RAG 컨텍스트 사용)
    if context_text:
        est = qwen.estimate_nutrition_with_context(food_name, context_text)
        print("[INFO] Estimated nutrition with RAG for:", food_name)
    else:
        # Chroma가 비었거나 오류 났을 때 fallback
        est = qwen.estimate_nutrition(food_name)
        print("[INFO] Estimated nutrition WITHOUT context for:", food_name)

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
