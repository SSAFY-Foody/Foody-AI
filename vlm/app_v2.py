# app_v2.py
import io
import os
import sys
import json
import re
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from PIL import Image
import numpy as np

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from sqlalchemy import create_engine, Column, String, Float, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from dotenv import load_dotenv

# =========================
# YOLOv3 관련 import
# =========================
# YOLOv3 디렉토리를 Python path에 추가
yolov3_path = Path(__file__).parent / "yolov3" / "yolov3"
sys.path.insert(0, str(yolov3_path))

from models import Darknet
from utils.utils import non_max_suppression, scale_coords, load_classes
from utils.datasets import letterbox

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
    confidence: Optional[float] = None  # YOLOv3 confidence


class MultipleFoodResponse(BaseModel):
    """단일/다중 통합 응답: foods 길이가 1이면 단일처럼 사용 가능"""
    foods: List[FoodResponse]
    total_count: int


# =========================
# 유틸: 소수점 둘째자리 반올림
# =========================
def round2(value: float) -> float:
    try:
        return round(float(value), 2)
    except Exception:
        return 0.0


# =========================
# YOLOv3 Food Detector
# =========================
class YOLOv3FoodDetector:
    def __init__(
        self,
        cfg_path: str = "yolov3/yolov3/cfg/yolov3-spp-403cls.cfg",
        weights_path: str = "yolov3/yolov3/weights/best_403food_e200b150v2.pt",
        names_path: str = "yolov3/yolov3/data/403food.names",
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.5,
        img_size: int = 416,
    ):
        """
        YOLOv3 기반 음식 객체 탐지 클래스
        - class_name은 보통 음식 코드 문자열(예: 01011001, 107071 등)
        """
        print("[INFO] Loading YOLOv3 Food Detector...")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        base_path = Path(__file__).parent
        cfg_path = str(base_path / cfg_path)
        weights_path = str(base_path / weights_path)
        names_path = str(base_path / names_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        self.model = Darknet(cfg_path, (img_size, img_size))

        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            print(f"[INFO] Loaded YOLOv3 weights from {weights_path}")
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        self.model.to(self.device).eval()

        if os.path.exists(names_path):
            self.class_names = load_classes(names_path)
            print(f"[INFO] Loaded {len(self.class_names)} food codes")
        else:
            print(f"[WARN] Class names file not found: {names_path}")
            self.class_names = [f"class_{i}" for i in range(403)]

    def preprocess_image(self, pil_image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        이미지를 YOLOv3 입력 형식으로 전처리
        Returns: (tensor[B,3,H,W], original_image_array(H,W,3))
        """
        img0 = np.array(pil_image)  # RGB

        # Letterbox resize (returns resized image)
        img = letterbox(img0, new_shape=self.img_size)[0]

        # ⚠️ YOLOv3 레포에 따라 BGR 기대하는 경우가 있어 아래 변환을 유지
        # HWC(RGB) -> CHW(BGR)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, img0

    def detect_foods(self, pil_image: Image.Image) -> List[Dict]:
        """
        Returns: list of dict
        - bbox: [x1,y1,x2,y2]
        - confidence: float
        - class_id: int
        - class_name: str (food code)
        - cropped_image: PIL.Image
        """
        img, img0 = self.preprocess_image(pil_image)

        with torch.no_grad():
            pred = self.model(img)[0]

        pred = non_max_suppression(
            pred,
            self.conf_threshold,
            self.iou_threshold,
            multi_label=False,
            classes=None,
            agnostic=False,
        )

        detections: List[Dict] = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                pil_img0 = Image.fromarray(img0)
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = [int(coord.item()) for coord in xyxy]
                    confidence = float(conf.item())
                    class_id = int(cls.item())
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

                    cropped = pil_img0.crop((x1, y1, x2, y2))

                    detections.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name,
                            "cropped_image": cropped,
                        }
                    )
                    print(f"[DETECT] YOLO code: {class_name} (conf={confidence:.3f}) at [{x1},{y1},{x2},{y2}]")

        print(f"[INFO] YOLOv3 detected {len(detections)} object(s)")
        return detections


# =========================
# Qwen VLM 클라이언트
# =========================
class QwenClient:
    def __init__(self):
        print("[INFO] Loading Qwen2.5-VL-3B-Instruct (Base Model)...")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        self.model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        print("[INFO] Base Qwen model loaded successfully!")

    def _post_process_food_name(self, food_name: str) -> str:
        translations = {
            "omelette": "오믈렛",
            "omelet": "오믈렛",
            "eggroll": "계란말이",
            "egg roll": "계란말이",
            "korean egg roll": "계란말이",
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

    def _is_valid_food_name(self, s: str) -> bool:
        if not s:
            return False
        s = s.strip()
        if s in {"-", "—", "_", "?", "없음", "모름", "알수없음", "알 수 없음", "unknown"}:
            return False
        if not re.search(r"[가-힣]", s):
            return False
        return True

    def predict_food_name(self, pil_image: Image.Image, yolo_hint: Optional[str] = None) -> str:
        """
        crop 이미지를 보고 '한글 음식명' 1개를 출력하도록 유도
        yolo_hint는 코드(예: 107071)라도 '참고 힌트'로만 사용
        """
        hint_text = ""
        if yolo_hint:
            hint_text = f"\n참고 힌트(코드일 수 있음): {yolo_hint}"

        prompt = (
            "너는 한국 음식 이미지 분류기다.\n"
            "이미지에서 보이는 음식 1개의 이름만 '한국어'로 출력해라.\n\n"
            "출력 규칙(매우 중요):\n"
            "1) 한국어 음식명만 출력\n"
            "2) 조사/문장/설명 금지 (예: '입니다', '같아요', '.', '사진 속' 금지)\n"
            "3) 따옴표/괄호/슬래시/이모지 금지\n"
            "4) 공백 없이 음식명만 출력 (최대 12자)\n"
            "5) '-' '?' '없음' 출력 금지. 모르겠어도 가장 유사한 음식명 1개 출력\n"
            f"{hint_text}\n\n"
            "정답(음식명만):"
        )

        messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt}]}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)

        text = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        # 문장/여분 제거
        food_name = text.splitlines()[0].strip()
        food_name = food_name.split()[0].strip()
        food_name = re.sub(r"[\"'()\[\]{}<>]", "", food_name).strip()
        food_name = self._post_process_food_name(food_name)

        if self._is_valid_food_name(food_name):
            return food_name

        # 재시도 프롬프트
        prompt2 = (
            "한국 음식명 1개만 한국어로 출력해라. 설명 금지.\n"
            "절대 '-', '?', '없음' 출력하지 마라.\n"
            "정답:"
        )
        messages2 = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt2}]}]
        inputs2 = self.processor.apply_chat_template(
            messages2,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        output2 = self.model.generate(**inputs2, max_new_tokens=20, do_sample=False)
        text2 = self.processor.decode(
            output2[0][inputs2["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        food_name2 = text2.splitlines()[0].strip()
        food_name2 = food_name2.split()[0].strip()
        food_name2 = re.sub(r"[\"'()\[\]{}<>]", "", food_name2).strip()
        food_name2 = self._post_process_food_name(food_name2)

        if self._is_valid_food_name(food_name2):
            return food_name2

        print(f"[WARN] VLM invalid outputs: '{food_name}' / '{food_name2}' -> fallback='음식'")
        return "음식"

    def estimate_nutrition_llm(self, food_name: str) -> dict:
        """순수 LLM 기반 영양 추론 (마지막 fallback)"""
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

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

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


# =========================
# 전역 객체 초기화
# =========================
yolo_detector = YOLOv3FoodDetector()
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

        ids, docs, metas = [], [], []
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


def rag_estimate_nutrition(food_name: str, top_k: int = 3) -> Optional[dict]:
    """Chroma로 유사 음식 영양정보 검색 후 평균"""
    result = food_collection.query(query_texts=[food_name], n_results=top_k)

    metadatas = result.get("metadatas", [[]])[0]
    if not metadatas:
        return None

    n = len(metadatas)
    sum_kcal = sum(float(m.get("kcal", 0.0)) for m in metadatas)
    sum_carb = sum(float(m.get("carb", 0.0)) for m in metadatas)
    sum_protein = sum(float(m.get("protein", 0.0)) for m in metadatas)
    sum_fat = sum(float(m.get("fat", 0.0)) for m in metadatas)
    sum_sugar = sum(float(m.get("sugar", 0.0)) for m in metadatas)
    sum_natrium = sum(float(m.get("natrium", 0.0)) for m in metadatas)

    best = metadatas[0]
    return {
        "standard": best.get("standard", "100g"),
        "kcal": sum_kcal / n,
        "carb": sum_carb / n,
        "protein": sum_protein / n,
        "fat": sum_fat / n,
        "sugar": sum_sugar / n,
        "natrium": sum_natrium / n,
    }


# =========================
# Nutrition Resolve (DB exact -> RAG -> LLM)
# =========================
def resolve_nutrition(db: Session, food_name: str) -> Tuple[str, dict]:
    """
    food_name(한글)을 기준으로:
    1) DB exact match -> DB 값
    2) 아니면 RAG
    3) 아니면 LLM fallback
    return: (response_name, est_dict)
    """
    normalized_name = food_name.replace(" ", "")

    food = (
        db.query(Foods)
        .filter(func.replace(Foods.name, " ", "") == normalized_name)
        .first()
    )

    if food:
        est = {
            "standard": food.standard,
            "kcal": float(food.kcal),
            "carb": float(food.carb),
            "protein": float(food.protein),
            "fat": float(food.fat),
            "sugar": float(food.sugar),
            "natrium": float(food.natrium),
        }
        return food.name, est

    est = rag_estimate_nutrition(food_name)
    if est:
        return food_name, est

    est = qwen.estimate_nutrition_llm(food_name)
    return food_name, est


# =========================
# FastAPI 앱 생성
# =========================
app = FastAPI(title="Foody - YOLOv3 -> Qwen -> Nutrition (single endpoint)")


@app.on_event("startup")
def on_startup():
    build_chroma_from_db()


# =========================
# 단일/다중 통합 Endpoint
# =========================
@app.post("/api/vlm/food", response_model=MultipleFoodResponse)
async def predict_food(image: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    단일/다중 통합 엔드포인트
    1) YOLOv3로 객체 탐지(개수 상관X)
    2) 각 탐지 crop 이미지를 Qwen에 넣어 한글 음식명 추출
    3) 음식명 기준으로 DB exact -> RAG -> LLM fallback 로 영양 추정
    4) foods 리스트로 반환 (단일이면 1개, 다중이면 여러개)
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드 가능합니다.")

    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 열 수 없습니다.")

    print("\n" + "=" * 80)
    print("[INFO] Pipeline: YOLOv3 detect -> Qwen name per crop -> Nutrition resolve")

    # 1) YOLO 탐지
    detections = yolo_detector.detect_foods(pil_image)

    # 2) YOLO가 못 잡으면 전체 이미지 1개 fallback
    if not detections:
        print("[WARN] YOLOv3 detected 0 objects. Fallback to full image as single item.")
        detections = [
            {
                "bbox": [0, 0, pil_image.size[0], pil_image.size[1]],
                "confidence": None,
                "class_id": -1,
                "class_name": None,
                "cropped_image": pil_image,
            }
        ]

    # 3) (선택) confidence 높은 순으로 처리 + 너무 많은 탐지 제한
    def conf_key(d: Dict) -> float:
        c = d.get("confidence")
        return -1.0 if c is None else float(c)

    detections = sorted(detections, key=conf_key, reverse=True)

    max_items = int(os.getenv("FOODY_MAX_FOODS", "8"))

    results: List[FoodResponse] = []
    seen = set()

    for det in detections[:max_items]:
        crop_img: Image.Image = det["cropped_image"]
        yolo_code = det.get("class_name")         # 코드(힌트)
        yolo_conf = det.get("confidence")         # float or None

        # 4) Qwen으로 한글 음식명 추출
        food_name = qwen.predict_food_name(crop_img, yolo_hint=yolo_code)
        key = food_name.replace(" ", "")

        # 중복 제거(같은 음식 여러 박스)
        if key in seen:
            print(f"[INFO] Skip duplicate name: {food_name}")
            continue
        seen.add(key)

        print(f"[INFO] YOLO_hint={yolo_code}, conf={yolo_conf}, Qwen_name='{food_name}'")

        # 5) 영양소 검색(기존 로직 수행)
        response_name, est = resolve_nutrition(db, food_name)

        results.append(
            FoodResponse(
                name=response_name,
                standard=est.get("standard", "100g"),
                kcal=round2(est.get("kcal", 0)),
                carb=round2(est.get("carb", 0)),
                protein=round2(est.get("protein", 0)),
                fat=round2(est.get("fat", 0)),
                sugar=round2(est.get("sugar", 0)),
                natrium=round2(est.get("natrium", 0)),
                confidence=float(yolo_conf) if isinstance(yolo_conf, (float, int)) else None,
            )
        )

    print(f"[INFO] Final foods count: {len(results)}")
    print("=" * 80 + "\n")

    return MultipleFoodResponse(foods=results, total_count=len(results))
