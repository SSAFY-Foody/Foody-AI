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

    code     = Column(String(45), primary_key=True)
    name     = Column(String(30), nullable=False)
    standard = Column(String(10), nullable=False)
    kcal     = Column(Float, nullable=False, default=0)

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
    confidence: Optional[float] = None  # YOLOv3 confidence 추가


class MultipleFoodResponse(BaseModel):
    """여러 음식이 감지된 경우의 응답"""
    foods: List[FoodResponse]
    total_count: int


# =========================
# YOLOv3 Food Detector
# =========================
class YOLOv3FoodDetector:
    def __init__(
        self,
        cfg_path: str = "yolov3/yolov3/cfg/yolov3-spp-403cls.cfg",
        weights_path: str = "yolov3/yolov3/weights/best_403food_e200b150v2.pt",
        names_path: str = "yolov3/yolov3/data/403food.names",
        conf_threshold: float = 0.2,  # 0.3 → 0.2로 낮춤 (더 많이 탐지)
        iou_threshold: float = 0.5,
        img_size: int = 416,
    ):
        """
        YOLOv3 기반 음식 객체 탐지 클래스
        
        Args:
            cfg_path: YOLOv3 config 파일 경로
            weights_path: 학습된 weights 파일 경로
            names_path: 클래스 이름 파일 경로
            conf_threshold: confidence threshold
            iou_threshold: IoU threshold for NMS
            img_size: 입력 이미지 크기
        """
        print("[INFO] Loading YOLOv3 Food Detector...")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # 상대 경로를 절대 경로로 변환
        base_path = Path(__file__).parent
        cfg_path = str(base_path / cfg_path)
        weights_path = str(base_path / weights_path)
        names_path = str(base_path / names_path)
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        # 모델 로드
        self.model = Darknet(cfg_path, (img_size, img_size))
        
        # Weights 로드
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            # strict=False: 학습 시 추가된 total_ops, total_params 같은 메타데이터 무시
            self.model.load_state_dict(checkpoint['model'], strict=False)
            print(f"[INFO] Loaded YOLOv3 weights from {weights_path}")
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        self.model.to(self.device).eval()
        
        # 클래스 이름 로드 (음식 코드: "01011001", "01012001" 등)
        if os.path.exists(names_path):
            self.class_names = load_classes(names_path)
            print(f"[INFO] Loaded {len(self.class_names)} food codes")
        else:
            print(f"[WARN] Class names file not found: {names_path}")
            self.class_names = [f"class_{i}" for i in range(403)]
    
    def preprocess_image(self, pil_image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        이미지를 YOLOv3 입력 형식으로 전처리
        
        Returns:
            (preprocessed_tensor, original_image_array)
        """
        # PIL to numpy
        img0 = np.array(pil_image)
        
        # Letterbox resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        
        # Convert BGR to RGB (OpenCV 형식에서 변환)
        # PIL은 이미 RGB이므로 변환 불필요
        
        # Normalize and convert to tensor
        img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW, RGB to BGR
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return img, img0
    
    def detect_foods(self, pil_image: Image.Image) -> List[Dict]:
        """
        이미지에서 음식 객체 탐지
        
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - class_id: int
            - class_name: str
            - cropped_image: PIL.Image
        """
        # 전처리
        img, img0 = self.preprocess_image(pil_image)
        
        # 추론
        with torch.no_grad():
            pred = self.model(img)[0]
        
        # NMS 적용
        pred = non_max_suppression(
            pred,
            self.conf_threshold,
            self.iou_threshold,
            multi_label=False,
            classes=None,
            agnostic=False
        )
        
        detections = []
        
        # 탐지 결과 처리
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # 좌표를 원본 이미지 크기로 변환
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                # 각 탐지 결과 처리
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = [int(coord.item()) for coord in xyxy]
                    confidence = conf.item()
                    class_id = int(cls.item())
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # 이미지 크롭
                    pil_img0 = Image.fromarray(img0)
                    cropped = pil_img0.crop((x1, y1, x2, y2))
                    
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,  # 음식 코드
                        "cropped_image": cropped
                    })
                    
                    print(f"[DETECT] Food code: {class_name} (conf={confidence:.3f}) at [{x1},{y1},{x2},{y2}]")
        
        print(f"[INFO] YOLOv3 detected {len(detections)} food(s)")
        return detections


# =========================
# Qwen VLM 클라이언트
# =========================
class QwenClient:
    def __init__(self):
        print("[INFO] Loading Qwen2.5-VL-3B-Instruct (Base Model)...")
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        
        # Base 모델만 로드 (Fine-tuned 모델 사용 안 함)
        self.model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("[INFO] Base Qwen model loaded successfully!")


    def predict_food_name(self, pil_image: Image.Image, yolo_hint: Optional[str] = None) -> str:
        """
        음식 이미지를 보고 음식명 추론
        
        Args:
            pil_image: 입력 이미지
            yolo_hint: YOLOv3가 탐지한 클래스명 (힌트로 사용)
        """
        # YOLOv3 힌트를 포함한 프롬프트 생성
        hint_text = ""
        if yolo_hint:
            hint_text = f"\n\n참고: 이 음식은 '{yolo_hint}'일 가능성이 있습니다."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {
                        "type": "text",
                        "text": (
                            "한국 음식 이미지를 분류합니다. 음식 이름을 한국어로만 출력하세요.\\n\\n"
                            "중요: 절대로 영어 단어를 사용하지 마세요!\\n"
                            "예: 'omelette' (X) → '오믈렛' (O)\\n\\n"
                            "한국 음식 예시:\\n"
                            "- 계란말이: 계란으로 만든 네모난 한국식 계란 요리 (돌돌 말려있음)\\n"
                            "- 김밥: 김으로 싼 밥\\n"
                            "- 불고기: 양념한 고기 구이\\n"
                            "- 떡볶이: 빨간 국물에 떡\\n\\n"
                            "주의사항:\\n"
                            "- 계란말이 ≠ 오믈렛 (계란말이는 네모나고 말려있음)\\n"
                            "- 반드시 한국어로만 답변\\n"
                            "- 한 단어로만 출력\\n"
                            f"{hint_text}\\n\\n"
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
        
        lower_name = food_name.lower().strip()
        
        if lower_name in translations:
            return translations[lower_name]
        
        for eng, kor in translations.items():
            if eng in lower_name:
                return kor
        
        return food_name

    def estimate_nutrition_llm(self, food_name: str) -> dict:
        """순수 LLM 기반 영양 추론"""
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


# 전역 객체 초기화
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
    model_name=os.getenv(
        "FOODY_EMBED_MODEL",
        "jhgan/ko-sroberta-multitask"
    )
)

food_collection = chroma_client.get_or_create_collection(
    name="food_nutrition",
    embedding_function=ko_embedding,
)


def build_chroma_from_db():
    """서버 시작 시, foods 테이블 내용을 Chroma에 인덱싱"""
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

        food_collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
        )
        print(f"[INFO] Indexed {end}/{total} foods into Chroma...")

    print("[INFO] Finished building Chroma index.")


def rag_estimate_nutrition(food_name: str, top_k: int = 3) -> Optional[dict]:
    """Chroma를 사용해서 유사한 음식들의 영양정보를 가져와 평균"""
    result = food_collection.query(
        query_texts=[food_name],
        n_results=top_k,
    )

    metadatas = result.get("metadatas", [[]])[0]
    if not metadatas:
        return None

    n = len(metadatas)
    sum_kcal = sum(m.get("kcal", 0.0) for m in metadatas)
    sum_carb = sum(m.get("carb", 0.0) for m in metadatas)
    sum_protein = sum(m.get("protein", 0.0) for m in metadatas)
    sum_fat = sum(m.get("fat", 0.0) for m in metadatas)
    sum_sugar = sum(m.get("sugar", 0.0) for m in metadatas)
    sum_natrium = sum(m.get("natrium", 0.0) for m in metadatas)

    best = metadatas[0]
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
app = FastAPI(title="Foody - YOLOv3 + Qwen2.5-VL Analyzer API v2")


@app.on_event("startup")
def on_startup():
    """서버 시작 시 1번만 실행"""
    build_chroma_from_db()



@app.post("/api/vlm/food", response_model=FoodResponse)
async def predict_food(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    YOLOv3 기반 음식 인식:
    1단계: YOLOv3로 음식 객체 탐지 및 분류
    2단계: YOLOv3 분류 결과로 DB/RAG에서 영양소 조회
    
    여러 음식이 탐지되면 가장 confidence가 높은 것만 반환
    """
    # 1) 이미지 체크
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드 가능합니다.")

    # 2) 이미지 로드
    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 열 수 없습니다.")

    # 3) YOLOv3로 음식 객체 탐지 및 분류
    print("[INFO] YOLOv3 food detection and classification...")
    detections = yolo_detector.detect_foods(pil_image)
    
    if not detections:
        # YOLOv3가 탐지 못하면 전체 이미지로 VLM 분류 (fallback)
        print("[WARN] No food detected by YOLOv3. Falling back to VLM on full image.")
        food_name = qwen.predict_food_name(pil_image)
        yolo_confidence = None
    else:
        # 가장 confidence가 높은 탐지 결과 사용
        best_detection = max(detections, key=lambda x: x["confidence"])
        food_name = best_detection["class_name"]  # YOLOv3 분류 결과를 그대로 사용
        yolo_confidence = best_detection["confidence"]
        
        print(f"[INFO] YOLOv3 detected: {food_name} (confidence={yolo_confidence:.3f})")

    
    normalized_name = food_name.replace(" ", "")
    print(f"[INFO] Food name: {food_name} (normalized: {normalized_name})")

    # 4) RDB에서 정확히 같은 이름이 있는지 체크
    food = (
        db.query(Foods)
        .filter(func.replace(Foods.name, " ", "") == normalized_name)
        .first()
    )

    if food:
        print(f"[INFO] Found exact match in DB: {food.name}")
        response_name = food.name
    else:
        response_name = food_name

    # 5) RAG로 Chroma에서 유사 음식들 검색
    est = rag_estimate_nutrition(food_name)

    if est:
        print(f"[INFO] Estimated via Chroma RAG: {food_name}")
    else:
        if food:
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
        confidence=yolo_confidence,
    )


@app.post("/api/vlm/food/multiple", response_model=MultipleFoodResponse)
async def predict_multiple_foods(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    한 이미지에서 여러 음식을 동시에 인식 (YOLOv3 분류 직접 사용)
    
    Returns:
        모든 탐지된 음식들의 리스트
    """
    # 1) 이미지 체크
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드 가능합니다.")

    # 2) 이미지 로드
    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 열 수 없습니다.")

    # 3) YOLOv3로 음식 객체 탐지 및 분류
    print("[INFO] YOLOv3 multiple food detection and classification...")
    detections = yolo_detector.detect_foods(pil_image)
    
    if not detections:
        raise HTTPException(404, "이미지에서 음식을 찾을 수 없습니다.")
    
    # 4) 각 탐지된 음식에 대해 영양소 추출
    results = []
    
    for idx, detection in enumerate(detections):
        food_name = detection["class_name"]  # YOLOv3 분류 결과 직접 사용
        yolo_confidence = detection["confidence"]
        
        print(f"[INFO] Processing food {idx+1}/{len(detections)}: {food_name} (conf={yolo_confidence:.3f})")
        
        normalized_name = food_name.replace(" ", "")
        
        # RDB 조회
        food = (
            db.query(Foods)
            .filter(func.replace(Foods.name, " ", "") == normalized_name)
            .first()
        )
        
        response_name = food.name if food else food_name
        
        # 영양소 추정
        est = rag_estimate_nutrition(food_name)
        
        if not est:
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
            else:
                est = qwen.estimate_nutrition_llm(food_name)
        
        results.append(FoodResponse(
            name=response_name,
            standard=est["standard"],
            kcal=est["kcal"],
            carb=est["carb"],
            protein=est["protein"],
            fat=est["fat"],
            sugar=est["sugar"],
            natrium=est["natrium"],
            confidence=yolo_confidence,
        ))
    
    return MultipleFoodResponse(
        foods=results,
        total_count=len(results)
    )
