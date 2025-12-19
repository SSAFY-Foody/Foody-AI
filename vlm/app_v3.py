# app.py
import io
import os
import re
import json
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# Response Models
# =========================================================
class BoxInfo(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls_id: int
    cls_name: str


class FoodNameItem(BaseModel):
    name: str
    box: Optional[BoxInfo] = None  # 단일 전체이미지 fallback 시 None 가능


class SingleFoodNameResponse(BaseModel):
    name: str


class MultiFoodNameResponse(BaseModel):
    foods: List[FoodNameItem]


# =========================================================
# Qwen Client (Food Name Classifier)
# =========================================================
class QwenClient:
    def __init__(self):
        model_id = os.getenv("FOODY_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[INFO] Loading VLM: {model_id} (dtype={torch_dtype}) ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        print("[INFO] VLM loaded successfully.")

    def _is_valid_food_name(self, s: str) -> bool:
        if not s:
            return False
        s = s.strip()
        if s in {"-", "—", "_", "?", "없음", "모름", "알수없음", "알 수 없음", "unknown"}:
            return False
        # 한국어 한 글자 이상 포함
        if not re.search(r"[가-힣]", s):
            return False
        return True

    def _post_process_food_name(self, food_name: str) -> str:
        # 필요시 영어 -> 한국어 보정 (간단 버전)
        translations = {
            "omelette": "오믈렛",
            "omelet": "오믈렛",
            "eggroll": "계란말이",
            "egg roll": "계란말이",
            "kimbap": "김밥",
            "kimchi": "김치",
            "ramen": "라면",
            "rice": "밥",
            "salad": "샐러드",
        }
        lower_name = food_name.lower().strip()
        if lower_name in translations:
            return translations[lower_name]
        for eng, kor in translations.items():
            if eng in lower_name:
                return kor
        return food_name

    def _generate_raw(self, pil_image: Image.Image, prompt: str, max_new_tokens: int = 32) -> str:
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        text = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        return text

    def predict_food_name(self, pil_image: Image.Image) -> str:
        # 1차 프롬프트
        prompt1 = (
            "너는 한국 음식 이미지 분류기다.\n"
            "이미지에서 '가장 중심이 되는 음식 1개'의 이름만 한국어로 출력해라.\n\n"
            "출력 규칙(매우 중요):\n"
            "1) 한국어 음식명만 출력\n"
            "2) 조사/문장/설명 금지 (예: '입니다', '같아요', '.' 금지)\n"
            "3) 따옴표/괄호/슬래시/이모지 금지\n"
            "4) 공백 없이 음식명만 출력 (최대 12자)\n\n"
            "예시:\n"
            "김밥\n"
            "계란말이\n"
            "떡볶이\n\n"
            "중요: '-' '?' '없음' 출력 금지. 반드시 가장 유사한 음식명 1개 출력.\n\n"
            "정답(음식명만):"
        )

        out = self._generate_raw(pil_image, prompt1, max_new_tokens=20)

        # 모델이 문장으로 말할 때 대비: 첫 토큰/라인 중심으로 정리
        out = out.strip().splitlines()[0].strip()
        out = out.split()[0].strip()
        out = re.sub(r"[\"'()\[\]{}<>]", "", out).strip()
        out = self._post_process_food_name(out)

        if self._is_valid_food_name(out):
            return out

        # 2차 강제
        prompt2 = (
            "너는 한국 음식 이미지 분류기다.\n"
            "모르겠어도 '-', '?', '없음'을 출력하지 마라.\n"
            "가장 유사한 한국 음식명 1개를 반드시 출력하라.\n"
            "음식명만 출력(설명 금지).\n"
            "정답:"
        )

        out2 = self._generate_raw(pil_image, prompt2, max_new_tokens=20)
        out2 = out2.strip().splitlines()[0].strip()
        out2 = out2.split()[0].strip()
        out2 = re.sub(r"[\"'()\[\]{}<>]", "", out2).strip()
        out2 = self._post_process_food_name(out2)

        if self._is_valid_food_name(out2):
            return out2

        print(f"[WARN] Invalid VLM outputs: '{out}' / '{out2}' -> fallback")
        return "음식"


# =========================================================
# YOLO Detector (Detection -> boxes)
# =========================================================
class YOLODetector:
    def __init__(self):
        self.enabled = False
        self.model = None

        self.weights = os.getenv("FOODY_YOLO_WEIGHTS", "").strip()
        self.conf = float(os.getenv("FOODY_YOLO_CONF", "0.35"))
        self.iou = float(os.getenv("FOODY_YOLO_IOU", "0.45"))
        self.max_det = int(os.getenv("FOODY_YOLO_MAX_DET", "10"))

        # 박스 필터(너무 작은 반찬 조각 등 제거용)
        self.min_area_ratio = float(os.getenv("FOODY_YOLO_MIN_AREA_RATIO", "0.01"))  # 전체 이미지 대비 1%

        if not self.weights:
            print("[WARN] FOODY_YOLO_WEIGHTS not set. Multi-food detection will fallback to single image.")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            self.enabled = True
            print(f"[INFO] YOLO loaded: {self.weights}")
        except Exception as e:
            print(f"[WARN] Failed to load YOLO. Fallback to single image. err={e}")

    def detect_boxes(self, pil_image: Image.Image) -> List[BoxInfo]:
        if not self.enabled:
            return []

        w, h = pil_image.size
        img_area = float(w * h)

        results = self.model.predict(
            source=pil_image,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )

        if not results:
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        names = getattr(self.model, "names", None) or {}

        out: List[BoxInfo] = []
        for b in r0.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item()) if b.conf is not None else 0.0
            cls_id = int(b.cls[0].item()) if b.cls is not None else -1
            cls_name = str(names.get(cls_id, cls_id))

            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            bw = max(0, x2i - x1i)
            bh = max(0, y2i - y1i)
            box_area = float(bw * bh)
            if img_area > 0 and (box_area / img_area) < self.min_area_ratio:
                continue

            out.append(
                BoxInfo(
                    x1=x1i, y1=y1i, x2=x2i, y2=y2i,
                    conf=conf, cls_id=cls_id, cls_name=cls_name
                )
            )

        # 화면 순서대로: 위->아래, 좌->우
        out.sort(key=lambda d: (d.y1, d.x1))
        return out

    @staticmethod
    def crop(pil_image: Image.Image, box: BoxInfo, pad: int = 8) -> Image.Image:
        w, h = pil_image.size
        x1 = max(0, box.x1 - pad)
        y1 = max(0, box.y1 - pad)
        x2 = min(w, box.x2 + pad)
        y2 = min(h, box.y2 + pad)
        return pil_image.crop((x1, y1, x2, y2))


# =========================================================
# App
# =========================================================
app = FastAPI(title="Foody - Multi Food Classification API (YOLO -> Crop -> Qwen)")

qwen = QwenClient()
yolo = YOLODetector()


def load_image_or_400(upload: UploadFile) -> Image.Image:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    try:
        content = upload.file.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        return pil_image
    except Exception:
        raise HTTPException(status_code=400, detail="이미지를 열 수 없습니다.")


@app.post("/api/vlm/food", response_model=SingleFoodNameResponse)
async def predict_single_food_name(image: UploadFile = File(...)):
    pil_image = load_image_or_400(image)
    name = qwen.predict_food_name(pil_image)
    return SingleFoodNameResponse(name=name)


@app.post("/api/vlm/food", response_model=MultiFoodNameResponse)
async def predict_multi_food_names(image: UploadFile = File(...)):
    pil_image = load_image_or_400(image)

    print("\n" + "=" * 80)
    print("[INFO] Multi-food pipeline start: YOLO -> Crop -> Qwen")

    boxes = yolo.detect_boxes(pil_image)
    print(f"[INFO] YOLO boxes: {len(boxes)}")

    # YOLO 실패 시: 전체 이미지로 1개만
    if not boxes:
        print("[WARN] YOLO returned 0 boxes. Fallback to single-food classification.")
        name = qwen.predict_food_name(pil_image)
        print("=" * 80 + "\n")
        return MultiFoodNameResponse(foods=[FoodNameItem(name=name, box=None)])

    # 박스별 분류
    foods: List[FoodNameItem] = []
    seen = set()

    for idx, box in enumerate(boxes, start=1):
        crop_img = yolo.crop(pil_image, box, pad=8)

        print(
            f"[DEBUG] Box#{idx} conf={box.conf:.3f} cls={box.cls_name} "
            f"xyxy=({box.x1},{box.y1},{box.x2},{box.y2})"
        )

        name = qwen.predict_food_name(crop_img)
        key = name.replace(" ", "")

        # 중복 제거(같은 음식 박스가 여러 개 잡히는 경우)
        if key in seen:
            print(f"[INFO] Skip duplicate: {name}")
            continue
        seen.add(key)

        foods.append(FoodNameItem(name=name, box=box))

        # 과도한 박스 제한(서비스 안정)
        if len(foods) >= 8:
            print("[WARN] Too many foods. Cut to 8 items.")
            break

    print(f"[INFO] Final foods: {[f.name for f in foods]}")
    print("=" * 80 + "\n")

    return MultiFoodNameResponse(foods=foods)
