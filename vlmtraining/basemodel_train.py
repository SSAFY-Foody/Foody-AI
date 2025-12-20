#!/usr/bin/env python3
"""
Qwen2.5-VL-3B-Instruct LoRA training from scratch (WSL + 12GB VRAM friendly)
- Train CSV: train_2.csv (image_path,label)
- Output: ./qwen25_v1/final
- 핵심: 이미지 토큰 폭발 방지(리사이즈) + apply_chat_template(tokenize=True)로 mismatch 방지
"""

import os
import csv
import json
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen25-kfood-train")


@dataclass
class Config:
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    train_csv: str = "train_2.csv"
    output_dir: str = "./qwen25_v1"

    # 224~256 추천. 448은 토큰 너무 커져서 max_length로 못 잡는 경우가 많음.
    train_image_size: Tuple[int, int] = (224, 224)

    # 텍스트 길이(이미지 토큰 포함). 12GB면 1536~2048 선이 무난.
    max_length: int = 1536

    # 원문 프롬프트 (학습에 영향주지 않음, 별도로 보관용)
    long_prompt: str = (
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

    # 실제 학습에 넣는 프롬프트
    # 원문 그대로 넣으면 터져서, 설명/예시/혼동 주의 규칙 제거한 단순 버전 사용
    train_prompt: str = (
        "너는 한국 음식 이미지 분류기다.\n"
        "이미지에서 가장 중심이 되는 음식 1개 이름만 한국어로 출력해라.\n"
        "규칙: 설명/조사/문장/기호/이모지/공백 금지. 음식명만(최대 12자).\n"
        "모르겠어도 '-', '?', '없음' 금지. 가장 유사한 음식명 1개 출력.\n"
        "정답(음식명만):"
    )

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 8e-5
    warmup_steps: int = 200

    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2

    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"

    # Quant
    use_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    dataloader_num_workers: int = 0  # WSL-safe


cfg = Config()


# ==================== Utils ====================
def normalize_label(label: str) -> str:
    s = str(label).strip()
    s = s.replace("\u200b", "")
    s = s.replace(" ", "")
    s = re.sub(r"[\"'()\[\]{}<>/\\]", "", s)
    s = s.split("\n")[0].strip()
    if len(s) > 12:
        s = s[:12]
    return s


def load_image_rgb(path: str, size: Tuple[int, int]) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e} -> blank")
        img = Image.new("RGB", size, color="white")

    # 학습 시 이미지 크기를 줄여서 이미지 토큰 수를 확 줄임
    img = img.resize(size, resample=Image.BICUBIC)
    return img


# ==================== Dataset ====================
class KoreanFoodVLMDataset(Dataset):
    """
    1) apply_chat_template(tokenize=True)를 사용해서 processor가 텍스트/이미지 토큰 정합성을 직접 맞추게 함
       -> "Mismatch in `image` token count" 류 에러를 피하기 좋음
    2) loss는 정답 토큰에만 걸리도록: prompt_len 계산해서 labels[:prompt_len] = -100
    """

    def __init__(self, csv_file: str, processor: Any, max_length: int, prompt: str, image_size: Tuple[int, int]):
        self.processor = processor
        self.max_length = max_length
        self.prompt = prompt
        self.image_size = image_size

        self.data: List[Dict[str, str]] = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({"image_path": row["image_path"], "label": row["label"]})

        self.labels = sorted(list({x["label"] for x in self.data}))
        logger.info(f"[DATA] {csv_file} | samples={len(self.data)} | labels={len(self.labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        image = load_image_rgb(item["image_path"], self.image_size)
        label = normalize_label(item["label"])

        # (A) prompt-only: user + (image,prompt) with generation prompt
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # (B) full: user(image+prompt) + assistant(label)
        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # squeeze batch dim
        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)

        # 길이 제한: 길면 잘라야 하는데,
        # 오른쪽(끝)에는 정답 토큰이 있어서 "오른쪽이 잘리면 학습이 망가짐"
        # 그래서 길면 '앞을 조금 줄이고' 끝(정답)을 살리는 방식으로 자름.
        if input_ids.shape[0] > self.max_length:
            overflow = input_ids.shape[0] - self.max_length
            input_ids = input_ids[overflow:]
            attention_mask = attention_mask[overflow:]

            # prompt_len도 같은 규칙으로 맞춰주기
            prompt_ids = prompt_inputs["input_ids"].squeeze(0)
            if prompt_ids.shape[0] > overflow:
                prompt_ids = prompt_ids[overflow:]
            else:
                # prompt가 전부 날아간 경우: 최소 0으로
                prompt_ids = prompt_ids[:0]

            prompt_len = prompt_ids.shape[0]
        else:
            prompt_len = prompt_inputs["input_ids"].shape[-1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # qwen2.5-vl multimodal fields
        if "pixel_values" in full_inputs and full_inputs["pixel_values"] is not None:
            batch["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in full_inputs and full_inputs["image_grid_thw"] is not None:
            batch["image_grid_thw"] = full_inputs["image_grid_thw"].squeeze(0)

        return batch


# ==================== Collator ====================
class QwenVLDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(x["input_ids"].shape[0] for x in features)

        def pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
            if x.shape[0] == max_len:
                return x
            pad = torch.full((max_len - x.shape[0],), pad_value, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        batch = {
            "input_ids": torch.stack([pad_1d(f["input_ids"], self.pad_token_id) for f in features]),
            "attention_mask": torch.stack([pad_1d(f["attention_mask"], 0) for f in features]),
            "labels": torch.stack([pad_1d(f["labels"], -100) for f in features]),
        }

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.stack([f["image_grid_thw"] for f in features])

        return batch


# ==================== Load model/processor ====================
def load_processor_and_model(cfg: Config):
    logger.info(f"[LOAD] processor: {cfg.base_model}")
    processor = AutoProcessor.from_pretrained(cfg.base_model, trust_remote_code=True)

    if cfg.use_4bit:
        logger.info("[LOAD] model: 4bit NF4")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            quantization_config=bnb,
            dtype=torch.float16,
        )
    else:
        logger.info("[LOAD] model: fp16")
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            dtype=torch.float16,
        )

    if torch.cuda.is_available():
        model = model.to("cuda")

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    logger.info("[LoRA] attach adapters")
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return processor, model


# ==================== Main ====================
def main():
    os.makedirs(cfg.output_dir, exist_ok=True)

    processor, model = load_processor_and_model(cfg)

    # 학습 안정성을 위해 train_prompt 사용
    dataset = KoreanFoodVLMDataset(
        csv_file=cfg.train_csv,
        processor=processor,
        max_length=cfg.max_length,
        prompt=cfg.train_prompt,
        image_size=cfg.train_image_size,
    )
    collator = QwenVLDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("[TRAIN] start")
    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    logger.info(f"[SAVE] final -> {final_dir}")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    with open(os.path.join(final_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.labels, f, ensure_ascii=False, indent=2)

    # 추론에서 쓰라고 원문 프롬프트도 저장
    with open(os.path.join(final_dir, "canonical_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(cfg.long_prompt)

    logger.info("Done!")


if __name__ == "__main__":
    main()
