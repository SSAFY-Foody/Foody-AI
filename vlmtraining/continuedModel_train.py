#!/usr/bin/env python3
"""
기존에 있던 LoRA 어댑터 모델을 불러와서 한국 음식 이미지 분류기 태스크로 추가 학습코드.
LoRA 어댑터만 불러와서 학습을 진행하므로, optimizer/scheduler 상태는 초기화됨.
python ./continuedModel_train.py --load_dir ./trained_models/qwen25_v4 --output_dir ./trained_models/qwen25_v5 --train_csv ./train_csvs/train_3.csv


python continuedModel_train.py \
  --load_dir ./trained_models/qwen25_v8 \
  --output_dir ./trained_models/qwen25_v9 \
  --train_csv ./train_csvs/train_3.csv \
  --epochs 5

"""

import os
import re
import csv
import json
import glob
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

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

from peft import PeftModel, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen25-train-v5")


# ==================== Config ====================
@dataclass
class TrainCfg:
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    load_dir: str = "./trained_models/qwen25_v7"
    output_dir: str = "./trained_models/qwen25_v8"
    train_csv: str = "./train_csvs/train_3.csv"

    train_image_size: Tuple[int, int] = (224, 224)
    max_length: int = 1536  # 긴 프롬프트 대응

    train_prompt: str = (
        "너는 한국 음식 이미지 분류기다.\n"
        "이미지에서 가장 중심이 되는 음식 1개 이름만 한국어로 출력해라.\n"
        "규칙: 설명/조사/문장/기호/이모지/공백 금지. 음식명만(최대 12자).\n"
        "모르겠어도 '-', '?', '없음' 금지. 가장 유사한 음식명 1개 출력.\n"
        "정답(음식명만):"
    )

    # Training
    num_train_epochs: float = 12.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 200

    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2

    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"

    use_4bit: bool = True
    dataloader_num_workers: int = 0


# ==================== Utilities ====================
def sanitize_path(p: str) -> str:
    return (p or "").strip().strip('"').strip("'")


def find_latest_checkpoint(root_dir: str) -> Optional[str]:
    ckpts = sorted(
        glob.glob(os.path.join(root_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else -1,
    )
    return ckpts[-1] if ckpts else None


def resolve_adapter_dir(root_dir: str) -> str:
    """Return directory that contains adapter_config.json (final > latest checkpoint > root)."""
    root_dir = sanitize_path(root_dir)

    final_dir = os.path.join(root_dir, "final")
    if os.path.isfile(os.path.join(final_dir, "adapter_config.json")):
        return final_dir

    latest = find_latest_checkpoint(root_dir)
    if latest and os.path.isfile(os.path.join(latest, "adapter_config.json")):
        return latest

    if os.path.isfile(os.path.join(root_dir, "adapter_config.json")):
        return root_dir

    raise FileNotFoundError(f"adapter_config.json not found under: {root_dir}")


def read_trainer_state_epoch(ckpt_dir: str) -> Optional[float]:
    p = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            st = json.load(f)
        ep = st.get("epoch", None)
        return float(ep) if ep is not None else None
    except Exception:
        return None


def normalize_label(label: str) -> str:
    """Make labels match inference rules: Korean only-ish, no spaces, max 12 chars."""
    s = str(label).strip()
    s = s.replace("\u200b", "")
    s = s.replace(" ", "")
    s = s.split("\n")[0].strip()
    # remove punctuation/quotes/brackets
    s = re.sub(r"[\"'()\[\]{}<>/\\]", "", s).strip()
    if len(s) > 12:
        s = s[:12]
    return s


def load_image_rgb(path: str, size: Tuple[int, int]) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning(f"[IMG] failed to load {path}: {e} -> blank")
        img = Image.new("RGB", size, color="white")
    return img.resize(size, resample=Image.BICUBIC)


# ==================== Dataset ====================
class KoreanFoodVLMDataset(Dataset):
    """
    Builds 2 sequences:
      - prompt-only: user(image+prompt) with add_generation_prompt=True  -> used to compute prompt length
      - full: user(image+prompt) + assistant(label) -> actual input/labels
    Handles too-long sequences by keeping the tail (so label tokens remain).
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        image = load_image_rgb(item["image_path"], self.image_size)
        label = normalize_label(item["label"])

        # prompt-only (to compute prompt length)
        prompt_msgs = [{
            "role": "user",
            "content": [{"type": "image", "image": image},
                        {"type": "text", "text": self.prompt}],
        }]

        # full conversation
        full_msgs = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image},
                            {"type": "text", "text": self.prompt}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ]

        prompt_inputs = self.processor.apply_chat_template(
            prompt_msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = self.processor.apply_chat_template(
            full_msgs,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)

        # If too long, keep tail so label survives
        if input_ids.shape[0] > self.max_length:
            overflow = int(input_ids.shape[0] - self.max_length)

            input_ids = input_ids[overflow:]
            attention_mask = attention_mask[overflow:]

            # align prompt length after truncation
            pids = prompt_inputs["input_ids"].squeeze(0)
            if pids.shape[0] > overflow:
                pids = pids[overflow:]
                prompt_len = pids.shape[0]
            else:
                prompt_len = 0
        else:
            prompt_len = prompt_inputs["input_ids"].shape[-1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # only learn assistant answer tokens

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "pixel_values" in full_inputs and full_inputs["pixel_values"] is not None:
            batch["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in full_inputs and full_inputs["image_grid_thw"] is not None:
            batch["image_grid_thw"] = full_inputs["image_grid_thw"].squeeze(0)

        return batch


class QwenVLDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)

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
def load_base_model(cfg: TrainCfg) -> torch.nn.Module:
    if cfg.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForVision2Seq.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            quantization_config=bnb,
            dtype=torch.float16,
        )
    else:
        base = AutoModelForVision2Seq.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            dtype=torch.float16,
        )

    if torch.cuda.is_available():
        base = base.to("cuda")

    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    base = prepare_model_for_kbit_training(base)
    return base


def load_model_with_adapter(cfg: TrainCfg) -> torch.nn.Module:
    adapter_dir = resolve_adapter_dir(cfg.load_dir)
    logger.info(f"[LOAD] adapter_dir = {adapter_dir}")

    base = load_base_model(cfg)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True, local_files_only=True)
    model.print_trainable_parameters()
    return model


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, required=True, help="folder containing final/ or checkpoint-* (LoRA adapter)")
    parser.add_argument("--output_dir", type=str, default="./qwen25_v2")
    parser.add_argument("--train_csv", type=str, default="train_2.csv")
    parser.add_argument("--resume_trainer_state", action="store_true", help="resume optimizer/scheduler state from latest checkpoint-*")
    parser.add_argument("--epochs", type=float, default=2.0, help="num_train_epochs")
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    cfg = TrainCfg(
        load_dir=sanitize_path(args.load_dir),
        output_dir=sanitize_path(args.output_dir),
        train_csv=sanitize_path(args.train_csv),
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        train_image_size=(args.img_size, args.img_size),
    )

    if not os.path.isdir(cfg.load_dir):
        raise RuntimeError(f"[ERROR] load_dir not found: {cfg.load_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Processor: use base model processor (stable & always exists)
    processor = AutoProcessor.from_pretrained(cfg.base_model, trust_remote_code=True)

    model = load_model_with_adapter(cfg)

    dataset = KoreanFoodVLMDataset(
        csv_file=cfg.train_csv,
        processor=processor,
        max_length=cfg.max_length,
        prompt=cfg.train_prompt,
        image_size=cfg.train_image_size,
    )

    collator = QwenVLDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    train_args = TrainingArguments(
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
        args=train_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    if args.resume_trainer_state:
        latest_ckpt = find_latest_checkpoint(cfg.load_dir)
        if latest_ckpt:
            prev_epoch = read_trainer_state_epoch(latest_ckpt)
            logger.info(f"[RESUME] latest_ckpt={latest_ckpt} prev_epoch={prev_epoch} target_epochs={cfg.num_train_epochs}")

            if prev_epoch is not None and prev_epoch >= cfg.num_train_epochs:
                logger.warning(
                    f"[RESUME] prev_epoch({prev_epoch}) >= target_epochs({cfg.num_train_epochs}) -> "
                    f"resume would end immediately. Start adapter-only training (optimizer reset)."
                )
                trainer.train()
            else:
                trainer.train(resume_from_checkpoint=latest_ckpt)
        else:
            logger.warning("[RESUME] no checkpoint-* found; train from adapter weights only")
            trainer.train()
    else:
        logger.info("[CONTINUE] train from adapter weights only (optimizer reset)")
        trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    logger.info(f"[SAVE] final -> {final_dir}")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    with open(os.path.join(final_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.labels, f, ensure_ascii=False, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
