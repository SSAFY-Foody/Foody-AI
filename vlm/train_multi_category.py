import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# =========================
# 설정
# =========================
# 학습할 음식 카테고리 및 이미지 개수
FOOD_CATEGORIES = {
    "계란말이": {
        "path": r"c:\Users\YJK\Documents\pjt\Foody-AI\vlm\dataset\전\전\계란말이",
        "num_images": 20,  # 전체 사용 (가장 중요)
    }
}

# 모델 설정
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./qwen_jeon_finetuned"

# 이어서 학습 설정
RESUME_FROM_CHECKPOINT = "./qwen_jeon_finetuned" # 이어서 학습하려면 경로 지정 (예: "./qwen_jeon_finetuned")

# LoRA 설정
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 학습 설정
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
NUM_EPOCHS = 15 # GPU 메모리 절약을 위해 에포크 조정
GRADIENT_ACCUMULATION_STEPS = 8

# 검증 비율
VAL_RATIO = 0.15  # 전체 데이터의 15%를 검증용으로


# =========================
# 다중 카테고리 데이터셋
# =========================
class MultiCategoryFoodDataset(Dataset):
    """여러 음식 카테고리를 학습하는 데이터셋"""
    
    def __init__(self, image_label_pairs: List[Tuple[str, str]], processor):
        """
        Args:
            image_label_pairs: [(image_path, food_name), ...] 리스트
            processor: Qwen processor
        """
        self.image_label_pairs = image_label_pairs
        self.processor = processor
        
    def __len__(self):
        return len(self.image_label_pairs)
    
    def __getitem__(self, idx):
        image_path, food_name = self.image_label_pairs[idx]
        
        # 이미지 로드
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # 프롬프트 구성 - 계란말이에 특별히 강조
        if "계란말이" in food_name:
            prompt_text = (
                "한국 음식 이미지를 분류합니다. 음식 이름을 한국어로만 출력하세요.\n\n"
                "중요: 이 음식은 '계란말이'입니다!\n"
                "계란말이 특징:\n"
                "- 네모난 모양으로 돌돌 말려있음\n"
                "- 속에 재료가 들어있음 (당근, 파, 양파 등)\n"
                "- 오믈렛과 다름 (오믈렛은 반달 모양)\n\n"
                "절대로 'omelette'라고 하지 마세요!\n"
                "정답: 계란말이\n\n"
                "음식 이름을 한 단어로만 출력하세요."
            )
        else:
            prompt_text = (
                "한국 음식 이미지를 분류합니다. 음식 이름을 한국어로만 출력하세요.\n\n"
                "한국 음식 예시:\n"
                f"- {food_name}\n"
                "- 계란말이: 네모나고 말린 계란 요리\n"
                "- 파전: 파가 들어간 전\n"
                "- 김치전: 김치가 들어간 전\n\n"
                "절대로 영어를 사용하지 마세요!\n"
                "음식 이름을 한 단어로만 출력하세요."
            )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": food_name}
                ]
            }
        ]
        
        # 프로세서로 인코딩
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # 배치 차원 제거
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs


# =========================
# 데이터 준비
# =========================
def prepare_multi_category_data(food_categories: Dict, val_ratio: float = 0.15):
    """여러 카테고리의 데이터를 수집하고 train/val로 분할"""
    
    all_image_label_pairs = []
    
    print(f"[INFO] 데이터 수집 중...")
    for food_name, config in food_categories.items():
        folder_path = config["path"]
        max_images = config["num_images"]
        
        if not os.path.exists(folder_path):
            print(f"[WARN] 폴더가 존재하지 않음: {folder_path}")
            continue
        
        # 이미지 파일 찾기
        image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        images = [
            str(p) for p in Path(folder_path).iterdir()
            if p.suffix in image_extensions
        ]
        
        # 랜덤 샘플링
        random.shuffle(images)
        selected_images = images[:max_images]
        
        # (image_path, food_name) 쌍으로 저장
        for img_path in selected_images:
            all_image_label_pairs.append((img_path, food_name))
        
        print(f"  - {food_name}: {len(selected_images)}개 이미지")
    
    print(f"[INFO] 전체 데이터: {len(all_image_label_pairs)}개")
    
    # 랜덤 셔플
    random.shuffle(all_image_label_pairs)
    
    # Train/Val 분할
    val_size = int(len(all_image_label_pairs) * val_ratio)
    train_size = len(all_image_label_pairs) - val_size
    
    train_pairs = all_image_label_pairs[:train_size]
    val_pairs = all_image_label_pairs[train_size:]
    
    print(f"[INFO] Train: {len(train_pairs)}개, Val: {len(val_pairs)}개")
    
    # 각 카테고리별 분포 출력
    print(f"\n[INFO] Train 데이터 분포:")
    for food_name in food_categories.keys():
        count = sum(1 for _, name in train_pairs if name == food_name)
        print(f"  - {food_name}: {count}개")
    
    return train_pairs, val_pairs


# =========================
# 데이터 콜레이터
# =========================
class DataCollatorForVisionSeq2Seq:
    """배치 데이터를 적절히 패딩하는 콜레이터"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        batch = {}
        
        # input_ids 패딩
        input_ids = [f["input_ids"] for f in features]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        batch["input_ids"] = input_ids_padded
        
        # attention_mask 생성
        attention_mask = (input_ids_padded != self.processor.tokenizer.pad_token_id).long()
        batch["attention_mask"] = attention_mask
        
        # labels 패딩
        labels = [f["labels"] for f in features]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        batch["labels"] = labels_padded
        
        # pixel_values - concatenate
        if "pixel_values" in features[0]:
            pixel_values = [f["pixel_values"] for f in features]
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        
        # image_grid_thw - stack
        if "image_grid_thw" in features[0]:
            image_grid_thw = [f["image_grid_thw"] for f in features]
            batch["image_grid_thw"] = torch.stack(image_grid_thw, dim=0)
        
        return batch


# =========================
# 메인 학습 함수
# =========================
def main():
    print("=" * 60)
    print("다중 카테고리 음식 인식 모델 Fine-tuning")
    print("=" * 60)
    
    # 1. 데이터 준비
    print("\n[1/5] 데이터 준비 중...")
    train_pairs, val_pairs = prepare_multi_category_data(FOOD_CATEGORIES, VAL_RATIO)
    
    # 2. 프로세서 및 모델 로드
    print("\n[2/5] 모델 및 프로세서 로드 중...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU가 필요합니다!")
    
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] CUDA Version: {torch.version.cuda}")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    # 이어서 학습 처리
    if RESUME_FROM_CHECKPOINT:
        if os.path.exists(os.path.join(RESUME_FROM_CHECKPOINT, "adapter_config.json")):
            print(f"[INFO] LoRA 어댑터 로드: {RESUME_FROM_CHECKPOINT}")
            model = AutoModelForVision2Seq.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, RESUME_FROM_CHECKPOINT)
            
            # LoRA 파라미터 학습 활성화
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
            
            is_lora_loaded = True
        else:
            print(f"[INFO] 기본 모델 로드: {MODEL_NAME}")
            model = AutoModelForVision2Seq.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            is_lora_loaded = False
    else:
        print(f"[INFO] 기본 모델 로드: {MODEL_NAME}")
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        is_lora_loaded = False
    
    # 3. LoRA 설정
    print("\n[3/5] LoRA 설정 중...")
    
    if not is_lora_loaded:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
    
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    
    # 4. 데이터셋 생성
    print("\n[4/5] 데이터셋 생성 중...")
    train_dataset = MultiCategoryFoodDataset(train_pairs, processor)
    val_dataset = MultiCategoryFoodDataset(val_pairs, processor)
    
    data_collator = DataCollatorForVisionSeq2Seq(processor)
    
    # 5. 학습 설정
    print("\n[5/5] 학습 시작...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 학습 시작
    print("\n" + "=" * 60)
    print("학습 시작!")
    print("=" * 60)
    
    trainer.train(resume_from_checkpoint=None)
    
    # 모델 저장
    print("\n[완료] 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ 학습 완료! 모델이 {OUTPUT_DIR}에 저장되었습니다.")
    print("\n사용 방법:")
    print("1. app.py에서 모델 경로를 변경:")
    print(f'   lora_path = "{OUTPUT_DIR}"')
    print("2. 서버 재시작 후 계란말이 및 전 카테고리 인식 성능 확인")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    
    main()
