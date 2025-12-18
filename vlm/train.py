import os
import random
from pathlib import Path
from typing import List, Dict
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
# 학습할 이미지 개수 (메모리 절약을 위해 소량만 사용)
NUM_TRAIN_IMAGES = 100  # GPU 메모리와 학습 시간의 균형
NUM_VAL_IMAGES = 20  # 검증 이미지 수

# 이미지 디렉토리
IMAGE_DIR = r"c:\Users\YJK\Documents\pjt\Foody-AI\vlm\21.한국음식\전\전\계란말이"

# 모델 설정
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./qwen_vlm_model"

# 이어서 학습 설정
# None: 처음부터 학습
# "./qwen_gyeranmari_finetuned": fine-tuned 모델에서 이어서 학습
# "./qwen_gyeranmari_finetuned/checkpoint-100": 특정 체크포인트에서 재개
RESUME_FROM_CHECKPOINT = "./qwen_vlm_model"

# LoRA 설정 (메모리 효율적인 학습)
LORA_R = 8  # 메모리 절약을 위해 rank 감소
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 학습 설정
BATCH_SIZE = 1  # GPU 메모리 부족 방지
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8  # 실질적인 배치 크기 = 1 * 8 = 8


# =========================
# 데이터셋 클래스
# =========================
class GyeranmariDataset(Dataset):
    """계란말이 이미지 학습용 데이터셋"""
    
    def __init__(self, image_paths: List[str], processor, food_name: str = "계란말이"):
        self.image_paths = image_paths
        self.processor = processor
        self.food_name = food_name
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # 이미지 로드
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            # 에러 발생 시 다음 이미지 시도
            return self.__getitem__((idx + 1) % len(self))
        
        # 프롬프트 구성 (음식 이름 추론 태스크)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "당신은 한국 음식 이미지 분류기입니다.\n"
                            "절대로 설명하지 말고, 절대로 영어를 섞지 말고,\n"
                            "음식 이름을 한국어 한 단어로만 출력하세요.\n\n"
                            "정확히 한 단어만 출력하세요."
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": self.food_name}
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
        
        # 배치 차원 제거 (DataLoader가 다시 추가함)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # labels 추가 (학습용)
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs


# =========================
# 데이터 준비
# =========================
def prepare_data(image_dir: str, num_train: int, num_val: int):
    """이미지 파일 목록을 train/val로 분할"""
    
    # 모든 이미지 파일 찾기
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    all_images = [
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix in image_extensions
    ]
    
    print(f"[INFO] Found {len(all_images)} images in {image_dir}")
    
    # 랜덤 샘플링
    random.shuffle(all_images)
    
    total_needed = num_train + num_val
    if len(all_images) < total_needed:
        print(f"[WARN] Not enough images. Using all {len(all_images)} images.")
        num_train = int(len(all_images) * 0.8)
        num_val = len(all_images) - num_train
    
    train_images = all_images[:num_train]
    val_images = all_images[num_train:num_train + num_val]
    
    print(f"[INFO] Train: {len(train_images)} images, Val: {len(val_images)} images")
    
    return train_images, val_images


# =========================
# 데이터 콜레이터
# =========================
class DataCollatorForVisionSeq2Seq:
    """배치 데이터를 적절히 패딩하는 콜레이터"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        # input_ids와 labels 패딩
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
        
        # labels 패딩 (-100은 loss 계산에서 무시됨)
        labels = [f["labels"] for f in features]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        batch["labels"] = labels_padded
        
        # pixel_values - Qwen VLM은 가변 크기 이미지를 지원하므로 concatenate
        if "pixel_values" in features[0]:
            pixel_values = [f["pixel_values"] for f in features]
            # 각 이미지의 크기가 다를 수 있으므로 리스트로 유지하거나 concatenate
            # Qwen VLM은 배치 차원에서 concatenate를 기대함
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        
        # image_grid_thw - 각 이미지의 그리드 정보 (반드시 stack 사용)
        if "image_grid_thw" in features[0]:
            image_grid_thw = [f["image_grid_thw"] for f in features]
            # image_grid_thw는 (batch_size, 3) 형태여야 함
            batch["image_grid_thw"] = torch.stack(image_grid_thw, dim=0)
        
        return batch



# =========================
# 메인 학습 함수
# =========================
def main():
    print("=" * 50)
    print("계란말이 인식 모델 Fine-tuning 시작")
    print("=" * 50)
    
    # 1. 데이터 준비
    print("\n[1/5] 데이터 준비 중...")
    train_images, val_images = prepare_data(IMAGE_DIR, NUM_TRAIN_IMAGES, NUM_VAL_IMAGES)
    
    # 2. 프로세서 및 모델 로드
    print("\n[2/5] 모델 및 프로세서 로드 중...")
    
    # GPU 필수 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU가 필요합니다!")
    
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] CUDA Version: {torch.version.cuda}")
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    # 이어서 학습할지 결정
    if RESUME_FROM_CHECKPOINT:
        import os
        # 체크포인트 폴더인지 LoRA 어댑터 폴더인지 확인
        if os.path.exists(os.path.join(RESUME_FROM_CHECKPOINT, "adapter_config.json")):
            # LoRA 어댑터 폴더
            print(f"[INFO] LoRA 어댑터 로드: {RESUME_FROM_CHECKPOINT}")
            print("[INFO] 기본 모델 로드 후 LoRA 어댑터 적용...")
            
            # 기본 모델 로드
            model = AutoModelForVision2Seq.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Gradient checkpointing 활성화
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            
            # LoRA 어댑터 로드
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, RESUME_FROM_CHECKPOINT)
            print("[INFO] LoRA 어댑터 로드 완료!")
            is_lora_loaded = True
            
        elif os.path.exists(os.path.join(RESUME_FROM_CHECKPOINT, "pytorch_model.bin.index.json")) or \
             os.path.exists(os.path.join(RESUME_FROM_CHECKPOINT, "model.safetensors.index.json")):
            # 전체 모델 체크포인트
            print(f"[INFO] 체크포인트에서 모델 로드: {RESUME_FROM_CHECKPOINT}")
            model = AutoModelForVision2Seq.from_pretrained(
                RESUME_FROM_CHECKPOINT,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            is_lora_loaded = hasattr(model, 'peft_config')
        else:
            raise ValueError(
                f"'{RESUME_FROM_CHECKPOINT}'는 유효한 체크포인트가 아닙니다.\n"
                f"LoRA 어댑터(adapter_config.json) 또는 모델 체크포인트가 필요합니다."
            )
    else:
        print(f"[INFO] 기본 모델 로드: {MODEL_NAME}")
        # GPU에서 모델 로드
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        is_lora_loaded = False
    
    # 3. LoRA 설정 (메모리 효율적인 학습)
    print("\n[3/5] LoRA 설정 중...")
    
    if not is_lora_loaded:
        # Gradient checkpointing 활성화 (메모리 절약)
        model.gradient_checkpointing_enable()
        
        # 모델을 양자화 학습에 맞게 준비
        model = prepare_model_for_kbit_training(model)
        
        print("[INFO] 새로운 LoRA 어댑터 추가 중...")
        # LoRA 설정
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention 레이어
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
    else:
        print("[INFO] LoRA가 이미 적용된 모델입니다. 기존 LoRA 사용.")
        print("[INFO] LoRA 파라미터를 학습 가능하도록 설정 중...")
        
        # 로드된 LoRA 파라미터를 학습 가능하도록 설정
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        
        print("[INFO] LoRA 파라미터 학습 활성화 완료!")

    
    
    # 학습 가능한 파라미터 출력
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        print("[INFO] 모델 파라미터 정보를 출력할 수 없습니다.")
    
    # 4. 데이터셋 생성
    print("\n[4/5] 데이터셋 생성 중...")
    train_dataset = GyeranmariDataset(train_images, processor)
    val_dataset = GyeranmariDataset(val_images, processor)
    
    # 데이터 콜레이터
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
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        bf16=True,  # bfloat16 사용 (RTX 5070에 최적화)
        dataloader_num_workers=0,  # Windows에서는 0으로 설정
        remove_unused_columns=False,
        report_to="none",  # wandb 등 사용 안 함
    )
    
    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 학습 시작
    print("\n" + "=" * 50)
    if RESUME_FROM_CHECKPOINT:
        print("이어서 학습 시작!")
        print(f"체크포인트: {RESUME_FROM_CHECKPOINT}")
    else:
        print("새로운 학습 시작!")
    print("=" * 50)
    
    # resume_from_checkpoint는 Trainer의 체크포인트에만 사용
    # LoRA 어댑터는 이미 모델에 로드되었으므로 None으로 설정
    import os
    if RESUME_FROM_CHECKPOINT and os.path.exists(os.path.join(RESUME_FROM_CHECKPOINT, "trainer_state.json")):
        resume_checkpoint = RESUME_FROM_CHECKPOINT
    else:
        resume_checkpoint = None
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # 모델 저장
    print("\n[완료] 모델 저장 중...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ 학습 완료! 모델이 {OUTPUT_DIR}에 저장되었습니다.")
    print("\n사용 방법:")
    print("1. app.py에서 모델 경로를 변경:")
    print(f'   MODEL_NAME = "{OUTPUT_DIR}"')
    print("2. 서버 재시작 후 계란말이 인식 성능 확인")


if __name__ == "__main__":
    # 랜덤 시드 고정 (재현성)
    random.seed(42)
    torch.manual_seed(42)
    
    main()
