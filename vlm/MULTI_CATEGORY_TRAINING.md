# 다중 카테고리 학습 가이드

## 목적
계란말이를 오믈렛으로 잘못 인식하는 문제를 해결하기 위해, "전" 카테고리의 여러 음식을 함께 학습하여 계란말이 인식 정확도를 향상시킵니다.

## 학습 데이터
- ✅ **계란말이**: 20개 이미지 (전체 사용)
- ✅ **파전**: 30개 이미지
- ✅ **김치전**: 19개 이미지
- ✅ **호박전**: 15개 이미지
- ✅ **감자전**: 13개 이미지

**총 97개 이미지** (Train: 약 82개, Val: 약 15개)

## 주요 특징

### 1. 계란말이 특화 프롬프트
계란말이 학습 시 다음 내용을 강조:
```
"이 음식은 '계란말이'입니다!"
- 네모난 모양으로 돌돌 말려있음
- 속에 재료가 들어있음
- 오믈렛과 다름 (오믈렛은 반달 모양)
절대로 'omelette'라고 하지 마세요!
```

### 2. 다중 카테고리 대조 학습
여러 "전" 종류를 함께 학습하여 각 음식의 특징을 더 잘 구분

### 3. 메모리 최적화
- Batch size: 1
- Gradient accumulation: 8
- GPU 메모리: 12GB (RTX 5070) 최적화

## 사용 방법

### 1. 학습 실행
```bash
cd c:\Users\YJK\Documents\pjt\Foody-AI\vlm
python train_multi_category.py
```

### 2. 학습 진행 상황
- **예상 시간**: 약 10-15분
- **에포크**: 5 에포크
- **체크포인트**: `./qwen_jeon_finetuned/checkpoint-*`에 저장
- **최종 모델**: `./qwen_jeon_finetuned/`에 저장

### 3. 모델 적용
학습 완료 후 `app.py` 수정:

```python
# app.py의 QwenClient.__init__() 메서드에서
lora_path = "./qwen_jeon_finetuned"  # 변경
```

### 4. 서버 재시작
```bash
# FastAPI 서버 재시작
python app.py
```

## 설정 조정

### 더 많은 이미지 사용
`train_multi_category.py`에서 `num_images` 값 증가:
```python
FOOD_CATEGORIES = {
    "계란말이": {
        "path": r"...",
        "num_images": 20,  # ← 여기 수정
    },
    # ...
}
```

### 이어서 학습
이전 모델에서 이어서 학습하려면:
```python
RESUME_FROM_CHECKPOINT = "./qwen_jeon_finetuned"
```

### 추가 카테고리
다른 음식도 추가 가능:
```python
FOOD_CATEGORIES = {
    # 기존 카테고리...
    "계란후라이": {
        "path": r"c:\Users\YJK\Documents\pjt\Foody-AI\vlm\dataset\전\전\계란후라이",
        "num_images": 7,
    },
}
```

## 학습 후 테스트

### 1. 계란말이 이미지로 테스트
- 계란말이 사진을 업로드
- "계란말이"로 인식되는지 확인

### 2. 오믈렛 이미지로 테스트 (선택)
- 오믈렛 사진을 업로드
- "오믈렛"으로 인식되는지 확인

### 3. 다른 전 종류 테스트
- 파전, 김치전, 호박전 등
- 각각 올바르게 인식되는지 확인

## 문제 해결

### GPU 메모리 부족
```python
# BATCH_SIZE는 이미 1로 최소화됨
# GRADIENT_ACCUMULATION_STEPS를 증가시켜도 됨
GRADIENT_ACCUMULATION_STEPS = 16  # 8 → 16으로
```

### 특정 카테고리만 학습
필요없는 카테고리를 FOOD_CATEGORIES에서 제거:
```python
FOOD_CATEGORIES = {
    "계란말이": {
        "path": r"...",
        "num_images": 20,
    },
    # 파전, 김치전 등은 주석 처리하거나 삭제
}
```

### 학습 시간 단축
```python
NUM_EPOCHS = 3  # 5 → 3으로 감소
```

## 추가 개선 방법

### 1. 오믈렛 negative 샘플 추가
```python
# 오믈렛 이미지 수집 후 추가
"오믈렛": {
    "path": r"c:\Users\YJK\Documents\pjt\Foody-AI\vlm\오믈렛_이미지들",
    "num_images": 20,
},
```

### 2. 데이터 증강 (Data Augmentation)
- 이미지 회전, 밝기 조절, 크롭 등
- Pillow나 albumentations 라이브러리 사용

### 3. 하이퍼파라미터 튜닝
- Learning rate: 1e-4 ~ 5e-4 실험
- LoRA rank: 8 ~ 16 실험
- Epochs: 3 ~ 7 실험

## 성능 비교

| 항목 | 이전 모델 | 현재 모델 (다중 카테고리) |
|------|-----------|--------------------------|
| 학습 데이터 | 계란말이 100개 | 전 카테고리 97개 |
| 계란말이 강조 | ❌ | ✅ 특화 프롬프트 |
| 대조 학습 | ❌ | ✅ 여러 전 종류 구분 |
| 예상 정확도 | 중간 | **높음** |

## 기대 효과

1. **계란말이 인식 정확도 향상** 🎯
2. **오믈렛과의 구분 명확화** ✅
3. **전 카테고리 전반적 인식 향상** 📈
4. **한국 음식 특화 모델** 🇰🇷
